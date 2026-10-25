# hb-rate-limit

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

**hb-rate-limit** provides stdlib-only async rate limiting primitives (pools, backoff, Retry-After handling) for Hummingbot sub-packages. The Python module is `rate_limit`.

It has no runtime dependencies and is an L0 leaf per ADR 0001 (amendment-1): it never imports hummingbot or any other `hb-*` package.

---

## Installation

```bash
pixi add hb-rate-limit
# or
pip install hb-rate-limit
```

---

## Development

```bash
pixi install
pixi run test
pixi run lint
pixi run format-check
pixi run check  # Full suite
```

---

## Usage

```python
from decimal import Decimal
from rate_limit import LinkedLimitWeightPair, RateLimit, RateLimiter

limiter = RateLimiter(
    [
        RateLimit.per_minute("REQUEST_WEIGHT", 1200),
        RateLimit.per_second(
            "klines", 10, linked_limits=[LinkedLimitWeightPair("REQUEST_WEIGHT", 2)]
        ),
    ],
    limits_pct=Decimal("0.8"),
)

async def fetch() -> None:
    # weights overrides the declared charge per limit id (each >= 1)
    async with limiter.execute_task("klines", weights={"REQUEST_WEIGHT": 5}):
        ...  # send the request

# Non-blocking: 0.0 means reserved, otherwise seconds to wait (nothing recorded)
wait = limiter.try_acquire("klines")

# Transport-agnostic backoff: 429 pauses the id and its dependents, 418 pauses everything
verdict = limiter.report_response("klines", 429, {"Retry-After": "3"})
```

A limit can be a token bucket instead of a sliding window: capacity `limit`, refilling
`limit / time_interval` per second. A connector's `PoolSpec(name, capacity, refill_rate)` maps to
`RateLimit(name, capacity, capacity / refill_rate, kind="token_bucket")`.

```python
bucket = RateLimiter([RateLimit("orders", 20, 10, kind="token_bucket")])  # burst 20, then 2/s
```

Cap an unreasonable server `Retry-After`, and lift a pause early (sleeping acquirers wake and
re-assess; pool accounting is untouched):

```python
limiter = RateLimiter([RateLimit.per_second("feed", 5)], max_retry_after_seconds=120)
limiter.report_response("feed", 429, {"Retry-After": "7200"})  # paused for 120s, not 7200s
limiter.unpause("feed")  # or limiter.unpause_all()
```

### Refunding a charge

`reserve()` waits exactly like `acquire()` but returns a `Charge` receipt. If the request never
went out (signing failed, say), `charge.refund()` hands the capacity back on every pool it
charged. A refund is idempotent and thread-safe; it is skipped for a limit id that has since been
removed (even if re-added) or changed kind or refill rate, and an expired charge gives nothing
back. A token bucket does not track individual charges, so it returns only the part that cannot
have refilled yet, `min(used, cost - refill * elapsed)`, which never over-credits. Waiters are
not woken early: they pick up the capacity when they next re-assess.

Use `refund_on_error=True` only for bodies that fail before the request reaches the venue
(signing, validation). An exception raised after sending (a 429, a response timeout, a parse
error) means the venue already counted the request, so refunding under-throttles; there, use
`reserve()` and refund explicitly.

```python
charge = await limiter.reserve("klines")
try:
    signed = sign(order)
except SigningError:
    charge.refund()  # nothing was sent: give the weight back
    raise
await send(signed)

# or let the context manager do it: an Exception in the body refunds, cancellation does not
async with limiter.execute_task("klines", refund_on_error=True):
    signed = sign(order)  # only pre-send work in here
await send(signed)
```

`limiter.bind("klines").reserve()` does the same with the bound weights.

Share one budget across instances with `get_pool_limiter(pool_key, limit)`; the kind and
`safety_margin_pct` (default 0.0; hb_compat shims pass 0.05) are part of the pool's identity.
The `run_paced` / `note_rate_limited` helpers take an optional `clock=` for limiters that can
pause but cannot report `is_paused()`. Tests use `rate_limit.testing.ManualClock` and the
`assert_conforms_*` suites. Opt-in suites cover features that are not part of the required
protocols; run one only if your limiter claims the feature (each is also a
`conformance_suites` fixture attribute): `assert_conforms_token_bucket(make)` with
`make(limits, clock, sleep)`, `assert_conforms_unpausable(make)` with `make(limits, clock, sleep)`
(driven by `rate_limit.testing.BlockingSleep`), `assert_conforms_retry_after_cap(make)` with
`make(cap, clock, backoff_factory)`, `assert_conforms_kind_change(make)` with
`make(limits, clock, sleep)` (`set_limits` carries usage across a window / token-bucket switch),
and `assert_conforms_refundable(make)` with `make(limits, clock, sleep)` (`reserve()` returns a
charge whose `refund()` frees its capacity at once, exactly once, on every linked pool).
For a limiter bound to one limit id (`limiter.bind("x")`, which also forwards `paused_until()`
and `report_throttled(retry_after, status=)`), `assert_conforms_bound_unpausable(make)` with
`make(limit, clock, sleep)` and `assert_conforms_bound_retry_after_cap(make)` with
`make(cap, clock, backoff_factory)` run the same scenarios; the bound unpausable suite skips the
`pause_all` and multi-id ones and checks the bound `unpause_all()` instead.
`assert_conforms_pausable(make)` returns the tier it ran, `"pool_pause"` (also checks `is_paused()`
and `blocked_for()`) or `"pausable"`; pass `require_pool_pause=True` to fail instead of falling
back. Every `make()` call must return a fresh limiter, not one sharing state with an earlier call.

### Per-key cooldowns (KeyedCooldown)

`KeyedCooldown(limit, time_interval)` keeps one sliding window per hashable key, with the same
inclusive boundary as the limiter's windows. Interactive Brokers allows no identical request within
15 seconds, and at most six requests per (contract, exchange, tickType) per 2 seconds:

```python
from rate_limit import KeyedCooldown

identical = KeyedCooldown(1, 15.0)
per_tick = KeyedCooldown(6, 2.0)

async with limiter.execute_task("ib_messages"):  # the global limit first
    await per_tick.acquire((con_id, exchange, tick_type))
    await identical.acquire(request_key)  # cooldowns last, right before the send
    await send(request)
```

It has the limiter's `acquire` / `try_acquire` / `blocked_for` shape with a key and an optional
weight (`weight > limit` raises `RateLimitWeightError`), plus `reset(key)` / `reset()` and
`len()` for the keys tracked. A key's window is created on its first charge (`blocked_for` never
creates one) and dropped once empty, so memory stays bounded without sweeping every key.
`KeyedCooldown` is independent of `RateLimiter` and not atomic with it: acquire the limiter first
and the cooldowns immediately before sending, so each key's stamp is as close as possible to the
real send.

### hummingbot compatibility (rate_limit.hb_compat)

`rate_limit.hb_compat` is a drop-in for `hummingbot.core.api_throttler` (`data_types`,
`async_throttler_base`, `async_request_context_base`, `async_throttler`) running on `RateLimiter`,
with no core change. It keeps the legacy surface: the mutable `RateLimit`, `_rate_limits` /
`_id_to_limit_map` / `_raw_limits`, keep-existing `add_rate_limits` (one warning per id), the
`max(1, floor)` share scaling, and `paused_until()` returning `None` when never paused.

Intentional behaviour changes:

- a weight that can never fit raises `RateLimitWeightError` where legacy hung
- duplicate linked ids are summed, and the `add_rate_limits` conflict check compares links as a multiset
- windows use the injected monotonic clock (no `time.time`)
- a `time_interval` <= 0 (or non-finite) raises `ValueError`
- no FIFO guarantee among waiters
- `TaskLog` is import-only

---

## Status

v0.4.0: adds `rate_limit.hb_compat`, the hummingbot `AsyncThrottler` drop-in; no core change.

v0.3.0: ships `py.typed`; refund receipts, where `RateLimiter.reserve()` / `BoundLimiter.reserve()`
return a `Charge` with `refund()`, plus `execute_task(..., refund_on_error=True)` and an opt-in
`refundable` conformance suite; `KeyedCooldown` for per-key windows; `BoundLimiter.paused_until()`
and `report_throttled()`, with the bound-limiter conformance suites `bound_unpausable` and
`bound_retry_after_cap`. The kit also gains the opt-in `token_bucket`, `unpausable`,
`retry_after_cap` and `kind_change` suites (plus `BlockingSleep`), closing the v0.2 kit gap.
`assert_conforms_pausable` now returns its `PausableTier` and takes `require_pool_pause`.
`acquire()` is unchanged and everything is additive, except that the `Pool` protocol gains
`refund()`: custom `Pool` implementations must add it.

v0.2.0: adds `TokenBucketPool` (per-limit `kind="token_bucket"`), `unpause` / `unpause_all`, an
opt-in `max_retry_after_seconds` cap, `safety_margin_pct` on `get_pool_limiter`, and an optional
`clock` for the pacing helpers. v0.1.0 shipped sliding-window pools, the registry, pause/backoff,
helpers and the conformance kit.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
