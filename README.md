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

Share one budget across instances with `get_pool_limiter(pool_key, limit)`; the kind and
`safety_margin_pct` (default 0.0; hb_compat shims pass 0.05) are part of the pool's identity.
The `run_paced` / `note_rate_limited` helpers take an optional `clock=` for limiters that can
pause but cannot report `is_paused()`. Tests use `rate_limit.testing.ManualClock` and the
`assert_conforms_*` suites.

---

## Status

v0.2.0: adds `TokenBucketPool` (per-limit `kind="token_bucket"`), `unpause` / `unpause_all`, an
opt-in `max_retry_after_seconds` cap, `safety_margin_pct` on `get_pool_limiter`, and an optional
`clock` for the pacing helpers. v0.1.0 shipped sliding-window pools, the registry, pause/backoff,
helpers and the conformance kit.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
