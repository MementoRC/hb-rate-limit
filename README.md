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

## Status

**Scaffold** — API per hb-rate-limit API spec v1 lands in v0.1.0. Only `__version__` is exported today.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
