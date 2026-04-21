**Dependency:** This PR depends on #47 (`pr3-checkpoint-stderr-tokens`). Please merge #47 first.

The `tests/conftest.py` shared fixtures (`_no_quota`) are introduced in #47. This PR extends conftest with `scripted_stream` + `receiver_tool` for scheduling-specific e2e tests.