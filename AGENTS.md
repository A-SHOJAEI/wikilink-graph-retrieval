# Agent Notes (Repo Local)

## Primary Commands

- `make setup`: create `.venv` and install deps (never install into system Python).
- `make all`: run the default smoke pipeline end-to-end.
- `make all CONFIG=configs/full.yaml`: run full pipeline (will download large dumps; not recommended on small disks).

## Environment Constraints

- Ubuntu PEP 668 compatible: always use `.venv/bin/pip`.
- `ensurepip` may be missing: `.venv` is created with `--without-pip` and bootstrapped via `get-pip.py`.

## Outputs That Must Exist

- `artifacts/results.json`
- `artifacts/report.md`

