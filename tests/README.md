# Tests

E2E-only test suite for gfmbench-api.

Install dependencies:

```bash
pip install -e ".[test]"
```

## Tiers

| Tier | Marker | What it checks | Network | GPU |
|------|--------|----------------|---------|-----|
| **Smoke** | `smoke` | Full eval pipeline with `MockGFMModel` and local fixture CSVs | No | No |
| **Download** | `download` | Tasks download HuggingFace data into an empty data directory | Yes | No |
| **Heavy** | `heavy` | Real DNABERT2 benchmark on 3 tasks; scores compared to pinned baseline CSV | Yes* | Recommended |

\*Heavy tests use cached data under `GFMBENCH_DATA_ROOT` when available.

### Smoke (`tests/e2e/test_smoke.py`)

Six tests covering supervised single-seq, supervised variant, zero-shot SNV/indel, CSV report pipeline, and graceful handling of missing model methods. Uses tiny local fixtures (`tests/fixtures/data/`), 8 samples per task.

### Download (`tests/e2e/test_download.py`)

Three tests: HF download for `gue_promoter_all` and `var_bench_coding_pathogenicity`, plus idempotent cache reuse for GUE. Starts from an empty temp directory per test. Tasks that need a reference genome (e.g. `songlab_clinvar`) are not covered here.

### Heavy (`tests/e2e/test_heavy.py`)

One regression test: runs `usage_examples/benchmark_runner.py` on three tasks (`gue_promoter_all`, `songlab_clinvar`, `var_bench_coding_pathogenicity`) with 100 samples, linear probe, 1 epoch, then compares metrics to `tests/fixtures/dnabert2_sanity_baseline.csv`.

## Running tests

### PR / default (smoke only)

```bash
pytest tests/ -m "not heavy and not download"
```

Or run smoke directly:

```bash
pytest tests/e2e/test_smoke.py -m smoke
```

### Download tests

Requires network. Tests are **skipped** unless you opt in:

```bash
export RUN_DOWNLOAD_TESTS=1
pytest tests/e2e/test_download.py -m download
```

### Heavy regression

Requires real benchmark data and a GPU-friendly env (DNABERT2 flash-attn). Tests are **skipped** unless you opt in:

```bash
export RUN_HEAVY_TESTS=1
export GFMBENCH_DATA_ROOT=/path/to/benchmark/data

pytest tests/e2e/test_heavy.py -m heavy
```

### Run everything

```bash
export RUN_DOWNLOAD_TESTS=1
export RUN_HEAVY_TESTS=1
export GFMBENCH_DATA_ROOT=/path/to/benchmark/data

pytest tests/
```


## Opt-in environment variables

Download and heavy tiers use env vars as a **second gate** alongside markers — even if tests are collected, they skip unless enabled:

| Variable | Tier | Required value | Purpose |
|----------|------|----------------|---------|
| `RUN_DOWNLOAD_TESTS` | Download | `1` | Enable HF download tests |
| `RUN_HEAVY_TESTS` | Heavy | `1` | Enable heavy regression |
| `GFMBENCH_DATA_ROOT` | Heavy | path to existing dir | Root directory for benchmark datasets |
| `UPDATE_HEAVY_BASELINE` | Heavy | `1` | Regenerate baseline CSV and skip assertion |

## Baselines

Pinned scores live in `tests/fixtures/` (e.g. `dnabert2_sanity_baseline.csv`). Each row: `task`, `metric`, `expected`, `atol` (default tolerance 0.02).

Regenerate after an intentional model or pipeline change:

```bash
export RUN_HEAVY_TESTS=1
export GFMBENCH_DATA_ROOT=/path/to/benchmark/data

UPDATE_HEAVY_BASELINE=1 pytest tests/e2e/test_heavy.py -m heavy -s
```
