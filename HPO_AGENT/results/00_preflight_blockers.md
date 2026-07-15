# Pre-flight validation — HPO campaigns BLOCKED

**Status:** No sweep launched. All four stub configs in `sweeps/HPO_AGENT/` fail
against the actual `system/main.py` CLI. Reported before burning GPU-hours.

Every claim below is verified by running the command, not by reading alone.

---

## B1 — `--data_root dataset/FedCMAPSS/` does not exist (breaks all 4 campaigns)

`flcore/datasets/fedcmapss.py:34` opens `os.path.join(data_root, 'tasks.json')`.
That path resolves to `dataset/FedCMAPSS/tasks.json`, which is absent in this clone.
The prepared data lives in `output/` (`tasks.json`, `cmapss_processed_{train,test}_data.csv`).

```
$ python system/main.py --data_root dataset/FedCMAPSS/ -gr 0
FileNotFoundError: [Errno 2] No such file or directory: 'dataset/FedCMAPSS/tasks.json'
```

Every run of every campaign would crash on startup. Note the *paper's own* sweeps
(`sweeps/methods_grid_1.yaml`) also use `dataset/FedCMAPSS/` — so on the authors'
machine that path existed. Fix is either `--data_root output/` or a symlink
`dataset/FedCMAPSS -> output/` (the symlink keeps our configs byte-identical to the paper's).

## B2 — Campaign 1: `alpha_coef` is not a CLI flag

```
$ python system/main.py --alpha_coef 0.01
main.py: error: unrecognized arguments: --alpha_coef 0.01
```

FedDyn reads **`args.alpha`** (`serverdyn_rul.py:22`, `clientdyn_rul.py:11`), exposed as
`-al` / `--alpha`. Renaming the sweep key fixes it.

### …and the premise about its default is wrong

`main.py:619` sets `--alpha` **default = 1.0**, not 0.01. No sweep in `sweeps/` ever passes
`--alpha`, so **the paper's published FedDyn numbers were produced with `alpha = 1.0`.**

This does not invalidate Campaign 1 — it sharpens it. `alpha=1.0` is an extremely heavy
dynamic regularizer against an MSE RUL loss, which is a very plausible cause of FedDyn's
2–3× collapse. The proposed grid `{1e-4 … 1.0}` already brackets the true default at its
top edge and searches downward, which is exactly the right direction. Only the README's
stated rationale ("default is 0.01") needs correcting.

## B3 — Campaign 2: `weight_decay` errors, `momentum` is a silent no-op

`weight_decay` is not a CLI arg → `unrecognized arguments` → crash.

`momentum` **parses** (it exists as `-mo`, added for FedDBE) but has **zero effect on
training**. Every RUL client optimizes with, verbatim (`clientbase.py:47`):

```python
self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
```

No `momentum=`, no `weight_decay=`. This is the most dangerous of the four failures: the
sweep would *not* error. It would run all 540 jobs, "succeed", and emit a
momentum × weight_decay heat-map over two axes that do nothing — 2× duplicate runs
presented as a tuning result.

Dropping both inert axes reduces Campaign 2 to `5 lr × 3 local_epochs × 2 model × 3 split`
= **90 real runs** (from 540).

## B4 — Campaign 4: `n_middleware` does not exist and cannot be swept

Not a CLI flag, and not a variable. FedCross's middleware count is derived, not configured
(`servercross_rul.py:32`):

```python
self.w_locals_num = self.num_join_clients
```

and `num_clients` is itself overwritten from the data (`serverbase_rul.py:22`,
`args.num_clients = num_clients`, read from `tasks.json`). So the middleware count is
**pinned to the client count of the task** (e.g. 10 for Task A) and cannot be set to 3/5/7
without editing `servercross_rul.py` — which the rules of engagement forbid.

Campaign 4 **as specified is impossible**. FedCross's genuinely tunable HPs are
`--fedcross_alpha` (default 0.99), `--collaberative_model_select_strategy` (0/1/2), and
`--first_stage_bound` (0) — but swapping those in is a change of study design, so I am not
doing it unilaterally.

## B5 — Environment

There is no `pfllib` conda env; the working one is **`flvit`** (torch 2.6.0+cu124, CUDA OK,
wandb 0.20.1, logged in as `lpalazzo`, entity `ngslung`). 4× H100 80GB.

## B6 — This is a SLURM node: bare `python` GPU runs are killed after ~40–80 s

The launch pattern in the prompt (`wandb agent <sweep_id>` × #GPUs, run directly) **cannot
work here**. Every GPU process started outside a SLURM allocation is SIGTERM'd (exit 143)
after 40–80 seconds. Isolated by bisection:

| Process | Outcome |
|---|---|
| `python` sleeping 90 s, no CUDA | **survives** |
| `python` holding a CUDA tensor, 90 s | **SIGTERM at 44 s** |
| same CUDA process under `srun --qos=debug` | **survives 100 s** |

The reaper is SLURM's GPU policing daemon (user `slurmwe+`, polling
`nvidia-smi --query-compute-apps` and mapping PIDs→users). Another user's (`ellesalvo`)
GPU processes persist precisely because they are inside SLURM jobs.

A 100-round run needs 40–100 minutes of continuous GPU residency, so **not a single run of
any campaign would ever have completed** under the prompt's launch pattern. All work must
be submitted via `srun`/`sbatch` with an explicit, valid QOS — a bare `srun` fails with
`Invalid qos specification` because the default QOS `normal` is not granted to this account.

### Quota is the binding constraint

`lpala` is in account `gpuusers` with QOS `debug,train`:

| QOS | Concurrent GPUs | Max jobs/user |
|---|---|---|
| `debug` | 1 | 1 |
| `train` | **3** | 3 |

So the hard ceiling is **3 concurrent GPU jobs**, not the 4 GPUs visible to `nvidia-smi`.

And that ceiling is already fully consumed: **128 PENDING jobs blocked on `QOSGrpGRES`
plus 1 RUNNING**, all belonging to a *different* project of yours —
`/home/lpala/fedgfe/sbatch/feda2v/xclass_perround_mc.sbatch` (`xcvgg-50n-1c-*`,
Account=gpuusers, QOS=train). FedCMAPSS jobs submitted to `train` today would queue behind
all 128. I have not touched them.

## B7 — Measured cost is ~4.5× the prompt's estimate, and Campaign 1 alone blows the budget

The prompt assumes ~15 min/run. Measured on an idle H100 (100 rounds, `-gr 100`):

| model | local_epochs | sec/round | → per 100-round run |
|---|---|---|---|
| LSTM | 1 | ~22 | **~37 min** |
| AttBiGRU | 5 | ~40 | ~68 min |
| AttBiGRU | 10 | ~55–60 | **~95–100 min** |

GPU utilisation is only 8–13 % at <1 GB — the loop is CPU/dataloader-bound (every client
re-reads the full 27 MB train CSV on *every* round, `fedcmapss.py:45`). Extra GPUs would
not help much; the 3-job QOS cap is what binds.

Projected at ~1.13 h/run average over `local_epochs ∈ {1,5,10}`:

| Campaign | Runs | Est. GPU-hours |
|---|---|---|
| 1 — FedDyn | 864 | **~975** |
| 2 — FedAvg | 90 | ~100 |
| 3 — SCAFFOLD | 144 | ~200 |
| 4 — FedCross | 144 | ~165 |
| **Total** | 1242 | **~1,440** |

The prompt's own escalation trigger is *"> 500 GPU-hours → the sweep is too large, prune."*
We are at **~3× that**, and **Campaign 1 alone (~975 GPU-h) exceeds it**. At the 3-GPU QOS
cap that is ~20 days of wallclock — and only *after* the 128 queued `fedgfe` jobs clear.

**Stopped for instructions. No sweep launched.**

---

## Campaign status

| Campaign | Runnable as written? | Blocker |
|---|---|---|
| 1 — FedDyn | No | `alpha_coef` → `alpha` (B2) + data_root (B1) |
| 2 — FedAvg Task B | No | `weight_decay` crashes; `momentum` inert (B3) + B1 |
| 3 — SCAFFOLD | **Yes** (after B1) | only data_root. `local_learning_rate` + `server_learning_rate` both exist and are used (`serverscaffold_rul.py:112`) |
| 4 — FedCross | **No — impossible** | `n_middleware` is not a hyperparameter (B4) |

## Secondary observation — pilot splits are not comparable to the paper

The paper's grid ran splits `[0,1,3,4,5,6,7,8,9]` — it **skips split 2**. The HPO pilot
specifies splits `[0,1,2]`. So a 3-split pilot mean would include one split with no paper
baseline, and would be compared against paper means computed over a different split set.
For the pilot (relative ranking of HPs) this is harmless; for any headline
"closes X% of the gap" number it is not. Recommend pilot on splits `[0,1,3]`.

---

## Proposed corrected mapping (not yet applied)

| Campaign | Stub key | Correct key | Note |
|---|---|---|---|
| 1 | `alpha_coef` | `alpha` | default is 1.0, grid searches below — good |
| 2 | `weight_decay` | *(drop)* | not wired into the optimizer |
| 2 | `momentum` | *(drop)* | parses but no-op → 540 runs collapse to 90 |
| 3 | — | — | already correct |
| 4 | `n_middleware` | *(none)* | needs a decision — see B4 |
| all | `data_root: dataset/FedCMAPSS/` | `output/` or symlink | |
