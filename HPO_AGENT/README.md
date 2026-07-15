# HPO Agent — FedCMAPSS Method-Level Hyperparameter Optimization

## Purpose

The FedCMAPSS paper [Sorrenti+ 2026] benchmarks four federated optimization
methods (**FedAvg**, **SCAFFOLD**, **FedDyn**, **FedCross**) on the C-MAPSS
turbofan dataset across five task templates (A: IID, B: domain shift,
C: label skew, D: feature skew, E: few-shot). Two observations from Tables II-VI:

1. **FedDyn is consistently the worst method across all five tasks** —
   RMSE 2-3× higher than the best per-task method. The paper explicitly
   flags: *"FedDyn exhibits an almost-constant trend which questions its
   suitability to the tasks at hand"* (§IV.C).
2. **FedAvg collapses on Task B (domain shift)**, especially with the CNN
   architecture (RMSE 46.78 ± 3.02, NASA σ ≈ 370). It is competitive on
   IID (Task A) but degrades severely when 4 clients each own a distinct
   sub-dataset (FD001-FD004).

The paper explicitly states (§III.D):
> *"Method-specific hyperparameters for FL methods are set to default
> values in the PFLLib library"* [pfllib.com].

**This is the hypothesis this agent tests**: whether the poor
performance of FedDyn and Task-B-FedAvg is a *hyperparameter selection
issue* (default PFLLib values not tuned for the RUL loss landscape) or
a *fundamental limitation of the algorithm* on this problem class.

If HPO closes the gap → the paper's conclusions on FedDyn need
revisiting; if HPO does *not* close the gap → the paper's framing is
strengthened.

---

## Agent Scope

**The agent may only**:
- Run sweeps on `system/main.py` with modified method-specific
  hyperparameters
- Analyze `wandb` runs and produce comparison tables
- Write new YAML sweep configs under `sweeps/HPO_AGENT/`
- Read the paper's tables (memorised below) as ground truth

**The agent MUST NOT**:
- Modify `system/flcore/servers/*_rul.py` or `system/flcore/clients/*_rul.py`
  (the algorithm implementations themselves)
- Modify the dataset splits — the ten seeds per task must remain identical
  to the paper's Tables II-VI for direct comparability
- Modify the neural architectures
- Change the RUL preprocessing (piecewise-linear clip at 125,
  normalization to [0,1])
- Change the number of rounds (100) or the evaluation protocol

Only what changes: **the method-specific hyperparameters**. This is a
controlled variable-isolation study.

---

## Fixed evaluation protocol (do not change)

| Setting | Value | Source |
|---|---|---|
| Rounds | 100 | Paper §III.F |
| Local epochs | Sweep `{1, 5, 10}` per method, pick winner | Paper Fig. 1 |
| Splits per task | 10 (indexed 0-9) | Paper §III.B |
| RUL clip | 125 cycles | Paper §III.F |
| Normalization | Linear [0,1] on RUL | Paper §III.F |
| Loss | MSE (+ method-specific reg terms) | Paper §III.F |
| Metrics | RMSE + NASA Score | Paper §III.E |
| Framework | PFLLib inside `ext/FedCMAPSS/` | Paper §III.D |
| Data root | `dataset/FedCMAPSS/` | Sweep configs |
| Dataset name | `FedCMAPSSWindow` | Sweep configs |

---

## Hyperparameters to search

### 1. FedDyn (`server = FedDyn_RUL`) — **highest priority**

The dynamic regularization introduces a term `α · ⟨θ - θ_global, h_i⟩ + α/2 · ‖θ - θ_global‖²`
where `h_i` accumulates local drift. When `α` is mistuned, the
regularization overwhelms the primary loss.

**Grid**:
| Hyperparameter | Values | Rationale |
|---|---|---|
| `alpha_coef` | `{1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 1.0}` | PFLLib default is 0.01; tuning both up and down |
| `client_learning_rate` | `{5e-4, 1e-3, 5e-3, 1e-2}` | Complementary to `alpha_coef` |
| `local_epochs` | `{1, 5, 10}` | Copy from Fig. 1 of paper |
| `model` | `LSTM_RUL`, `AttBiGRU_RUL` | Focus on the two architectures where FedDyn showed *least* disaster (Task E LSTM 34.41 RMSE) |
| `task` | `A` and `B` initially | Isolate whether the issue is IID (A) or non-IID (B) |
| `split` | `0..2` initially | Only 3 seeds for grid pruning; then run best config on all 10 |

**Expected size**: 6 × 4 × 3 × 2 × 2 × 3 = **864 runs** (~15 min each →
216 GPU-hours). Prune with hyperband after 20 rounds.

**Success criterion**: FedDyn best config on Task A LSTM should reach
RMSE ≤ 22 (paper reports 23.45). On Task B LSTM ≤ 22 (paper 40.83).
If achieved, the paper's conclusion on FedDyn was HP-driven.

**File to create**: `sweeps/HPO_AGENT/feddyn_grid_alpha.yaml`

### 2. FedAvg on Task B (`server = FedAvg_RUL`) — **second priority**

The CNN failure on Task B (RMSE 46.78, NASA σ ≈ 370) is the sharpest
anomaly in the paper's tables. FedAvg has no method-specific
hyperparameter but the *client* learning rate is critical.

**Grid**:
| Hyperparameter | Values |
|---|---|
| `client_learning_rate` | `{1e-4, 5e-4, 1e-3, 5e-3, 1e-2}` |
| `local_epochs` | `{1, 5, 10}` |
| `momentum` | `{0.0, 0.9}` — if the optimizer supports it |
| `weight_decay` | `{0.0, 1e-5, 1e-4}` |
| `model` | `Chen_CNN_RUL`, `LSTM_RUL` — CNN because it's the failure case; LSTM as control |
| `task` | `B` only |
| `split` | `0..2` |

**Expected size**: 5 × 3 × 2 × 3 × 2 × 3 = **540 runs**

**Success criterion**: FedAvg CNN Task B RMSE should reach the FedAvg
LSTM level on Task B (21.14). If achieved, the CNN failure is
optimization-related, not architectural.

**File to create**: `sweeps/HPO_AGENT/fedavg_taskB_lr.yaml`

### 3. SCAFFOLD (`server = SCAFFOLD_RUL`) — **third priority**

Explore whether the control-variate learning rate is set inconsistently
with the primary rate.

**Grid**:
| Hyperparameter | Values |
|---|---|
| `client_learning_rate` | `{1e-3, 5e-3, 1e-2}` |
| `server_learning_rate` | `{0.5, 1.0}` |
| `local_epochs` | `{5, 10}` |
| `model` | `AttBiGRU_RUL`, `LSTM_RUL` |
| `task` | `C` and `E` (where SCAFFOLD underperforms) |
| `split` | `0..2` |

**Expected size**: 3 × 2 × 2 × 2 × 2 × 3 = **144 runs**

**Success criterion**: SCAFFOLD Task C AttBiGRU RMSE ≤ 20.18 (FedCross's number).

**File to create**: `sweeps/HPO_AGENT/scaffold_lr_pairs.yaml`

### 4. FedCross (`server = FedCross_RUL`) — **sanity/robustness check only**

FedCross wins several tasks in the paper. Verify robustness of the
default `n_middleware` choice.

**Grid**:
| Hyperparameter | Values |
|---|---|
| `n_middleware` | `{3, 5, 7}` |
| `local_epochs` | `{5, 10}` |
| `model` | `LSTM_RUL`, `AttBiGRU_RUL` |
| `task` | `B`, `C` (paper best) |
| `split` | `0..2` |

**Expected size**: 3 × 2 × 2 × 2 × 3 = **72 runs**

**Success criterion**: no dramatic swing → default `n_middleware`
already close to optimum.

**File to create**: `sweeps/HPO_AGENT/fedcross_middleware.yaml`

---

## Run protocol

1. Activate the FedCMAPSS environment (see `env_cuda_latest.yaml` and
   `prepare.sh`).
2. Register a fresh W&B project: `wandb project create HPO_AGENT_FedCMAPSS`.
3. Create sweep: `wandb sweep sweeps/HPO_AGENT/<config>.yaml`.
4. Launch agents on available GPUs — recommend `wandb agent
   <sweep_id>` in parallel on N ≤ 4 shards per GPU.
5. After each sweep completes, produce `HPO_AGENT/results/<config>.md`
   with:
   - Best config table (all HP values + median RMSE + std across 3 splits)
   - Delta vs paper's Table II/III/IV/V/VI numbers
   - Verdict: "HP-driven collapse" / "algorithm-driven collapse" / "inconclusive"
6. When best HP found on 3-split pilot: rerun on all 10 splits for final
   statistics comparable with the paper.

## Wandb project convention

- Project name: `HPO_AGENT_FedCMAPSS`
- Group: sweep filename without extension (e.g. `feddyn_grid_alpha`)
- Tags: `{task}`, `{model}`, `{algorithm}`, `hpo_agent`

## Deliverables (per method sweep)

1. **`sweeps/HPO_AGENT/<config>.yaml`** — the sweep definition
2. **`HPO_AGENT/results/<config>.md`** — best-config table + verdict
3. **`HPO_AGENT/results/<config>_all_splits.md`** — final full-10-split
   confirmation table
4. **`HPO_AGENT/final_report.md`** — cross-method meta-analysis
   answering: *"can any of these methods be salvaged by HPO on this
   benchmark?"*

## Reference — paper's baseline numbers to beat

### Table II — Task A (IID, RMSE)

| | FedAvg | SCAFFOLD | FedDyn | FedCross |
|---|---|---|---|---|
| LSTM | 17.66 ± 1.11 | **17.36 ± 1.11** | 23.45 ± 0.72 | 18.81 ± 1.08 |
| AFT | 17.19 ± 1.09 | **16.59 ± 1.41** | 43.71 ± 1.12 | 18.79 ± 1.30 |
| AttBiGRU | 17.84 ± 1.01 | **17.46 ± 1.67** | 31.32 ± 3.17 | 22.37 ± 1.59 |
| RNN | 18.97 ± 0.81 | **18.96 ± 0.88** | 41.82 ± 0.31 | 20.01 ± 0.80 |
| CNN | 14.79 ± 0.58 | **14.56 ± 0.55** | 40.45 ± 0.41 | 15.48 ± 0.76 |

### Table III — Task B (domain shift, RMSE)

| | FedAvg | SCAFFOLD | FedDyn | FedCross |
|---|---|---|---|---|
| LSTM | 21.14 ± 0.58 | 18.42 ± 0.38 | 40.83 ± 1.61 | **16.43 ± 0.31** |
| AFT | 29.30 ± 3.22 | 17.30 ± 0.86 | 47.17 ± 3.19 | **15.85 ± 0.63** |
| AttBiGRU | 27.17 ± 1.35 | 17.84 ± 0.37 | 38.24 ± 2.32 | **16.90 ± 0.38** |
| RNN | 28.03 ± 2.74 | **20.06 ± 0.37** | 44.82 ± 0.35 | 20.27 ± 0.51 |
| CNN | 46.78 ± 3.02 | 37.30 ± 1.71 | 42.26 ± 0.34 | **33.69 ± 1.31** |

### Table IV — Task C (label skew, RMSE)

| | FedAvg | SCAFFOLD | FedDyn | FedCross |
|---|---|---|---|---|
| LSTM | 29.48 ± 0.34 | 27.31 ± 1.05 | 39.92 ± 2.40 | **17.28 ± 0.92** |
| AFT | 30.32 ± 0.33 | 31.22 ± 1.91 | 49.44 ± 1.26 | **17.75 ± 0.70** |
| AttBiGRU | 30.87 ± 0.24 | 32.15 ± 2.39 | 36.30 ± 2.11 | **20.18 ± 0.68** |
| RNN | 30.52 ± 0.50 | 34.69 ± 4.51 | 45.24 ± 0.28 | **18.24 ± 0.33** |
| CNN | 35.12 ± 0.38 | 33.10 ± 0.59 | 42.93 ± 0.20 | **26.22 ± 0.68** |

### Table V — Task D (feature skew, RMSE)

| | FedAvg | SCAFFOLD | FedDyn | FedCross |
|---|---|---|---|---|
| LSTM | 14.12 ± 0.65 | 12.58 ± 0.25 | 24.67 ± 1.05 | **12.28 ± 0.23** |
| AFT | 16.30 ± 0.64 | **15.64 ± 0.21** | 39.86 ± 0.38 | 15.44 ± 0.41 |
| AttBiGRU | 15.15 ± 0.25 | 14.13 ± 0.08 | 33.84 ± 0.93 | **14.05 ± 0.15** |
| RNN | **13.84 ± 0.38** | 15.63 ± 0.11 | 40.06 ± 0.33 | 15.05 ± 0.23 |
| CNN | 17.15 ± 0.31 | 17.90 ± 0.38 | 40.24 ± 0.71 | **15.40 ± 0.22** |

### Table VI — Task E (few-shot, RMSE)

| | FedAvg | SCAFFOLD | FedDyn | FedCross |
|---|---|---|---|---|
| LSTM | **27.30 ± 0.82** | 28.92 ± 0.99 | 34.41 ± 1.70 | 28.15 ± 1.10 |
| AFT | **34.42 ± 1.26** | 34.81 ± 1.30 | 41.00 ± 0.45 | 36.94 ± 1.41 |
| AttBiGRU | **32.24 ± 0.80** | 32.74 ± 0.86 | 42.29 ± 3.65 | 33.13 ± 0.83 |
| RNN | **41.11 ± 0.86** | 40.01 ± 0.64 | 41.10 ± 0.90 | 40.44 ± 0.74 |
| CNN | **29.69 ± 0.59** | 30.35 ± 1.01 | 40.25 ± 0.51 | 32.69 ± 0.80 |

**Bold** = per-row best in the paper.

## Non-goals

- **Not** proposing new algorithms.
- **Not** modifying data splits, evaluation protocol, or paper's five-task taxonomy.
- **Not** searching over architectures (already covered by paper's Tables I-VI).
- **Not** exploring personalized FL (pFL) methods — outside scope.

## Contact protocol

If the agent identifies HP configurations that close ≥ 50 % of the
gap for FedDyn/Task-B-FedAvg on the pilot 3-split runs, escalate
before running the full 10-split confirmation — this may warrant a
short comment/reply to FedCMAPSS authors before publishing.
