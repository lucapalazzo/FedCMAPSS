# AGENT PROMPT — FedCMAPSS Hyperparameter Optimization

> **Copy-paste this file into a fresh Claude Code / Cursor / any-MCP-agent session.**
> Everything the agent needs to run the four HPO campaigns autonomously
> is in this file plus `README.md` in the same directory.

---

## Role

You are an autonomous experimenter operating inside the local clone of
FedCMAPSS at `/home/lpala/fedsky/ext/FedCMAPSS`. Your single mission
is to determine whether the four federated optimization methods
benchmarked in Sorrenti et al. 2026 (FedAvg, SCAFFOLD, FedDyn,
FedCross) can be improved by tuning their method-specific
hyperparameters — hyperparameters the paper left at PFLLib defaults.

**You do NOT touch FedSKY.** Another agent owns that codebase; stay
inside `/home/lpala/fedsky/ext/FedCMAPSS`.

---

## What is broken vs. what is unclear

Read `README.md` in this directory for the full ground-truth table
(Tables II–VI from the paper). The empirical picture is:

| Method | Where it fails | Suspected cause |
|---|---|---|
| **FedDyn** | Every task, 2-3× worse than best (Tables II-VI) | `alpha_coef` default (0.01 in PFLLib) probably wrong for RUL loss landscape — dynamic regularizer dominates the MSE gradient |
| **FedAvg** | Task B `Chen_CNN_RUL` RMSE 46.78 ± 3.02 (NASA σ≈370) | Client drift under 4-clients × 4-FDs domain shift; possibly LR too high for CNN |
| **SCAFFOLD** | Task C AttBiGRU RMSE 32.15 (vs FedCross 20.18) | Control-variate LR mismatch on strong label skew |
| **FedCross** | Wins Tables III-V but **why** and **how robustly** is unstated | `n_middleware` default (3 in PFLLib) never swept — is it optimal, or would 5/7 be better/worse? |

Your job is to answer, quantitatively, for each method:

1. Is the paper's poor RMSE HP-driven (fixable by tuning) or
   algorithm-driven (fundamental to the method under this data
   heterogeneity)?
2. For the winner (FedCross), is the default `n_middleware` optimal or
   is there headroom?

---

## Rules of engagement

**You MUST NOT:**

- Modify `system/flcore/servers/*_rul.py` or `system/flcore/clients/*_rul.py`
  (the algorithms' code).
- Modify the dataset splits — the 10 seeds per task must stay
  bit-identical to the paper's.
- Modify the neural architectures in
  `system/flcore/trainmodel/models.py`.
- Modify the RUL preprocessing (piecewise-linear clip at 125,
  normalize to [0,1]).
- Change the number of communication rounds (100).
- Explore personalized-FL (pFL) methods — outside scope.
- Touch anything in `/home/lpala/fedsky` outside of
  `/home/lpala/fedsky/ext/FedCMAPSS/`.

**You MAY:**

- Create YAML sweep configs under `sweeps/HPO_AGENT/`.
- Launch W&B sweeps and agents.
- Read `wandb` runs and produce comparison markdown reports under
  `HPO_AGENT/results/`.
- Query the paper's tables (transcribed verbatim in `README.md`).
- Prune sweeps early (e.g. after 20 rounds) when a config is clearly
  dominated.
- Escalate to the human when a config appears to close ≥ 50 % of the
  gap to the best per-task RMSE — the human may want to redirect
  before you spend 10× the compute on the full 10-split confirmation.

---

## How to actually run the sweeps

Deployment mechanics (skills, subagents, SLURM template) are documented in
[`DEPLOYMENT.md`](./DEPLOYMENT.md) — the parent FedSKY workspace exposes
these skills:

- `/launch-wandb-sweep <yaml> n_agents=<N> mode=<auto|direct|slurm|remote>`
- `/deploy-gpu-agents <sweep_id> n_agents=<N> mode=<...>` (attach more workers)
- `/monitor-sweeps <sweep_id>` (state + best runs + per-HP marginals)

And these subagents (invoke via the Agent tool):
- `sweep-launcher` — fire-and-forget sweep bring-up
- `sweep-analyst` — analyze results + write the campaign report

The SLURM template is `scripts/slurm/wandb_agent.sbatch` (in the parent
FedSKY repo — the skills submit it for you; you don't invoke it directly).

**Recommended mode per campaign** (see `DEPLOYMENT.md` table):

| Campaign | Grid | Mode | # agents |
|---|---|---|---|
| 1 (FedDyn) | 864 | SLURM | 12-16 |
| 2 (FedAvg Task B) | 540 | SLURM | 8-16 |
| 3 (SCAFFOLD) | 144 | direct | 4-8 |
| 4 (FedCross) | 72 | direct | 4-8 |

## Environment setup (do this once)

```bash
cd /home/lpala/fedsky/ext/FedCMAPSS
# Env already prepared by the human; verify:
conda env list | grep -E "pfllib|flvit"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pfllib   # or flvit — check what has PFLlib + torch installed

# Data prep — already run by the human; verify:
ls output/tasks.json output/cmapss_processed_*.csv

# W&B — expect the human's login to be active:
wandb login --relogin  # only if `wandb whoami` fails
```

If any of the above fails, DO NOT try to install/re-run — report and
stop. The data-prep step (`private/fedcmapss_dataset_create.py`)
requires the raw NASA C-MAPSS files symlinked into
`private/dataset/`, which the human already did.

---

## The four campaigns (execute in this order)

Each campaign has:
1. A YAML config already stubbed under `sweeps/HPO_AGENT/`
2. Explicit HP grid + expected wallclock
3. A success criterion (an RMSE threshold to beat)
4. A required deliverable file in `results/`

### Campaign 1 — FedDyn `alpha_coef` sweep (HIGHEST priority)

**Config**: `sweeps/HPO_AGENT/feddyn_grid_alpha.yaml`
**Grid**: 6 × 4 × 3 × 2 × 2 × 3 = 864 runs
**Wallclock**: ~15 min/run × 864 ÷ #GPUs
**Success criterion**: Any HP combination brings FedDyn LSTM Task A RMSE to
≤ 22 (paper: 23.45) OR FedDyn LSTM Task B RMSE to ≤ 22 (paper: 40.83).

**Actions**:
1. `wandb sweep sweeps/HPO_AGENT/feddyn_grid_alpha.yaml` → save sweep_id
2. Launch N agents: `wandb agent <sweep_id>` × #GPUs
3. When done: write `results/feddyn_grid_alpha.md` with:
   - Best-3 config table (all HP values, mean RMSE across 3 splits, mean NASA Score)
   - Per-alpha_coef heat-map: alpha_coef × local_lr, colored by RMSE
   - Verdict: **"HP-driven"** if closes ≥ 50 % of gap; **"algorithm-driven"** otherwise
4. If verdict = HP-driven: rerun best config on all 10 splits (`split: [0..9]`)
   → confirm with `results/feddyn_grid_alpha_all_splits.md`

### Campaign 2 — FedAvg on Task B (`Chen_CNN_RUL` catastrophic case)

**Config**: `sweeps/HPO_AGENT/fedavg_taskB_lr.yaml`
**Grid**: 5 × 3 × 2 × 3 × 2 × 3 = 540 runs
**Success criterion**: FedAvg CNN Task B RMSE ≤ 21 (LSTM baseline).

**Actions**:
1. Same launch pattern as Campaign 1.
2. Deliverable `results/fedavg_taskB_lr.md`:
   - Best-3 config table
   - LR × momentum × weight_decay 3-panel heat-map
   - Verdict + 10-split confirm if verdict = HP-driven

### Campaign 3 — SCAFFOLD (`client_lr` × `server_lr` pair)

**Config**: `sweeps/HPO_AGENT/scaffold_lr_pairs.yaml`
**Grid**: 3 × 2 × 2 × 2 × 2 × 3 = 144 runs
**Success criterion**: SCAFFOLD Task C AttBiGRU RMSE ≤ 20.18 (FedCross's number).

**Actions**: same pattern, `results/scaffold_lr_pairs.md`.

### Campaign 4 — FedCross (`n_middleware` sanity)

**Config**: `sweeps/HPO_AGENT/fedcross_middleware.yaml`
**Grid**: 3 × 2 × 2 × 2 × 3 = 72 runs
**Success criterion**: no dramatic swing (± 1 RMSE) → default is fine;
if 5 or 7 is materially better, flag it.

**Actions**: same pattern, `results/fedcross_middleware.md`.

---

## Cross-campaign meta-analysis

After all four campaigns complete, produce **`results/final_report.md`**
answering, in ≤ 500 words:

1. Is any of the four methods salvageable via HPO on this benchmark?
2. If yes, which method(s) and by how much?
3. Which methods remain fundamentally limited (algorithm-driven
   failure)?
4. What single change to the paper's Table II-VI methodology would
   most affect its conclusions?
5. Recommend whether FedCMAPSS authors should update their paper with
   an HP-tuning appendix.

Attach:
- One CSV `results/final_gaps.csv` with columns:
  `method, task, model, paper_rmse, best_hp_rmse, gap_closed_pct, verdict`
- One PNG `results/gap_closure_summary.png` — bar chart of gap
  closure per (method, task) combination.

---

## Deliverables checklist

- [ ] `sweeps/HPO_AGENT/feddyn_grid_alpha.yaml` (already stubbed by
      human; verify it launches)
- [ ] `sweeps/HPO_AGENT/fedavg_taskB_lr.yaml` (idem)
- [ ] `sweeps/HPO_AGENT/scaffold_lr_pairs.yaml` (idem)
- [ ] `sweeps/HPO_AGENT/fedcross_middleware.yaml` (idem)
- [ ] `results/feddyn_grid_alpha.md`
- [ ] `results/feddyn_grid_alpha_all_splits.md` (if verdict = HP-driven)
- [ ] `results/fedavg_taskB_lr.md`
- [ ] `results/fedavg_taskB_lr_all_splits.md` (if HP-driven)
- [ ] `results/scaffold_lr_pairs.md`
- [ ] `results/scaffold_lr_pairs_all_splits.md` (if HP-driven)
- [ ] `results/fedcross_middleware.md`
- [ ] `results/final_report.md`
- [ ] `results/final_gaps.csv`
- [ ] `results/gap_closure_summary.png`

---

## W&B conventions

- **Project**: `HPO_AGENT_FedCMAPSS`
- **Group** (auto-derived from YAML `name`): `feddyn_grid_alpha`, etc.
- **Tags per run**: `task-{A..E}`, `model-{LSTM|AFT|AttBiGRU|RNN|Chen_CNN}`,
  `algo-{FedAvg|SCAFFOLD|FedDyn|FedCross}`, `hpo-agent`,
  `campaign-{1..4}`, `pilot` (3-split) or `full` (10-split confirm).

---

## Reporting cadence

- After each **campaign** completes: post its `results/*.md` file
  path to the human's chat and pause. Do NOT auto-start the next
  campaign — wait for `continue` (either explicit reply or an
  automated CI-style green light).
- After the final report: post `results/final_report.md` and stop.

---

## Anti-patterns to avoid

- Do not run all four campaigns before delivering intermediate results.
- Do not skip the 3-split → 10-split confirmation for HP-driven verdicts.
- Do not add new HP dimensions unilaterally (e.g. `dropout`, `optimizer`).
- Do not run `n_middleware > 7` in Campaign 4 — memory footprint grows
  linearly and PFLLib's FedCross default caps there.
- Do not compare NASA Score means when σ is comparable to μ — the
  paper's own §IV.B footnote flags this. Report σ only.
- Do not touch FedSKY. Not even to peek.

---

## Escalation triggers (stop and consult the human)

- Any single-run RMSE < 12 on Task A (implausibly good — check for
  data leakage in your sweep config).
- Any campaign that would consume > 500 GPU-hours based on the timing
  observed after the first 30 runs — the sweep is too large, prune.
- The dataset re-generation script triggers (should never happen if
  `output/` is intact).
- Any conda/pytorch/wandb error you can't resolve in ≤ 2 attempts.

Good luck. Return concise verdicts, one campaign at a time.
