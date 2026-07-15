# DEPLOYMENT — How to actually run the HPO campaigns

> Companion to `AGENT_PROMPT.md`. This document explains **how**, not **what**.
> Skills + subagents referenced here live in the parent FedSKY repo at
> `/home/lpala/fedsky/.claude/{skills,agents}/`. You (the HPO agent) invoke
> them; you do not need to read their internals.

---

## Available skills you can invoke

Prefix with `/`. All three are auto-discovered by Claude Code when this
file's parent repo is the active workspace.

| Skill | Purpose |
|---|---|
| `/launch-wandb-sweep` | Create the sweep from a YAML + spawn N agents in one shot |
| `/deploy-gpu-agents` | Attach more workers to an ALREADY existing sweep |
| `/monitor-sweeps` | Query sweep state, best runs, per-HP marginals |

## Available subagents you can spawn via the Agent tool

| Subagent | Purpose |
|---|---|
| `sweep-launcher` | Fire-and-forget sweep bring-up (returns a JSON status blob) |
| `sweep-analyst` | Read runs, produce report + verdict against the paper's baseline |

## Deployment modes — pick per campaign

Each host has one of these modes. The skills auto-detect; you may also
force with an explicit flag.

- **`direct`** — bare-metal or long-lived VM with ≥1 GPU visible via `nvidia-smi`. Agents run as background processes with `CUDA_VISIBLE_DEVICES`.
- **`slurm`** — HPC head node. Agents are submitted as sbatch jobs; the shared template is `scripts/slurm/wandb_agent.sbatch`.
- **`remote`** — SSH access to a fleet of GPU hosts (list in `~/.claude/hosts.txt`). Agents are spawned via SSH.

Rule of thumb for the four HPO_AGENT campaigns:

| Campaign | Grid size | Recommended mode | # agents |
|---|---|---|---|
| Campaign 1 — FedDyn `alpha_coef` | 864 runs | SLURM | 8-16 |
| Campaign 2 — FedAvg Task B | 540 runs | SLURM | 8-16 |
| Campaign 3 — SCAFFOLD lr pairs | 144 runs | direct | 4-8 |
| Campaign 4 — FedCross middleware | 72 runs | direct | 4-8 |

## End-to-end walkthrough for Campaign 1

```
User: /launch-wandb-sweep ext/FedCMAPSS/sweeps/HPO_AGENT/feddyn_grid_alpha.yaml
      n_agents=12 mode=slurm project=HPO_AGENT_FedCMAPSS
```

Claude Code (via the `launch-wandb-sweep` skill):

1. Validates the YAML
2. `wandb sweep <yaml>` → captures `entity/HPO_AGENT_FedCMAPSS/xyz1234`
3. Submits 12 × `sbatch scripts/slurm/wandb_agent.sbatch <sweep_id>`
4. Waits 10 s, verifies each agent registered
5. Returns:
   - Sweep URL
   - 12 SLURM Job IDs
   - Log paths

Then, periodically:

```
User: /monitor-sweeps entity/HPO_AGENT_FedCMAPSS/xyz1234
```

When the sweep hits its natural end (or when you decide to prune):

```
User: /analyze-sweep entity/HPO_AGENT_FedCMAPSS/xyz1234 \
      target=test/rmse paper_baseline=23.45 \
      output=ext/FedCMAPSS/HPO_AGENT/results/feddyn_grid_alpha.md
```

Claude Code spawns the `sweep-analyst` subagent, which writes the report + gives you a verdict (HP-driven vs algorithm-driven).

## Multi-host distributed setup

If SLURM is saturated OR you have SSH access to `gpu-01..gpu-04`, drop this into `~/.claude/hosts.txt`:

```
gpu-01
gpu-02
gpu-03
gpu-04
```

Then the same sweep can be scaled with:

```
User: /deploy-gpu-agents entity/proj/xyz1234 n_agents=16 mode=remote
```

The skill spawns 4 agents per host across the 4 hosts.

## Cross-mode composition

Nothing stops you from running a sweep with agents in mixed modes: 8 SLURM jobs + 4 direct local processes + 4 remote SSH agents, all attached to the SAME sweep_id. W&B coordinates.

Recommended pattern for time-critical campaigns:

1. `/launch-wandb-sweep` with `mode=direct n_agents=4` (immediate start on the head node)
2. Add SLURM workers: `/deploy-gpu-agents mode=slurm n_agents=12`
3. If SSH hosts idle: `/deploy-gpu-agents mode=remote n_agents=16 hosts=~/.claude/hosts.txt`

Total workers: 32 → sweep of 864 runs finishes in ~30 wallclock minutes at 15 min per run.

## Failure recovery

- **Agent dies mid-run**: W&B will re-queue the config. If ≥ 3 agents fail with the same error, stop and inspect one crashed run.
- **SLURM QOSMaxJobsPerUser**: `/deploy-gpu-agents` reduces `n_agents` and retries automatically once, then reports.
- **Sweep stalled** (no state change in 20 min AND runs pending): re-invoke `/deploy-gpu-agents` — usually a batch of agents died silently.

## Cleanup

At the end of each campaign:

```
User: /stop-agents <sweep_id>
```

The launcher tracks PIDs / JobIDs in `runs/logs/sweep-<id>-*.txt` and kills them cleanly.

## Reporting

Per campaign, drop the analyst's output in `ext/FedCMAPSS/HPO_AGENT/results/`:

- `feddyn_grid_alpha.md` (Campaign 1)
- `fedavg_taskB_lr.md` (Campaign 2)
- `scaffold_lr_pairs.md` (Campaign 3)
- `fedcross_middleware.md` (Campaign 4)

After all four: `final_report.md` + `final_gaps.csv` + `gap_closure_summary.png`.

## What the FedSKY team is doing in parallel

The FedSKY agent (a separate agent + skill fleet) is running its own
experiments on FedCMAPSS splits via
`configs/train/fedcmapss_task{A..E}_fedhead.json`. **You do not need
to coordinate.** FedSKY writes to `runs/fedcmapss_task*` and W&B
project `fedsky-cmapss`; your writes go to
`HPO_AGENT_FedCMAPSS` — no collisions. If a shared GPU is contested,
SLURM QoS ordering resolves it.

## Never do

- **Never** `wandb sweep --stop` unless the user explicitly tells you to.
- **Never** modify `scripts/slurm/wandb_agent.sbatch` beyond the `--time`, `--mem`, and `--qos` fields — the rest is shared with FedSKY.
- **Never** re-generate the CMAPSS splits — `ext/FedCMAPSS/output/tasks.json` is FROZEN for the campaign.
