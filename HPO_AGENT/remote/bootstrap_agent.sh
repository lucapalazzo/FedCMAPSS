#!/usr/bin/env bash
# Bootstrap a W&B sweep agent on a remote machine — FedCMAPSS HPO.
#
# The data is NEVER regenerated. `output/` on the origin host is the golden copy and is
# rsync'd verbatim. Rationale: private/fedcmapss_dataset_create.py clusters operating
# conditions with sklearn KMeans(random_state=42). numpy's PCG64 is stable across
# versions, but sklearn's KMeans init/algorithm defaults are NOT — regenerating under a
# different sklearn can silently produce DIFFERENT splits, which would break
# comparability with the paper's Tables II-VI *and* between our own workers, with no
# error raised. So: ship the bytes, verify md5, refuse to run on mismatch.
#
# Usage (on each remote server):
#   ORIGIN=lpala@<origin-host> ./bootstrap_agent.sh
#
# Optional:
#   COUNT=3          # runs per agent (omit = drain the sweep)
#   SLURM=1          # wrap the agent in sbatch (REQUIRED on GPU-policed SLURM nodes)
#   REPO=~/FedCMAPSS # where to clone

set -eo pipefail   # no `-u`: conda's activation scripts trip on unbound vars

SWEEP="ngslung/HPO_AGENT_FedCMAPSS/y21gj2xn"
REPO="${REPO:-$HOME/FedCMAPSS}"
ORIGIN_PATH="/home/lpala/fedsky/ext/FedCMAPSS"

# --- reference checksums (origin host, 2026-07-11) ---
read -r -d '' EXPECTED <<'EOF' || true
83c7935083b7013b8539535156e23f6e  output/tasks.json
6e2032e4316b6882269adfed3afb025a  output/cmapss_processed_train_data.csv
ad3b0190e68934c09ab7f5bb7a8cc02b  output/cmapss_processed_test_data.csv
EOF

echo "==> 1/5 repo"
[ -d "$REPO/.git" ] || git clone https://github.com/perceivelab/FedCMAPSS.git "$REPO"
cd "$REPO"

echo "==> 2/5 data (rsync from origin — NOT regenerated)"
if [ -n "$ORIGIN" ]; then
  rsync -av --progress "$ORIGIN:$ORIGIN_PATH/output/" output/
else
  echo "    ORIGIN unset; assuming output/ is already present (shared FS?)"
fi

echo "==> 3/5 INTEGRITY GATE — splits must be bit-identical to the paper's"
if ! echo "$EXPECTED" | md5sum -c --status -; then
  echo "!!! CHECKSUM MISMATCH in output/ — splits differ from the origin host."
  echo "!!! Refusing to start: results would not be comparable to Tables II-VI."
  echo "!!! Re-rsync from origin. Do NOT regenerate the dataset."
  echo "$EXPECTED" | md5sum -c - || true
  exit 1
fi
echo "    OK: tasks.json + both CSVs match the golden copy (10 splits x 5 tasks)."

# main.py is invoked by the sweep with --data_root dataset/FedCMAPSS/ (a relative path),
# so this symlink must exist and the agent must run from the repo root.
ln -sfn ../output dataset/FedCMAPSS
[ -f dataset/FedCMAPSS/tasks.json ] || { echo "!!! symlink broken"; exit 1; }

echo "==> 4/5 env + wandb auth"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -qE "^flvit\s" || conda env create -f env_cuda_latest.yaml -n flvit
conda activate flvit

# ~/.local/lib/pythonX/site-packages shadows the conda env and drags in a stale,
# broken wandb on the system python (observed on worker4: python3.8 + user-site wandb
# -> "SettingsStatic has no attribute log_internal" -> wandb service dies -> runs are
# registered server-side but never log a single step). Hard-fail instead.
export PYTHONNOUSERSITE=1

python - <<'PY' || { echo "!!! env non pulito — vedi sopra"; exit 1; }
import sys, os
env = os.environ.get("CONDA_PREFIX", "")
ok = True
if not sys.prefix.startswith(env) or not env:
    print(f"!!! python NON e' quello dell'env: {sys.executable}"); ok = False
try:
    import wandb, torch
except Exception as e:
    print(f"!!! import fallito: {e}"); sys.exit(1)
if not wandb.__file__.startswith(env):
    print(f"!!! wandb viene da FUORI l'env: {wandb.__file__}"); ok = False
if not wandb.api.api_key:
    print("!!! non loggato — esegui: wandb login <KEY>"); ok = False
print(f"    python {sys.version.split()[0]} | wandb {wandb.__version__} | torch {torch.__version__} "
      f"| cuda {torch.cuda.is_available()}")
sys.exit(0 if ok else 1)
PY

echo "==> 5/5 agent"
export OMP_NUM_THREADS=4
export WANDB__SERVICE_WAIT=300
# `python -m wandb` (not the `wandb` binary): forces the env's interpreter regardless of PATH.
AGENT_CMD="python -m wandb agent ${COUNT:+--count $COUNT} $SWEEP"

if [ "$SLURM" = "1" ]; then
  # A bare GPU process on a policed SLURM node is SIGTERM'd after ~40-80s, so the agent
  # must live inside an allocation. Tune --qos/--time to the local cluster's limits.
  sbatch --job-name=hpo-feddyn --partition=gpu --gres=gpu:1 --cpus-per-task=8 \
         --time=05:45:00 --wrap="cd $REPO && source \$(conda info --base)/etc/profile.d/conda.sh && conda activate flvit && $AGENT_CMD"
  echo "    submitted via sbatch"
else
  echo "    running agent in foreground: $AGENT_CMD"
  exec $AGENT_CMD
fi
