#!/bin/bash
# Agent W&B su un nodo SLURM QUALSIASI (h100, dgx, ...): N agent in UNA allocazione.
#
# A differenza di agents_h100.sbatch NON cabla --partition/--qos/--gres: li passi sulla
# riga di comando di sbatch, cosi' lo stesso file vale su cluster diversi. Esempio dgx:
#
#   sbatch --partition=<part> --qos=debug --gres=shard:1 --cpus-per-task=28 \
#          --time=05:45:00 \
#          --export=ALL,SWEEP=ngslung/HPO_AGENT_FedCMAPSS/<id>,AGENTS=24,EST_RUN_MIN=110 \
#          HPO_AGENT/sbatch/agents_slurm.sbatch
#
# PRIMA di usarlo verifica su dgx (i valori di h100 NON sono garantiti):
#   sacctmgr -n show qos format=Name,MaxWall,GrpTRES,MaxJobsPU   # nomi QOS + wall + limiti
#   sinfo -N -o "%N %c %G"                                       # gres: c'e' 'shard'?
#   nproc                                                        # per dimensionare AGENTS
# Se dgx non espone 'shard', usa --gres=gpu:1 (una GPU intera: il carico e' CPU-bound,
# 250 MB VRAM/agent, quindi va bene lo stesso).
#
#SBATCH --job-name=hpo-agents
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# --partition / --qos / --gres / --cpus-per-task / --time: dalla riga di comando sbatch.

set -eo pipefail   # niente -u: gli script di attivazione conda usano var non definite

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flvit
# la working dir e' quella da cui hai lanciato sbatch: DEVE essere la root del repo
[ -f system/main.py ] || { echo "!!! lancia sbatch dalla ROOT del repo FedCMAPSS"; exit 1; }

export PYTHONNOUSERSITE=1
export WANDB__SERVICE_WAIT=300
export OMP_NUM_THREADS=1

SWEEP="${SWEEP:?serve --export ...,SWEEP=ngslung/HPO_AGENT_FedCMAPSS/<id>}"
AGENTS="${AGENTS:-24}"
EST_RUN_MIN="${EST_RUN_MIN:-110}"   # 40 per Stage 1a (le=1), 110 per campagne le=5/10

echo "host=$(hostname) job=$SLURM_JOB_ID cpus=${SLURM_CPUS_PER_TASK:-?} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "sweep=$SWEEP agents=$AGENTS est_run=${EST_RUN_MIN}min"

if [ -n "$SLURM_CPUS_PER_TASK" ] && [ "$AGENTS" -gt "$SLURM_CPUS_PER_TASK" ]; then
  echo "!!! AGENTS ($AGENTS) > cpus-per-task ($SLURM_CPUS_PER_TASK): sovraccarico CPU."; exit 1
fi

# --- preflight: NON basta importare wandb. E' init() che fallisce, e main.py:184 in quel
#     caso disabilita il logging e allena lo stesso -> run "running" che non produce nulla,
#     config bruciata (un grid sweep non la rimette in coda). Inoltre verifica la patch. ---
python - <<'PY'
import os, wandb, torch, inspect
from flcore.clients.clientscaffold_rul import clientSCAFFOLD_RUL
e = os.environ["CONDA_PREFIX"]
assert wandb.__file__.startswith(e), f"wandb fuori env: {wandb.__file__}"
assert torch.cuda.is_available(), "no CUDA"
src = inspect.getsource(clientSCAFFOLD_RUL)
assert "max(self.num_batches, 1)" in src, "!!! codice NON patchato: manca il fix SCAFFOLD /0"
r = wandb.init(project="smoke_test", reinit=True); r.log({"x": 1}); r.finish()
print("PREFLIGHT OK:", wandb.__version__, torch.__version__, "| patch SCAFFOLD presente")
PY
test -f dataset/FedCMAPSS/tasks.json || { echo "!!! symlink dati mancante"; exit 1; }

END_TS=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oP 'EndTime=\K[^ ]+' | head -1)
DEADLINE=$(date -d "$END_TS" +%s 2>/dev/null || echo $(( $(date +%s) + 345*60 )))

# GUARD ANTI-WALL: una run uccisa dal wall diventa 'crashed' e la sua config e' persa per
# sempre. Ogni agent controlla il tempo residuo prima di iniziare una run.
run_agent() {
  local id=$1 quick=0
  while true; do
    local left=$(( (DEADLINE - $(date +%s)) / 60 ))
    if [ "$left" -lt "$EST_RUN_MIN" ]; then
      echo "[agent $id] restano ${left}min < ${EST_RUN_MIN}min: esco."; return 0; fi
    local t0=$(date +%s)
    python -m wandb agent --count 1 "$SWEEP" || { echo "[agent $id] errore"; return 0; }
    if [ $(( $(date +%s) - t0 )) -lt 120 ]; then
      quick=$(( quick + 1 )); [ "$quick" -ge 2 ] && { echo "[agent $id] sweep esaurito"; return 0; }
      sleep 10
    else quick=0; fi
  done
}

for i in $(seq 1 "$AGENTS"); do run_agent "$i" & sleep 2; done
wait
echo "tutti gli agent hanno terminato."
