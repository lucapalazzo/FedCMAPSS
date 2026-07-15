#!/usr/bin/env bash
# Lancia N agent per GPU sullo stesso sweep. Gli agent si spartiscono le config
# da soli lato W&B: nessun coordinamento, nessun duplicato.
#
# Perche' piu' processi per GPU: il carico e' CPU/dataloader-bound, non GPU-bound.
# Misurato: utilizzo GPU 8-13%, <1 GB di memoria per run. Il limite vero e' la CPU
# (ogni client rilegge il CSV da 27 MB a ogni round), quindi il numero di agent va
# dimensionato sui CORE, non sulle GPU.
#
#   ./start_agents.sh              # default: 4 agent x GPU
#   PER_GPU=6 GPUS="0 1" ./start_agents.sh
#   COUNT=3 ./start_agents.sh      # ogni agent esce dopo 3 run

set -eo pipefail
cd ~/FedCMAPSS
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flvit
export PYTHONNOUSERSITE=1
export WANDB__SERVICE_WAIT=300

SWEEP="${SWEEP:-ngslung/HPO_AGENT_FedCMAPSS/qaehrlpt}"
GPUS="${GPUS:-0 1}"
PER_GPU="${PER_GPU:-4}"
COUNT="${COUNT:-3}"
THREADS="${THREADS:-2}"

N_GPUS=$(echo $GPUS | wc -w)
TOTAL=$(( N_GPUS * PER_GPU ))
CORES=$(nproc)

echo "GPU: [$GPUS]  agent/GPU: $PER_GPU  -> $TOTAL agent concorrenti"
echo "core disponibili: $CORES | OMP_NUM_THREADS=$THREADS -> $(( TOTAL * THREADS )) thread richiesti"
if [ $(( TOTAL * THREADS )) -gt "$CORES" ]; then
  echo "!!! ATTENZIONE: sovraccarico CPU. Gli agent si pesteranno i piedi e ogni run"
  echo "!!! rallentera', annullando il guadagno. Riduci PER_GPU o THREADS."
  exit 1
fi

# preflight: NON basta importare wandb. E' init() che fallisce, e in quel caso
# main.py disabilita il logging in silenzio e allena lo stesso (main.py:182):
# la run risulta "running" ma non logga mai nulla, e la config e' bruciata.
python - <<'PY'
import os, wandb, torch
e = os.environ["CONDA_PREFIX"]
assert wandb.__file__.startswith(e), f"wandb fuori env: {wandb.__file__}"
assert torch.cuda.is_available(), "no CUDA"
r = wandb.init(project="smoke_test", reinit=True); r.log({"x": 1}); r.finish()
print("PREFLIGHT OK:", wandb.__version__, torch.__version__)
PY
test -f dataset/FedCMAPSS/tasks.json || { echo "symlink dati mancante"; exit 1; }

mkdir -p logs
for gpu in $GPUS; do
  for i in $(seq 1 "$PER_GPU"); do
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THREADS \
      nohup python -m wandb agent --count "$COUNT" "$SWEEP" \
      > "logs/agent_g${gpu}_${i}.log" 2>&1 &
    sleep 2   # sfasa le partenze: evita 8 processi che leggono il CSV insieme
  done
done

echo "lanciati $TOTAL agent. log in logs/agent_g*.log"
echo "monitoraggio: python HPO_AGENT/monitor.py --watch"
wait
