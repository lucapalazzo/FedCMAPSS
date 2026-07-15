#!/bin/bash
# Agent W&B su h100 — molti agent dentro UNA sola allocazione SLURM.
#
# Perche' cosi':
#  * Su h100 i processi GPU lanciati FUORI da SLURM vengono uccisi dopo ~40-80s dal
#    GPU-policing daemon. Quindi si passa per forza da sbatch.
#  * La quota QOS `train` da' 3 GPU intere ed e' gia' satura (altri utenti). Ma il nodo
#    espone anche `shard:8` (GPU frazionabili) e nessuno shard risulta allocato:
#    --gres=shard:1 viene concesso subito senza toccare il limite gres/gpu=3. Verificato.
#  * Il carico e' CPU-bound: ~250 MB di VRAM e ~2% di GPU per run, ma ~1 core pieno.
#    La risorsa vera di h100 sono i 384 core (280 liberi), non le GPU.
#
# Quindi: 1 shard di GPU + molte CPU + N agent dentro la stessa allocazione.
#
#   sbatch --export=ALL,SWEEP=<id>,AGENTS=24,EST_RUN_MIN=40 HPO_AGENT/sbatch/agents_h100.sbatch
#
#SBATCH --job-name=hpo-agents
#SBATCH --partition=gpu
# QOS `debug`: corsia SEPARATA da `train`. Non intacca la quota gres/gpu=3 di `train`,
# che e' satura e su cui pendono ~76 job `fedgfe` dell'utente. MaxJobsPU=1, ma non e' un
# problema: il disegno di questo script e' 1 solo job con N agent dentro.
#SBATCH --qos=debug
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=28
# QOS `train`: MaxWall=06:00:00 (verificato con sacctmgr). Chiedere di piu' fa restare il
# job PENDING per sempre con reason=QOSMaxWallDurationPerJobLimit.
#SBATCH --time=05:45:00
#SBATCH --output=/home/lpala/fedsky/ext/FedCMAPSS/HPO_AGENT/logs/h100_%j.out
#SBATCH --error=/home/lpala/fedsky/ext/FedCMAPSS/HPO_AGENT/logs/h100_%j.err

# no `set -u`: gli script di attivazione conda usano variabili non definite
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flvit
cd /home/lpala/fedsky/ext/FedCMAPSS

export PYTHONNOUSERSITE=1
export WANDB__SERVICE_WAIT=300
export OMP_NUM_THREADS=1        # una run non satura nemmeno un core: 1 thread basta

SWEEP="${SWEEP:-ngslung/HPO_AGENT_FedCMAPSS/qaehrlpt}"
AGENTS="${AGENTS:-24}"
# Durata stimata di UNA run, in minuti. Serve al guard anti-wall (sotto).
#   Stage 1a (local_epochs=1) ~30 min  -> EST_RUN_MIN=40
#   Campagne 2-4 (local_epochs 5/10)   -> EST_RUN_MIN=110
EST_RUN_MIN="${EST_RUN_MIN:-40}"

echo "host=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "sweep=$SWEEP agents=$AGENTS est_run=${EST_RUN_MIN}min"

# --- dimensionamento ---
# Ogni agent tiene la GPU occupata ~2%. Con lo shard CO-ABITIAMO una GPU fisica con il
# job di un altro utente: oltre ~30 agent la satureremmo, rallentando il loro lavoro e il
# nostro. Il limite NON e' la VRAM (250 MB x 28 = 7 GB su 80 GB): e' CPU + convivenza.
if [ "$AGENTS" -gt "$SLURM_CPUS_PER_TASK" ]; then
  echo "!!! AGENTS ($AGENTS) > cpus-per-task ($SLURM_CPUS_PER_TASK): sovraccarico CPU."
  exit 1
fi

# preflight: non basta importare wandb. E' init() che fallisce, e main.py:182 in quel caso
# DISABILITA il logging e continua ad allenare -> la run sembra viva, gira per ore, non
# produce nulla, e la config resta bruciata.
python - <<'PY'
import os, wandb, torch
e = os.environ["CONDA_PREFIX"]
assert wandb.__file__.startswith(e), f"wandb fuori env: {wandb.__file__}"
assert torch.cuda.is_available(), "no CUDA"
r = wandb.init(project="smoke_test", reinit=True); r.log({"x": 1}); r.finish()
print("PREFLIGHT OK:", wandb.__version__, torch.__version__)
PY
test -f dataset/FedCMAPSS/tasks.json || { echo "!!! symlink dati mancante"; exit 1; }

# Deadline reale dell'allocazione (SLURM la espone; fallback: ora + 5h45m)
END_TS=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oP 'EndTime=\K[^ ]+' | head -1)
DEADLINE=$(date -d "$END_TS" +%s 2>/dev/null || echo $(( $(date +%s) + 345*60 )))

# GUARD ANTI-WALL — il punto piu' importante di questo script.
# `wandb agent --count N` non sa nulla del wall-time: se il job scade a meta' di una run,
# quella run muore `crashed` e in un GRID sweep la sua config NON torna mai in coda.
# Percio' non uso --count N: chiamo `--count 1` in loop e prima di ogni run verifico che
# ci sia tempo per completarla. Meglio un agent che esce in anticipo che una config persa.
run_agent() {
  local id=$1
  local quick=0
  while true; do
    local left=$(( (DEADLINE - $(date +%s)) / 60 ))
    if [ "$left" -lt "$EST_RUN_MIN" ]; then
      echo "[agent $id] restano ${left}min < ${EST_RUN_MIN}min: esco senza iniziare una run che non finirei."
      return 0
    fi
    local t0=$(date +%s)
    python -m wandb agent --count 1 "$SWEEP" || { echo "[agent $id] agent uscito con errore"; return 0; }
    # A sweep esaurito `wandb agent` esce SUBITO e con successo: senza questo controllo il
    # while lo rilancerebbe all'infinito, bruciando CPU a vuoto fino al wall-time.
    # Due uscite consecutive in meno di 2 minuti = non c'e' piu' lavoro.
    if [ $(( $(date +%s) - t0 )) -lt 120 ]; then
      quick=$(( quick + 1 ))
      [ "$quick" -ge 2 ] && { echo "[agent $id] sweep esaurito: esco."; return 0; }
      sleep 10
    else
      quick=0
    fi
  done
}

for i in $(seq 1 "$AGENTS"); do
  run_agent "$i" &
  sleep 2   # sfasa le partenze: evita N processi che parsano insieme il CSV da 27 MB
done

wait
echo "tutti gli agent hanno terminato."
