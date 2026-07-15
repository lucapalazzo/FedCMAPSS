# Setup di un nuovo nodo SLURM (dgx) per gli sweep — runbook

Obiettivo: far girare gli agent W&B su `dgx` usando gli shard della QOS `debug`.
Ogni passo ha una verifica: se la verifica non passa, FERMATI, non proseguire.

Riferimento h100 (sorgente di verita' dei dati): `10.97.9.93`
Fork con il codice patchato: `https://github.com/lucapalazzo/FedCMAPSS.git` (branch `hpo-agent-fixes`)
Sweep SCAFFOLD attivo: `ngslung/HPO_AGENT_FedCMAPSS/pt7gmli9`

---

## 1. Repo — clona il FORK e prendi il branch patchato

Non clonare perceivelab: il fork ha gia' i 3 fix + tutti gli strumenti.

```bash
cd ~
git clone https://github.com/lucapalazzo/FedCMAPSS.git
cd FedCMAPSS
git checkout hpo-agent-fixes
```

Verifica — i 3 file patchati devono avere questi md5 (identici a h100):
```bash
md5sum system/main.py system/flcore/trainmodel/rul_model_factory.py \
       system/flcore/clients/clientscaffold_rul.py
```
```
451273768ccd6d3d76b98d1fb6c83f55  system/main.py
55a80a255342b89645ac2b01d17d4310  system/flcore/trainmodel/rul_model_factory.py
b24f45c1d84b8a2cff47bfcce343e39f  system/flcore/clients/clientscaffold_rul.py
```

## 2. Dati — rsync da h100 (NON rigenerare)

I dati (output/) non sono in git. Copiali dalla golden copy. Se dgx condivide il
filesystem home con h100, salta e fai puntare il symlink alla copia esistente.

```bash
rsync -av lpala@10.97.9.93:/home/lpala/fedsky/ext/FedCMAPSS/output/ output/
ln -sfn ../output dataset/FedCMAPSS
```

Verifica INTEGRITA' — gli split devono essere identici a quelli del paper:
```bash
md5sum -c - <<'EOF'
83c7935083b7013b8539535156e23f6e  output/tasks.json
6e2032e4316b6882269adfed3afb025a  output/cmapss_processed_train_data.csv
ad3b0190e68934c09ab7f5bb7a8cc02b  output/cmapss_processed_test_data.csv
EOF
test -f dataset/FedCMAPSS/tasks.json && echo "symlink OK"
```
Se un md5 non torna: NON proseguire, ri-rsync. Split diversi = risultati non
confrontabili con le Tabelle II-VI.

## 3. Ambiente conda + wandb

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -qE '^flvit\s' || conda env create -f env_cuda_latest.yaml -n flvit
conda activate flvit
export PYTHONNOUSERSITE=1            # tiene fuori ~/.local (ci ha gia' rotto worker4)
wandb login                          # incolla la tua API key se chiede
```

Verifica che wandb sia quello dell'env e che init() FUNZIONI (non basta importarlo):
```bash
python -c "import wandb,os; e=os.environ['CONDA_PREFIX']; \
assert wandb.__file__.startswith(e), wandb.__file__; \
r=wandb.init(project='smoke_test'); r.log({'x':1}); r.finish(); print('WANDB OK')"
```
Se non stampa `WANDB OK`: `pip install -U 'wandb==0.20.1'` e riprova.

## 4. Smoke test — la patch SCAFFOLD gira senza crash?

```bash
python -u system/main.py --dataset FedCMAPSSWindow --data_root dataset/FedCMAPSS/ \
  --task E --split 0 --model LSTM_RUL --algorithm SCAFFOLD_RUL \
  --local_learning_rate 0.01 --server_learning_rate 1.0 --local_epochs 5 \
  --output_sigmoid true -gr 2 2>&1 | grep -iE "RMSE|Error|Traceback|nan"
```
Deve stampare `Global Test RMSE` senza `nan`/`Traceback`. (Task E crashava con /0
sul codice non patchato.)

## 5. Scopri i parametri SLURM di dgx (NON assumere quelli di h100)

```bash
sacctmgr -n show qos format=Name%12,MaxWall%10,GrpTRES%14,MaxJobsPU%8   # QOS: nome, wall, limiti
sinfo -N -o "%N %P %c %G"                                               # partizione + gres (c'e' 'shard'?)
nproc                                                                   # per dimensionare AGENTS
```
Annota: nome partizione, nome QOS debug, MaxWall (il --time deve stare sotto),
e se il gres 'shard' esiste. Se NON esiste, usa `--gres=gpu:1`.

## 6. Lancia gli agent (shard debug)

Dalla ROOT del repo. Adatta partition/qos/time ai valori del passo 5.
`EST_RUN_MIN=110` perche' SCAFFOLD ha local_epochs 5/10 (~70-100 min/run).

```bash
sbatch --partition=<PART> --qos=debug --gres=shard:1 --cpus-per-task=28 \
       --time=<sotto il MaxWall, es 05:45:00> \
       --export=ALL,SWEEP=ngslung/HPO_AGENT_FedCMAPSS/pt7gmli9,AGENTS=24,EST_RUN_MIN=110 \
       HPO_AGENT/sbatch/agents_slurm.sbatch
```

Lo script fa da solo: preflight (wandb init + presenza patch), guard anti-wall
(non inizia una run che non finirebbe entro il wall), e spartizione config via W&B.

Se la QOS debug ammette 1 solo job/utente e serve piu' tempo delle 6h, incatena i
job: aggiungi `--array=1-6%1` (6 job da 5h45, uno alla volta).

## 7. Verifica che stia loggando (dal tuo portatile o da h100)

```bash
python HPO_AGENT/monitor.py     # NB: cambia SWEEP in testa al file per puntare a pt7gmli9
```
Oppure la UI: https://wandb.ai/ngslung/HPO_AGENT_FedCMAPSS/sweeps/pt7gmli9

---

## Aggiornamenti futuri del codice

Quando su h100 committo+pusho una modifica:
```bash
cd ~/FedCMAPSS && git pull fork hpo-agent-fixes   # (o: origin, se hai clonato il fork come origin)
# poi ri-verifica i 3 md5 del passo 1
```
