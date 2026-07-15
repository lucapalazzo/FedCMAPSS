# Sweep pronti — comandi di lancio

Progetto W&B: `ngslung/HPO_AGENT_FedCMAPSS`

| # | Campagna | Sweep ID | Run | Durata/run | Su worker4 (14 agent) | Stato |
|---|---|---|---|---|---|---|
| 1a | FedDyn — `alpha` × `lr` (screen) | **`qaehrlpt`** | 24 | ~30 min | ~1 h | **in corso** |
| 2 | FedAvg Task B — `lr` × `local_epochs` | `p4vwqwzr` | 90 | ~30–100 min | ~7 h | pronto |
| 3 | SCAFFOLD — `lr` × `server_lr` | `3r5f1vsi` | 144 | ~70–100 min | ~14 h | pronto |
| 4 | FedCross — `fedcross_alpha` × strategia | `fcvtz8ec` | 144 | ~70 min | ~12 h | pronto |
| 1b | FedDyn — griglia estesa | *non creato* | ≤864 | — | — | **dipende da 1a** |

Le campagne 2–4 hanno `local_epochs` 5 e 10, quindi ogni run costa **2–3× lo Stage 1a**.

---

## ⚠️ Prima di lanciare 2–4: aspetta il controllo `alpha=1.0`

Tutte le campagne condividono lo stesso setup (`output_sigmoid=true`, batch size, 100 round,
preprocessing). Lo Stage 1a contiene l'**unico controllo** che verifica che quel setup
riproduca davvero l'articolo: la cella `alpha=1.0` (il default del paper) deve dare
**RMSE ≈ 23.45** (Tabella II, FedDyn LSTM Task A).

- Se **riproduce** → il setup è valido, e le campagne 2–4 sono interpretabili.
- Se **non riproduce** → c'è un confondente che affligge *tutte* le campagne, e lanciarle
  significherebbe spendere **~33 ore di macchina** per numeri non confrontabili con le
  Tabelle II–VI.

Costa ~1 ora aspettare. Non vale la pena rischiare 33 ore.

---

## Lancio (su worker4, dalla root del repo)

Una campagna per volta: worker4 ha **16 core → ~14 agent**. Lanciarne due insieme
significa dimezzare gli agent per ciascuna, senza guadagno.

```bash
# Campagna 2 — FedAvg Task B (90 run)
SWEEP=ngslung/HPO_AGENT_FedCMAPSS/p4vwqwzr PER_GPU=7 GPUS="0 1" COUNT=7 THREADS=1 ./start_agents.sh

# Campagna 3 — SCAFFOLD (144 run)
SWEEP=ngslung/HPO_AGENT_FedCMAPSS/3r5f1vsi PER_GPU=7 GPUS="0 1" COUNT=11 THREADS=1 ./start_agents.sh

# Campagna 4 — FedCross (144 run)
SWEEP=ngslung/HPO_AGENT_FedCMAPSS/fcvtz8ec PER_GPU=7 GPUS="0 1" COUNT=11 THREADS=1 ./start_agents.sh
```

`COUNT` = run per agent ≈ `run_totali / 14`, così ogni agent esce da solo a fine lavoro.

## Monitoraggio

`HPO_AGENT/monitor.py` è cablato sullo Stage 1a. Per le altre campagne cambia `SWEEP` in
testa al file — oppure guarda direttamente la UI W&B, che per griglie a più assi è più
leggibile della matrice testuale.

**La chiave della metrica è `global/test_rmse`** (`serverbase_rul.py:138`), non
`"Global Test RMSE"` — quella è solo la stringa stampata a video. Interrogare l'API con la
chiave sbagliata restituisce zero risultati anche su run sanissime.

## Se una run fallisce

Un grid sweep W&B **non rimette in coda le config fallite**. Se l'ambiente si rompe a metà
campagna (wandb, dati, disco), le config bruciate sono perse e va ricreato lo sweep.
Per questo `start_agents.sh` fa un preflight con `wandb.init()` vero: `main.py:182`
intercetta il fallimento di wandb, **disabilita il logging e continua ad allenare** — la run
sembra viva, gira per ore e non produce nulla.
