# Mappa delle campagne — cosa stiamo provando, e cosa no

Aggiornato: 2026-07-14

## Risposta breve alla domanda

**Stiamo testando gli iperparametri di FedDyn. Di FedCross non è partito nulla.**

L'unico sweep lanciato è lo **Stage 1a** della Campagna 1: 24 run che variano un solo
parametro vero (`alpha`, la forza del regolarizzatore dinamico di FedDyn) più il learning
rate. FedCross è la Campagna 4, non ancora avviata.

Non stiamo indagando "i problemi di FedCross": nell'articolo **FedCross vince** (Tabelle
III-V). Su FedCross la domanda è diversa e più debole — *"la sua configurazione di default
è ottimale o c'è margine?"* — ed è l'ultima della lista.

---

## Le 4 campagne: cosa chiede ognuna

| # | Metodo | Domanda | Stato |
|---|---|---|---|
| **1** | **FedDyn** | È il **peggiore in tutti e 5 i task** (2-3× peggio del migliore). È colpa di un iperparametro sbagliato o del metodo in sé? | **IN CORSO** (Stage 1a) |
| 2 | FedAvg | Crolla sul Task B con la CNN (RMSE 46.78). È un problema di learning rate? | non lanciata |
| 3 | SCAFFOLD | Sottoperforma sul Task C (32.15 vs 20.18 di FedCross). Colpa del server LR? | non lanciata |
| 4 | FedCross | **Vince.** Il default è ottimale o c'è margine? (sanity check) | non lanciata |

L'ordine è per priorità: FedDyn è il fallimento più grosso e quello con l'ipotesi più
precisa, quindi va per primo.

---

## Perché FedDyn per primo: l'ipotesi

FedDyn aggiunge alla loss MSE un termine di regolarizzazione dinamica pesato da `alpha`:

```
loss += alpha/2 * ||theta - theta_global||^2  -  <theta, old_grad>
```

Se `alpha` è troppo grande, **la regolarizzazione domina la loss MSE** e il modello smette
di imparare la RUL: predice qualcosa di quasi costante. È esattamente il sintomo che
l'articolo descrive (§IV.C): *"FedDyn exhibits an almost-constant trend"*.

**Il fatto chiave che ho scoperto leggendo il codice:** nessuno sweep dell'articolo passa
mai `--alpha`. Quindi le loro cifre FedDyn sono state prodotte con il **default di argparse,
`alpha = 1.0`** (`main.py:619`) — non 0.01 come dice il README dell'agente. Un `alpha` di
1.0 contro una loss MSE è enorme. L'ipotesi ne esce **rafforzata**, e dice anche in che
direzione cercare: **verso il basso**.

## Cosa varia lo Stage 1a (sweep `y21gj2xn`, 24 run)

Tutto il resto è **bloccato**, così qualsiasi variazione di RMSE è attribuibile solo a
questi due assi:

| Asse | Valori | Nota |
|---|---|---|
| `alpha` | `1e-4, 1e-3, 1e-2, 1e-1, 0.5, **1.0**` | 1.0 = il default dell'articolo, in cima alla griglia |
| `local_learning_rate` | `5e-4, 1e-3, 5e-3, 1e-2` | interagisce con alpha |

Costanti: `LSTM_RUL`, task `A`, split `0`, `local_epochs=1`, 100 round.

**Baseline da battere:** Tabella II, FedDyn LSTM Task A = **23.45 ± 0.72**.
**Successo:** RMSE ≤ 22.

Il verdetto sarà uno di due:
- **"HP-driven"** → abbassando alpha l'RMSE crolla ⇒ il fallimento di FedDyn
  nell'articolo è un artefatto di tuning, e le conclusioni vanno riviste.
- **"algorithm-driven"** → l'RMSE non si muove ⇒ FedDyn è davvero inadatto al problema,
  e l'articolo ne esce **confermato**.

Entrambi gli esiti sono un risultato pubblicabile. Non stiamo cercando di "salvare"
FedDyn: stiamo cercando di capire se il suo fallimento è reale.

## Perché solo 24 run e non 864

La griglia piena (864 run) costa **~975 GPU-ore** misurate — da sola sfonda il tetto di
500 GPU-ore che il prompt stesso impone. Lo Stage 1a risponde alla domanda centrale per
**~15 GPU-ore**. Se alpha è la leva, si vede subito. Poi si allarga solo attorno al
vincitore (Stage 1b: task B, AttBiGRU, split `[0,1,3]`, local_epochs `{1,5,10}`).

---

## Dove trovare gli sweep

| Sweep | Cosa | Link |
|---|---|---|
| `qaehrlpt` | **Stage 1a FedDyn (24 run) — quello attivo** | https://wandb.ai/ngslung/HPO_AGENT_FedCMAPSS/sweeps/qaehrlpt |
| `y21gj2xn` | primo tentativo, fermato: 10/24 config bruciate da guasti d'ambiente (wandb rotto, symlink dati mancante) senza produrre metriche. **Non riutilizzabile**: un grid sweep non rimette in coda le config fallite. | — |

Progetto W&B: `ngslung/HPO_AGENT_FedCMAPSS`.

## Config sul disco

| File | Run | Stato |
|---|---|---|
| `sweeps/HPO_AGENT/feddyn_stage1a_screen.yaml` | 24 | **lanciato** (`y21gj2xn`) |
| `sweeps/HPO_AGENT/feddyn_grid_alpha.yaml` | 864 | pronto, non lanciato (Stage 1b) |
| `sweeps/HPO_AGENT/fedavg_taskB_lr.yaml` | 90 | pronto, non lanciato |
| `sweeps/HPO_AGENT/scaffold_lr_pairs.yaml` | 144 | pronto, non lanciato |
| `sweeps/HPO_AGENT/fedcross_middleware.yaml` | 144 | pronto, non lanciato — **ridisegnato**, vedi sotto |

---

## Nota su FedCross (Campagna 4): la domanda originale era impossibile

Il prompt chiedeva di variare `n_middleware` su `{3,5,7}`. **Quel parametro non esiste.**
Il numero di modelli middleware è derivato, non configurabile (`servercross_rul.py:32`):

```python
self.w_locals_num = self.num_join_clients
```

ed è quindi **incollato al numero di client del task**. Non è impostabile a 3/5/7 senza
modificare `servercross_rul.py`, cosa che le regole vietano.

Su tua approvazione la Campagna 4 è stata ridisegnata sui veri parametri di FedCross
(`fedcross_alpha`, `collaberative_model_select_strategy`), che rispondono alla stessa
domanda: *il default è ottimale?* Ma resta **l'ultima in coda**.

---

## Stato reale in questo momento (attenzione)

Lo sweep `y21gj2xn` **non ha ancora prodotto un solo numero**. 8 config su 24 sono già
state consumate da run fallite per problemi di ambiente sui worker (wandb rotto, dati
mancanti), non da problemi scientifici:

- 4 `crashed`, 1 `failed`, 3 `running` senza metriche (`steps=0`)

**Un grid sweep di W&B non rimette in coda le config fallite.** Le combinazioni bruciate
(tutta la riga `alpha=1e-4`, quasi tutta `alpha=1e-3`) sono perse. Quando l'ambiente sarà
stabile va **ricreato uno sweep pulito**, altrimenti la griglia resta con dei buchi
proprio nella zona di `alpha` basso, che è quella che ci interessa di più.
