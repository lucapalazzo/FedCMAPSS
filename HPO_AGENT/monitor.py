#!/usr/bin/env python
"""Monitoraggio dello sweep FedDyn Stage 1a.

    python HPO_AGENT/monitor.py            # stato attuale
    python HPO_AGENT/monitor.py --watch    # aggiorna ogni 60s

ATTENZIONE ALLA CHIAVE: la metrica su W&B si chiama `global/test_rmse`
(serverbase_rul.py:138). La stringa "Global Test RMSE" e' solo quello che il
codice STAMPA a video: interrogare W&B con quella restituisce zero risultati
anche su run perfettamente sane. Mi ci sono gia' bruciato una volta.
"""
import argparse, time, sys
from collections import Counter
import wandb

# jsjdreag = Stage 1a rifatto su codice patchato (asse `lr` finalmente attivo).
# qaehrlpt = versione precedente, girata col bug del factory che forzava lr=0.01:
#            resta valida come misura del solo effetto di `alpha`, ma NON mescolare i due.
SWEEP = "ngslung/HPO_AGENT_FedCMAPSS/jsjdreag"
KEY = "global/test_rmse"
PAPER = 23.45   # Tabella II, FedDyn LSTM Task A
TARGET = 22.0   # soglia di successo
ALPHAS = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
LRS = [0.0005, 0.001, 0.005, 0.01]


def snapshot():
    runs = list(wandb.Api().sweep(SWEEP).runs)
    cells = {}
    for r in runs:
        h = r.history(keys=[KEY], pandas=False)
        rmse = h[-1][KEY] if h else None
        cells[(r.config.get("alpha"), r.config.get("local_learning_rate"))] = (r.state, len(h), rmse)

    print(f"\nsweep {SWEEP.split('/')[-1]} — {len(runs)}/24 config | "
          f"paper={PAPER}  target<={TARGET}")
    print(f"stati: {dict(Counter(r.state for r in runs))}\n")

    # griglia: righe = alpha, colonne = lr. alpha=1.0 e' il DEFAULT del paper.
    #
    # Mostro SEMPRE rmse@round: senza il round la griglia inganna. Le run giovani
    # stanno tutte vicino all'RMSE iniziale (simile per ogni lr), quindi celle con
    # valori quasi identici NON vogliono dire "il lr non conta" -- vogliono dire
    # "queste run sono al round 5". Solo le celle a round 100 sono confrontabili
    # fra loro e con la Tabella II.
    print(f"{'alpha \\ lr':>10s} " + "".join(f"{lr:>14}" for lr in LRS))
    for a in ALPHAS:
        row = f"{a:>10} "
        for lr in LRS:
            c = cells.get((a, lr))
            if c is None:
                row += f"{'.':>14}"
            else:
                state, n, rmse = c
                if rmse is None:
                    row += f"{'avvio':>14}"
                else:
                    done_ = (state == "finished")
                    mark = ("*" if rmse <= TARGET else ("+" if rmse < PAPER else " ")) if done_ else "~"
                    row += f"{rmse:>8.2f}@{n:<3d}{mark:<2}"
        print(row + ("   <-- default del paper" if a == 1.0 else ""))

    done = [c for c in cells.values() if c[0] == "finished" and c[2] is not None]
    print(f"\n  formato: rmse@round   ('~' = ancora in corso, valore NON confrontabile)")
    print(f"  .  non ancora girata")
    print(f"  +  conclusa, meglio del paper ({PAPER})    *  conclusa, sotto la soglia ({TARGET})")
    print(f"\nrun concluse con RMSE: {len(done)}/24")
    if done:
        best = min(done, key=lambda c: c[2])
        print(f"miglior RMSE finora: {best[2]:.2f}")
    # nota: i valori delle run ancora in corso sono PARZIALI (round < 100)
    live = [(k, v) for k, v in cells.items() if v[0] == "running"]
    if live:
        print(f"in corso ({len(live)}): valori parziali, non ancora a 100 round")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true")
    p.add_argument("--every", type=int, default=60)
    a = p.parse_args()
    while True:
        try:
            snapshot()
        except Exception as e:
            print(f"errore: {e}", file=sys.stderr)
        if not a.watch:
            break
        time.sleep(a.every)
