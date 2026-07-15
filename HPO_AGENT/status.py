#!/usr/bin/env python
"""Monitor GENERICO per qualsiasi sweep HPO (FedDyn/SCAFFOLD/FedAvg/FedCross).

    python HPO_AGENT/status.py <sweep_id> [--watch] [--every 60]
    python HPO_AGENT/status.py pt7gmli9

Tre sezioni:
  STATO     - avanzamento, host (dgx/worker4/...), agent vivi
  PROBLEMI  - run crashate, run che NON loggano metriche, config perse
  RISULTATI - run concluse a 100 round, ordinate per RMSE, vs baseline del paper

NB: la metrica su W&B e' `global/test_rmse` (serverbase_rul.py:138), NON la stringa
"Global Test RMSE" stampata a video. Interrogare la chiave sbagliata da' 0 risultati
anche su run sanissime -- ci siamo gia' bruciati una volta.
"""
import sys, argparse, time
from collections import Counter, defaultdict
import wandb

ENTITY_PROJECT = "ngslung/HPO_AGENT_FedCMAPSS"
KEY = "global/test_rmse"
# tutte le metriche globali loggate da serverbase_rul.py:133
KEYS = ["global/test_rmse", "global/train_rmse", "global/test_nasa_score", "global/train_nasa_score"]
FINAL_ROUND = 100   # una run "conclusa" arriva a round 100 (history len ~101)

# Baseline del paper da battere, per campagna. (task, model) -> (paper_rmse, best_in_row, criterio)
PAPER = {
    # SCAFFOLD - Tabella IV/VI
    ("C", "AttBiGRU_RUL"): ("SCAFFOLD C AttBiGRU", 32.15, 20.18, "<= 20.18 (FedCross)"),
    ("C", "LSTM_RUL"):     ("SCAFFOLD C LSTM",     27.31, 17.28, "<= 17.28 (FedCross)"),
    ("E", "AttBiGRU_RUL"): ("SCAFFOLD E AttBiGRU", 32.74, 32.24, "<= 32.24 (FedAvg)"),
    ("E", "LSTM_RUL"):     ("SCAFFOLD E LSTM",     28.92, 27.30, "<= 27.30 (FedAvg)"),
}


def final_rmse(run):
    h = run.history(keys=[KEY], pandas=False)
    return (len(h), h[-1][KEY] if h else None)


def last_stats(run):
    """Ultimo punto loggato: round + tutte le metriche globali. {} se non ha ancora loggato."""
    h = run.history(keys=KEYS, pandas=False)
    if not h:
        return {"round": 0}
    last = h[-1]
    return {
        "round": len(h),
        "test_rmse": last.get("global/test_rmse"),
        "train_rmse": last.get("global/train_rmse"),
        "test_nasa": last.get("global/test_nasa_score"),
    }


def cfg_str(c):
    """Etichetta compatta della config (solo gli assi variati piu' comuni)."""
    parts = [f"{c.get('task')}", f"{str(c.get('model','')).replace('_RUL','')}"]
    for k, sym in [("alpha", "a"), ("local_learning_rate", "lr"),
                   ("server_learning_rate", "slr"), ("local_epochs", "le"),
                   ("fedcross_alpha", "ca"), ("collaberative_model_select_strategy", "cmss")]:
        if c.get(k) is not None:
            parts.append(f"{sym}={c.get(k)}")
    return " ".join(parts)


def snapshot(sweep_id, detail=False):
    s = wandb.Api().sweep(f"{ENTITY_PROJECT}/{sweep_id}")
    runs = list(s.runs)
    n = len(runs)
    print("=" * 64)
    print(f"SWEEP {sweep_id}   ({s.config.get('name','?')})   sweep_state={s.state}")
    print("=" * 64)

    # ---------- pre-calcolo ----------
    info = []
    for r in runs:
        rounds, rmse = final_rmse(r)
        info.append((r, rounds, rmse))

    # ---------- STATO ----------
    st = Counter(r.state for r in runs)
    hosts = Counter((r.metadata or {}).get("host", "?") for r in runs)
    # host -> tipo GPU (per sapere DOVE gira: dgx=A100, worker4=..., h100=H100)
    host_gpu = {}
    for r in runs:
        m = r.metadata or {}
        if m.get("host"):
            host_gpu.setdefault(m["host"], m.get("gpu", "?"))
    finished = [(r, rd, v) for (r, rd, v) in info if r.state == "finished" and v is not None]
    running = [(r, rd, v) for (r, rd, v) in info if r.state == "running"]
    print("\n### STATO")
    print(f"  config toccate : {n}")
    print(f"  stati          : {dict(st)}")
    print(f"  concluse (r{FINAL_ROUND}): {len(finished)}")
    print(f"  DOVE gira      :")
    for h, c in hosts.most_common():
        print(f"      {h:10s} {c:>3} run   [{host_gpu.get(h,'?')}]")
    if running:
        rounds_live = [rd for (_, rd, _) in running]
        print(f"  in corso       : {len(running)}  (round: min {min(rounds_live)}, max {max(rounds_live)})")

    # ---------- RUN IN CORSO (dettaglio: round + RMSE corrente) ----------
    if running:
        print(f"\n### RUN IN CORSO ({len(running)})  round/{FINAL_ROUND} + RMSE corrente (parziale)")
        print(f"  {'round':>7} {'test_rmse':>10} {'train':>8}  {'host':>8}  config")
        for r, rd, v in sorted(running, key=lambda t: -t[1]):   # piu' avanti in alto
            st_ = last_stats(r)
            tr = st_.get("test_rmse"); trn = st_.get("train_rmse")
            bar = f"{rd:>3}/{FINAL_ROUND}"
            print(f"  {bar:>7} {(f'{tr:.2f}' if tr is not None else '-'):>10} "
                  f"{(f'{trn:.2f}' if trn is not None else '-'):>8}  "
                  f"{(r.metadata or {}).get('host','?'):>8}  {cfg_str(r.config)}")

    # ---------- PROBLEMI ----------
    print("\n### PROBLEMI")
    crashed = [r for (r, rd, v) in info if r.state in ("crashed", "failed")]
    # run 'running' da un po' ma con 0 round loggati = sospette (wandb muto / no metriche)
    stuck = [(r, rd) for (r, rd, v) in info
             if r.state == "running" and rd == 0 and (r.summary.get("_runtime") or 0) > 300]
    if crashed:
        print(f"  ! {len(crashed)} run CRASHED/FAILED (in un grid sweep le loro config sono perse):")
        cc = Counter((r.config.get("task"), r.config.get("model")) for r in crashed)
        for (tk, mdl), c in cc.most_common(6):
            print(f"      {c}x  task={tk} model={mdl}")
    if stuck:
        print(f"  ! {len(stuck)} run 'running' da >5min con 0 metriche (wandb muto? config bruciata):")
        for r, _ in stuck[:5]:
            print(f"      {r.name[:40]}  host={(r.metadata or {}).get('host','?')}")
    if not crashed and not stuck:
        print("  nessuno.")

    # ---------- RISULTATI ----------
    print("\n### RISULTATI (concluse a r{}, ordinate per RMSE)".format(FINAL_ROUND))
    if not finished:
        print("  nessuna run ancora conclusa a 100 round.")
    else:
        finished.sort(key=lambda t: t[2])
        # migliore per (task, model)
        best = {}
        for r, rd, v in finished:
            k = (r.config.get("task"), r.config.get("model"))
            if k not in best or v < best[k][2]:
                best[k] = (r, rd, v)
        limit = len(finished) if detail else 15
        print(f"  {'RMSE':>7} {'NASA_sc':>9} {'task':>4} {'model':>13} {'lr':>7} {'slr':>4} {'le':>3} {'host':>8}   note")
        for r, rd, v in finished[:limit]:
            c = r.config
            k = (c.get("task"), c.get("model"))
            nasa = last_stats(r).get("test_nasa")
            host = (r.metadata or {}).get("host", "?")
            note = ""
            if k in PAPER:
                name, paper_v, target, crit = PAPER[k]
                if v <= target: note = f"** BATTE {crit}"
                elif v < paper_v: note = f"+ meglio del paper ({paper_v})"
            mark = " <BEST" if k in best and best[k][0].id == r.id else ""
            print(f"  {v:>7.2f} {(f'{nasa:.0f}' if nasa is not None else '-'):>9} "
                  f"{str(c.get('task')):>4} {str(c.get('model')):>13} "
                  f"{str(c.get('local_learning_rate')):>7} {str(c.get('server_learning_rate','-')):>4} "
                  f"{str(c.get('local_epochs')):>3} {host:>8}   {note}{mark}")
        if not detail and len(finished) > 15:
            print(f"  ... e altre {len(finished)-15} concluse  (usa --detail per vederle tutte)")

        # riepilogo vs paper per le celle-criterio
        print("\n  -- vs criterio del paper --")
        for k, (name, paper_v, target, crit) in PAPER.items():
            if k in best:
                r, rd, v = best[k]
                verdict = "BATTUTO" if v <= target else ("meglio del paper" if v < paper_v else "sopra")
                print(f"    {name:22s}: best {v:5.2f}  (paper {paper_v}, criterio {crit}) -> {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_id")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--detail", action="store_true", help="mostra TUTTE le run concluse, non solo le 15 migliori")
    a = ap.parse_args()
    while True:
        try:
            snapshot(a.sweep_id, detail=a.detail)
        except Exception as e:
            print(f"errore: {e}", file=sys.stderr)
        if not a.watch:
            break
        time.sleep(a.every)
