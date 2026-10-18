"""Funil das regras pós-treino (fora da amostra, máquina on): pontos anômalos por canal -> voto -> 45 min -> 48 h -> episódios."""
import sys, json
sys.path.insert(0, "frente_ocsvm")
import pandas as pd
import comum as C
from a4_variabilidade import canais_da_replica
from d1_disputa import combinacoes, TASKS_JSON
from src.cnn1d_ae.scoring import combine_channels_vote, apply_min_duration_filter, apply_refractory

reps = combinacoes(json.loads(TASKS_JSON.read_text()))
cat = C.catalogo()
L = []
for rep in ["prod", "todos_cnn1d", "todos_cnn1d_mensal", "cnn1d_mensal_so_vib", "cnn1d_mensal_so_temp", "cnn1d_mensal_so_oleo"]:
    df = canais_da_replica(reps[rep]); idx, op = df.index, df["operational_state"]
    c = {k: df[k] for k in C.CANAIS[:3]}
    c["c4"] = C.canal_alarme(idx, cat.loc[cat.tag.isin(C.TAGS_C4), "t"], C.REF["janela_c4_h"])
    sel = (idx >= C.SPLIT) & (op == "on").values
    n = lambda s: int(pd.Series(s, index=idx)[sel].astype(bool).sum())
    voto = combine_channels_vote(c, min_votes=2)
    dur = apply_min_duration_filter(pd.DataFrame({"is_anom_point": voto.astype(int)}), C.REF["dur_min"])["is_anom_point"]
    fin = apply_refractory(dur.astype(bool), refractory_minutes=C.REF["refrat_h"] * 60.0)
    cls, _ = C.avalia(fin, op, C.trips(idx))
    f = cls[cls.start >= C.SPLIT]
    L.append(dict(rep=rep, temp=n(c[C.CANAIS[0]]), vib=n(c[C.CANAIS[1]]), oleo=n(c[C.CANAIS[2]]),
                  algum_modelo=n(c[C.CANAIS[0]] | c[C.CANAIS[1]] | c[C.CANAIS[2]]), c4=n(c["c4"]),
                  voto2=n(voto), apos_45min=n(dur), apos_48h=n(fin),
                  episodios=len(f), FP=int((f.classe == "falso_positivo").sum()), deteccoes=int((f.classe == "deteccao").sum()),
                  pontos_on=int(sel.sum())))
T = pd.DataFrame(L).set_index("rep")
pd.set_option("display.width", 250)
print(T.to_string())
print("\n% de pontos on:"); print((T[["temp", "vib", "oleo", "algum_modelo", "voto2", "apos_45min", "apos_48h"]].div(T.pontos_on, axis=0) * 100).round(2).to_string())
T.to_csv(sys.argv[1])
