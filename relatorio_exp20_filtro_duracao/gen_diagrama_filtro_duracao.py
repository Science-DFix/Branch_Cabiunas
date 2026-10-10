import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

np.random.seed(7)

# eixo de tempo sintetico, cadencia de 30s, ~40 minutos de janela
t0 = pd.Timestamp("2026-01-01 00:00:00")
idx = pd.date_range(t0, periods=80, freq="30s")  # 40 min
minutes = np.arange(len(idx)) * 0.5

threshold = 1.0
baseline = 0.35 + 0.08 * np.random.randn(len(idx))

score = baseline.copy()

# episodio de ruido curto (~2 min = 4 amostras) perto do minuto 8
noise_start, noise_len = 16, 4
score[noise_start:noise_start + noise_len] += 0.9

# precursor real sustentado (~14 min = 28 amostras) a partir do minuto 20
prec_start, prec_len = 40, 28
ramp = np.linspace(0, 1.0, 6)
score[prec_start:prec_start + 6] += ramp
score[prec_start + 6:prec_start + prec_len] += 1.0 + 0.05 * np.random.randn(prec_len - 6)

fig, ax = plt.subplots(figsize=(9, 4.2))

ax.plot(minutes, score, color="#184F95", linewidth=1.4, label="score do modelo")
ax.axhline(threshold, color="#D03B3B", linestyle="--", linewidth=1.2, label="limiar (percentil do treino)")

# sombreamento do episodio de ruido (removido)
noise_mask = score > threshold
noise_region = (minutes >= minutes[noise_start] - 0.25) & (minutes <= minutes[noise_start + noise_len - 1] + 0.25)
ax.axvspan(minutes[noise_start] - 0.25, minutes[noise_start + noise_len - 1] + 0.25,
           color="#EB6834", alpha=0.18)
ax.annotate("episódio curto\n(~2 min < 4,5 min)\n→ removido",
            xy=(minutes[noise_start + noise_len // 2], score[noise_start + 1]),
            xytext=(minutes[noise_start] - 2, 1.65),
            fontsize=9, color="#B24A16",
            arrowprops=dict(arrowstyle="->", color="#B24A16", lw=1.1))

# sombreamento do precursor real (mantido)
above_prec = np.where(score[prec_start:prec_start + prec_len] > threshold)[0]
p0 = prec_start + above_prec[0]
p1 = prec_start + above_prec[-1]
ax.axvspan(minutes[p0] - 0.25, minutes[p1] + 0.25, color="#0CA30C", alpha=0.15)
ax.annotate("episódio sustentado\n(~14 min ≥ 4,5 min)\n→ mantido (alarme)",
            xy=(minutes[(p0 + p1) // 2], score[(p0 + p1) // 2]),
            xytext=(minutes[p0] + 2, 1.9),
            fontsize=9, color="#0A7A0A",
            arrowprops=dict(arrowstyle="->", color="#0A7A0A", lw=1.1))

ax.set_xlabel("tempo (minutos)")
ax.set_ylabel("score de anomalia (u.a.)")
ax.set_ylim(0, 2.3)
ax.set_xlim(0, minutes[-1])
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Filtro de duração mínima: exemplo ilustrativo (dados sintéticos)", fontsize=11)

fig.tight_layout()
fig.savefig("/tmp/claude-1000/-home-dvar-REPO-CABIUNAS/85b4269a-b5f0-450b-9615-b38e2751d331/scratchpad/relatorio_exp20/diagrama_filtro_duracao.png",
            dpi=200)
print("saved")
