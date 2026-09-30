"""Scorers de referencia para avalia_normalidade.py.

`ScorerMax` e COPIA FIEL de scripts/pdm_fisico/ablacao.py::ScorerMax -- o scorer
dos canais `t` e `p` do detector publicado. Copiado, nao importado, porque
`ablacao.py` chama `main()` no nivel do modulo (sem `if __name__ == "__main__"`):
importar dispara a ablacao inteira e exige grade2min.parquet no diretorio. Se o
original mudar, atualize aqui.

Rode com --scorer-path apontando para ESTE diretorio e o de scripts/pdm_fisico
(que tem o pacote cabiunas_pdm):
    --scorer-path .claude/skills/avalia-normalidade/scripts:scripts/pdm_fisico
    --scorer scorers:ScorerMax --col pca_recon
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cabiunas_pdm import scoring as S


class ScorerMax(S.MultivariateScorer):
    """Maximo do erro por sensor, normalizado pelo p99 do proprio sensor no
    baseline, com piso PHI (nenhum sensor mais de 1/PHI vezes mais sensivel que
    a mediana da familia)."""
    PHI = 0.10

    def fit(self, baseline: pd.DataFrame) -> "ScorerMax":
        super().fit(baseline)
        X = baseline.dropna()[self.cols]
        Xs = self.scaler.transform(X)
        e = (Xs - self.pca.inverse_transform(self.pca.transform(Xs))) ** 2
        p = np.nanpercentile(e, 99, axis=0)
        self.sens_p99_ = np.maximum(p, self.PHI * np.nanmedian(p))
        r, _ = self._raw_scores(X)
        self.recon_p99 = float(np.nanpercentile(r, 99))
        return self

    def _raw_scores(self, df):
        X = df[self.cols]
        mask = X.notna().all(axis=1).to_numpy()
        recon = np.full(len(X), np.nan); maha = np.full(len(X), np.nan)
        if mask.any():
            Xs = self.scaler.transform(X[mask])
            e = (Xs - self.pca.inverse_transform(self.pca.transform(Xs))) ** 2
            if getattr(self, "sens_p99_", None) is None:
                recon[mask] = np.mean(e, axis=1)          # durante o fit inicial
            else:
                recon[mask] = np.max(e / self.sens_p99_, axis=1)
            maha[mask] = self.lw.mahalanobis(Xs)
        return recon, maha
