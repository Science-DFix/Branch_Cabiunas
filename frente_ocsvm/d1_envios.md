# D1 — diário de envios ao Cica

Uma linha por evento, a mais nova embaixo. Fonte da verdade dos IDs: `d1_tasks.json` (o `envia` grava ao enfileirar).
Para retomar: `python frente_ocsvm/d1_disputa.py envia --so <nome>` pula o que já está em `d1_tasks.json`.

| quando (UTC-3) | task | id | evento |
|---|---|---|---|
| 16/10 16:43 | iforest_temp | 2a248bc14b3f40bdb1068d1f7eceae4c | completed 16:52 |
| 16/10 16:48 | iforest_vib | 886f9e65ac314041a37e2f9d44bb491c | completed 17:06 |
| 16/10 16:57 | iforest_oleo | 3387ed70d22445ff83ec7b2f3611a8a0 | completed 17:10 |
| 16/10 17:02 | dense_temp | e1c0b41fc6a246079c9ec94e8382594a | completed 17:17 |
| 16/10 17:11 | dense_vib | 11f7c65d2d9a434680032a659cbcee85 | completed 17:26 |
| 16/10 17:24 | dense_oleo | fc90a0eb928a4cf9ab6b66362e70fc1e | completed 17:33 |
| 16/10 17:32 | cnn1d_temp | 14348f72381043e1a5bd41e5db7e6d4c | completed 17:43 |
| 16/10 17:39 | cnn1d_vib | ee9c1a9e98124a8cbf1dd49675381dea | completed 18:00 |
| 16/10 18:03 | cnn1d_oleo | 80fbbc239b334846a9eadce85c241bb8 | completed 18:15 |
| 16/10 18:13 | cnn1d_mensal_temp | 2c19e1b9dc424f6e80867457cd8f12d9 | completed 19:14 |
| 17/10 21:30 | cnn1d_mensal_vib | 71647eac561e49dc8d3798d47b151baa | enfileirada |
| 17/10 21:30 | cnn1d_mensal_vib | 71647eac561e49dc8d3798d47b151baa | queued |
| 17/10 21:31 | cnn1d_mensal_vib | 71647eac561e49dc8d3798d47b151baa | in_progress |
| 17/10 22:55 | cnn1d_mensal_vib | 71647eac561e49dc8d3798d47b151baa | failed |
| 17/10 23:55 | cnn1d_mensal_vib | 71647eac561e49dc8d3798d47b151baa | FALHA: exit 137 (OOM) no 1º trial do 8º mês (fev/2026); TF/Keras não devolve a memória de cada fit. Correção: 1 processo por mês no walk-forward do cnn1d. Removida do d1_tasks.json para reenvio |
| 17/10 23:51 | cnn1d_mensal_vib | 6fbead3c83fd47958e5d8fcc48016a6d | enfileirada |
| 17/10 23:51 | cnn1d_mensal_vib | 6fbead3c83fd47958e5d8fcc48016a6d | queued |
| 17/10 23:52 | cnn1d_mensal_vib | 6fbead3c83fd47958e5d8fcc48016a6d | in_progress |
| 18/10 02:12 | cnn1d_mensal_vib | 6fbead3c83fd47958e5d8fcc48016a6d | completed |
| 18/10 02:14 | cnn1d_mensal_vib | 6fbead3c83fd47958e5d8fcc48016a6d | completed (10/10 meses; correção do OOM validada no worker) |
| 18/10 02:23 | cnn1d_mensal_oleo | 7a7b77bf45ef48ae827370c3e52c9216 | enfileirada |
| 18/10 02:24 | cnn1d_mensal_oleo | 7a7b77bf45ef48ae827370c3e52c9216 | in_progress |
| 18/10 03:34 | cnn1d_mensal_oleo | 7a7b77bf45ef48ae827370c3e52c9216 | completed |
| 18/10 03:34 | — | — | fila do D1 concluída |
| 18/10 — | — | — | rodada 2 (PCA), commit 67c3f7c: fila pca_temp, pca_vib, pca_oleo, pca_mensal_temp, pca_mensal_vib, pca_mensal_oleo |
| 18/10 11:57 | pca_temp | 8077dc2e288541c481b63da200ec3680 | enfileirada |
| 18/10 11:57 | pca_temp | 8077dc2e288541c481b63da200ec3680 | in_progress |
| 18/10 12:00 | pca_temp | 8077dc2e288541c481b63da200ec3680 | completed |
| 18/10 12:00 | pca_vib | 3a7b65aab93e48128f9db37d9507faca | enfileirada |
| 18/10 12:01 | pca_vib | 3a7b65aab93e48128f9db37d9507faca | in_progress |
| 18/10 12:06 | pca_temp | 8077dc2e288541c481b63da200ec3680 | completed |
| 18/10 12:06 | pca_vib | 3a7b65aab93e48128f9db37d9507faca | in_progress |
| 18/10 12:07 | pca_vib | 3a7b65aab93e48128f9db37d9507faca | completed |
| 18/10 12:14 | pca_oleo | 47964c2e46f04a5f9758c193684c94fb | enfileirada |
| 18/10 12:15 | pca_oleo | 47964c2e46f04a5f9758c193684c94fb | in_progress |
| 18/10 12:17 | pca_oleo | 47964c2e46f04a5f9758c193684c94fb | completed |
| 18/10 12:21 | pca_mensal_temp | d42c0e474a254cd09824ca6582d7f000 | enfileirada |
| 18/10 12:22 | pca_mensal_temp | d42c0e474a254cd09824ca6582d7f000 | in_progress |
| 18/10 12:24 | pca_mensal_temp | d42c0e474a254cd09824ca6582d7f000 | completed |
| 18/10 12:28 | pca_mensal_vib | a6d764a1fc65427788cdc8970eb1399a | enfileirada |
| 18/10 12:28 | pca_mensal_vib | a6d764a1fc65427788cdc8970eb1399a | in_progress |
| 18/10 12:36 | pca_mensal_vib | a6d764a1fc65427788cdc8970eb1399a | completed |
| 18/10 12:41 | pca_mensal_oleo | 37a895cc62f042579b636129f6e6fc01 | enfileirada |
| 18/10 12:42 | pca_mensal_oleo | 37a895cc62f042579b636129f6e6fc01 | in_progress |
| 18/10 12:44 | pca_mensal_oleo | 37a895cc62f042579b636129f6e6fc01 | completed |
| 18/10 12:44 | — | — | fila do D1 concluída |
| 18/10 — | — | — | rodada 1d (CNN-1D mensal desde 01/2025), commit 4773a46: fila cnn1d_mensal_jan_temp, _vib, _oleo |
| 18/10 13:21 | cnn1d_mensal_jan_temp | bbc71b16f97842dcb429bafda2d89c32 | enfileirada |
| 18/10 13:22 | cnn1d_mensal_jan_temp | bbc71b16f97842dcb429bafda2d89c32 | in_progress |
| 18/10 14:31 | cnn1d_mensal_jan_temp | bbc71b16f97842dcb429bafda2d89c32 | completed |
| 18/10 14:35 | cnn1d_mensal_jan_vib | f0701a1675444a6fb406043c4e7257a4 | enfileirada |
| 18/10 14:36 | cnn1d_mensal_jan_vib | f0701a1675444a6fb406043c4e7257a4 | in_progress |
| 18/10 16:17 | cnn1d_mensal_jan_oleo | aab72b804b4e4e228e83d9ca77cea99a | enfileirada em paralelo com a vib (ok do usuário; 3 workers livres) |
| 18/10 17:38 | cnn1d_mensal_jan_vib | f0701a1675444a6fb406043c4e7257a4 | completed |
| 18/10 17:38 | cnn1d_mensal_jan_oleo | aab72b804b4e4e228e83d9ca77cea99a | in_progress |
| 18/10 18:27 | cnn1d_mensal_jan_oleo | aab72b804b4e4e228e83d9ca77cea99a | completed |
| 18/10 18:27 | — | — | fila do D1 concluída |
| 18/10 18:34 | cnn1d_mensal_jan_vib | f0701a1675444a6fb406043c4e7257a4 | FALHA silenciosa: status completed, mas o grupo morreu no 16º período ('process pool terminated abruptly' no filho, 9º trial) e não gerou point_anomalies. Rodou em paralelo com o óleo na mesma máquina/GPU (provável falta de memória). Removida do d1_tasks.json para reenvio sozinha |
| 18/10 18:35 | cnn1d_mensal_jan_vib | 892e93c1698149aebc4e84817217755b | in_progress |
| 18/10 21:27 | cnn1d_mensal_jan_vib | 892e93c1698149aebc4e84817217755b | completed |
| 18/10 21:27 | — | — | fila do D1 concluída |
