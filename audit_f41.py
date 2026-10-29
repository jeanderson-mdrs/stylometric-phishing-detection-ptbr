import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Mesma grade descrita na minuta.
PARAM_GRID = {
    "max_depth": [3, 6, 9],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [100, 200, 300]
}

SEED = 42
COL_LABEL = "Label"
COL_GERADOR = "Modelo_Gerador"

# O vetor completo inclui F1 a F60.
FEATURES_COMPLETAS = feature_cols.copy()

def executar_lolo(df, features, nome_cenario):
    resultados = []
    predicoes = []

    geradores = sorted(df[COL_GERADOR].dropna().unique())

    for gerador_teste in geradores:
        inicio = time.time()

        treino = df[df[COL_GERADOR] != gerador_teste].copy()
        teste = df[df[COL_GERADOR] == gerador_teste].copy()

        X_train = treino[features].astype(float)
        y_train = treino[COL_LABEL].astype(int)

        X_test = teste[features].astype(float)
        y_test = teste[COL_LABEL].astype(int)

        # A validação interna usa exclusivamente as três LLMs de treino.
        cv_interna = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED
        )

        classificador = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=1
        )

        busca = GridSearchCV(
            estimator=classificador,
            param_grid=PARAM_GRID,
            scoring="f1",
            cv=cv_interna,
            n_jobs=-1,
            refit=True,
            verbose=0
        )

        busca.fit(X_train, y_train)
        modelo = busca.best_estimator_

        y_pred = modelo.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        ).ravel()

        f1 = f1_score(
            y_test, y_pred, pos_label=1, zero_division=0
        )

        resultados.append({
            "cenario": nome_cenario,
            "gerador_teste": gerador_teste,
            "n_treino": len(treino),
            "n_teste": len(teste),
            "n_features": len(features),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_phishing": precision_score(
                y_test, y_pred, pos_label=1, zero_division=0
            ),
            "recall_phishing": recall_score(
                y_test, y_pred, pos_label=1, zero_division=0
            ),
            "f1_phishing": f1,
            "f1_macro": f1_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "melhores_parametros": json.dumps(
                busca.best_params_, ensure_ascii=False
            ),
            "f1_cv_interno": busca.best_score_,
            "tempo_segundos": round(time.time() - inicio, 2)
        })

        # Guardar predições para comparações pareadas posteriores.
        for pos, (idx, real, previsto) in enumerate(
            zip(teste.index, y_test, y_pred)
        ):
            predicoes.append({
                "cenario": nome_cenario,
                "gerador_teste": gerador_teste,
                "indice_original": idx,
                "y_true": int(real),
                "y_pred": int(previsto)
            })

        print(
            f"{nome_cenario} | {gerador_teste} | "
            f"F1={f1:.4f} | "
            f"F1 interno={busca.best_score_:.4f}"
        )

    df_resultados = pd.DataFrame(resultados)
    df_predicoes = pd.DataFrame(predicoes)

    return df_resultados, df_predicoes


resultados_full, predicoes_full = executar_lolo(
    df=df_feat,
    features=FEATURES_COMPLETAS,
    nome_cenario="A_completo_60"
)

resultados_full.to_csv(
    OUT_DIR / "lolo_A_completo_60.csv",
    index=False
)

predicoes_full.to_csv(
    OUT_DIR / "predicoes_A_completo_60.csv",
    index=False
)

display(resultados_full)

print(
    "F1 médio LOLO:",
    round(resultados_full["f1_phishing"].mean(), 4)
)

FEATURES_SEM_F41 = [
    c for c in FEATURES_COMPLETAS
    if c != F41
]

assert len(FEATURES_SEM_F41) == 59
assert F41 not in FEATURES_SEM_F41

resultados_sem_f41, predicoes_sem_f41 = executar_lolo(
    df=df_feat,
    features=FEATURES_SEM_F41,
    nome_cenario="B_sem_F41"
)

resultados_sem_f41.to_csv(
    OUT_DIR / "lolo_B_sem_F41.csv",
    index=False
)

predicoes_sem_f41.to_csv(
    OUT_DIR / "predicoes_B_sem_F41.csv",
    index=False
)

display(resultados_sem_f41)

def encontrar_feature(numero):
    encontrados = [
        c for c in FEATURES_COMPLETAS
        if re.match(rf"^F{numero}_", c)
    ]
    if len(encontrados) != 1:
        raise ValueError(
            f"Esperada uma coluna F{numero}; encontradas: {encontrados}"
        )
    return encontrados[0]

ATRIBUTOS_CORRELATOS_C1 = [
    encontrar_feature(18),  # densidade de pronomes
    encontrar_feature(37),  # proporção de verbos
    encontrar_feature(45)   # pronomes de segunda/terceira pessoa
]

ATRIBUTOS_CORRELATOS_C2 = ATRIBUTOS_CORRELATOS_C1 + [
    encontrar_feature(58)   # marcadores de urgência
]

FEATURES_C1 = [
    c for c in FEATURES_COMPLETAS
    if c not in [F41] + ATRIBUTOS_CORRELATOS_C1
]

FEATURES_C2 = [
    c for c in FEATURES_COMPLETAS
    if c not in [F41] + ATRIBUTOS_CORRELATOS_C2
]

print("F41:", F41)
print("Removidos no cenário C1:", [F41] + ATRIBUTOS_CORRELATOS_C1)
print("Quantidade de atributos C1:", len(FEATURES_C1))

print("\nRemovidos no cenário C2:", [F41] + ATRIBUTOS_CORRELATOS_C2)
print("Quantidade de atributos C2:", len(FEATURES_C2))

resultados_C1, predicoes_C1 = executar_lolo(
    df_feat,
    FEATURES_C1,
    "C1_sem_F41_correlatos"
)

resultados_C2, predicoes_C2 = executar_lolo(
    df_feat,
    FEATURES_C2,
    "C2_sem_F41_correlatos_urgencia"
)

resultados_C1.to_csv(
    OUT_DIR / "lolo_C1_sem_F41_correlatos.csv",
    index=False
)

predicoes_C1.to_csv(
    OUT_DIR / "predicoes_C1_sem_F41_correlatos.csv",
    index=False
)

resultados_C2.to_csv(
    OUT_DIR / "lolo_C2_sem_F41_correlatos_urgencia.csv",
    index=False
)

predicoes_C2.to_csv(
    OUT_DIR / "predicoes_C2_sem_F41_correlatos_urgencia.csv",
    index=False
)

cenarios = {
    "A_completo_60": resultados_full,
    "B_sem_F41": resultados_sem_f41,
    "C1_sem_F41_correlatos": resultados_C1,
    "C2_sem_F41_correlatos_urgencia": resultados_C2
}

resumo = []

for nome, tabela in cenarios.items():
    resumo.append({
        "cenario": nome,
        "n_atributos": int(tabela["n_features"].iloc[0]),
        "F1_LOLO_medio": tabela["f1_phishing"].mean(),
        "acuracia_media": tabela["accuracy"].mean(),
        "precisao_media": tabela["precision_phishing"].mean(),
        "recall_medio": tabela["recall_phishing"].mean()
    })

df_resumo = pd.DataFrame(resumo)

f1_referencia = df_resumo.loc[
    df_resumo["cenario"] == "A_completo_60",
    "F1_LOLO_medio"
].iloc[0]

df_resumo["delta_F1_vs_completo"] = (
    df_resumo["F1_LOLO_medio"] - f1_referencia
)

df_resumo.to_csv(
    OUT_DIR / "resumo_comparativo_ablação.csv",
    index=False
)

display(
    df_resumo.round(4)
)


comparacao_geradores = resultados_full[
    ["gerador_teste", "f1_phishing"]
].rename(columns={"f1_phishing": "F1_completo"})

for nome, tabela in [
    ("F1_sem_F41", resultados_sem_f41),
    ("F1_sem_correlatos", resultados_C1),
    ("F1_sem_urgencia", resultados_C2)
]:
    comparacao_geradores = comparacao_geradores.merge(
        tabela[["gerador_teste", "f1_phishing"]].rename(
            columns={"f1_phishing": nome}
        ),
        on="gerador_teste",
        how="inner"
    )

display(comparacao_geradores.round(4))

comparacao_geradores.to_csv(
    OUT_DIR / "comparacao_por_gerador.csv",
    index=False
)