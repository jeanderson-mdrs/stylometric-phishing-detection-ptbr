

# ANÁLISE DOS FALSOS NEGATIVOS DA GEMMA — LOLO / XGBOOST
# Reproduz a lógica de treinamento do script original e exporta:
# 1. Métricas por gerador
# 2. Médias dos atributos estilométricos
# 3. Comparação Gemma x outros geradores
# 4. Falsos negativos e acertos da Gemma
# 5. Amostras textuais para análise qualitativa
# 6. Modelo treinado e metadados da execução


import os
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ---------------- CONFIGURAÇÃO ----------------

# Ajuste o caminho caso o arquivo esteja em outra pasta.
CAMINHO_FEATURES = "/content/drive/MyDrive/Colab Notebooks/Datasets/Novo/dataset_completo_features.csv"

PASTA_SAIDA = (
    "/content/drive/MyDrive/Colab Notebooks/"
    "Analise_Gemma_LOLO"
)

# Nome exato ou parte do nome da Gemma, ignorando maiúsculas/minúsculas.
IDENTIFICADOR_GEMMA = "gemma"

SEED = 42
N_SPLITS = 5

PARAM_GRID = {
    "max_depth": [3, 6, 9],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [100, 200, 300],
}

os.makedirs(PASTA_SAIDA, exist_ok=True)

# ---------------- FUNÇÕES AUXILIARES ----------------

def salvar_csv(df, nome):
    caminho = os.path.join(PASTA_SAIDA, nome)
    df.to_csv(caminho, index=False, encoding="utf-8-sig")
    print(f"Arquivo salvo: {caminho}")

def resumo_numerico(df, colunas):
    """Estatísticas descritivas dos atributos."""
    dados = df[colunas].apply(pd.to_numeric, errors="coerce")
    resultado = pd.DataFrame({
        "media": dados.mean(),
        "desvio_padrao": dados.std(),
        "mediana": dados.median(),
        "q1": dados.quantile(0.25),
        "q3": dados.quantile(0.75),
        "minimo": dados.min(),
        "maximo": dados.max(),
        "n_validos": dados.notna().sum(),
    })
    resultado.index.name = "Atributo"
    return resultado.reset_index()

# ---------------- CARREGAMENTO ----------------

if not os.path.exists(CAMINHO_FEATURES):
    raise FileNotFoundError(
        f"Arquivo não encontrado:\n{CAMINHO_FEATURES}\n"
        "Verifique o caminho e monte o Google Drive."
    )

df = pd.read_csv(CAMINHO_FEATURES)

print("=" * 70)
print("1. AUDITORIA DO DATASET")
print("=" * 70)
print("Dimensões:", df.shape)
print("Colunas:", df.columns.tolist())

colunas_obrigatorias = ["Label", "Modelo_Gerador"]
ausentes = [c for c in colunas_obrigatorias if c not in df.columns]
if ausentes:
    raise ValueError(f"Colunas obrigatórias ausentes: {ausentes}")

# Mantém apenas os atributos F1...F60, conforme o desenho experimental.
import re

# Identifica as features F1 a F60 mesmo quando possuem nomes descritivos.
features = [
    c for c in df.columns
    if re.match(r"^F(?:[1-9]|[1-5][0-9]|60)_", c)
]

# Ordena numericamente pelo identificador da feature.
features = sorted(
    features,
    key=lambda c: int(re.match(r"^F(\d+)_", c).group(1))
)

if len(features) != 60:
    raise ValueError(
        f"Esperadas 60 features F1–F60, mas foram encontradas "
        f"{len(features)}: {features}"
    )

print(f"Total de features identificadas: {len(features)}")
print(features)

if not features:
    raise ValueError(
        "Não encontrei colunas F1...F60. "
        "Confirme se o arquivo é dataset_completo_features.csv."
    )

if len(features) != 60:
    print(
        f"ATENÇÃO: foram encontradas {len(features)} features, "
        "e não 60. Confira o esquema do dataset."
    )

# Converte atributos para valores numéricos.
df[features] = df[features].apply(pd.to_numeric, errors="coerce")

if df[features].isna().any().any():
    colunas_nan = df[features].columns[
        df[features].isna().any()
    ].tolist()
    raise ValueError(
        "Existem valores ausentes ou não numéricos nas features. "
        f"Colunas afetadas: {colunas_nan}. "
        "Investigue antes de treinar; este script não os imputa."
    )

print("\nDistribuição por gerador e rótulo:")
print(pd.crosstab(df["Modelo_Gerador"], df["Label"]))

print("\nGeradores identificados:")
for gerador in df["Modelo_Gerador"].dropna().unique():
    print(" -", repr(gerador))

gemma_matches = [
    g for g in df["Modelo_Gerador"].dropna().unique()
    if IDENTIFICADOR_GEMMA.lower() in str(g).lower()
]
if len(gemma_matches) != 1:
    raise ValueError(
        "Não foi possível identificar exatamente um gerador Gemma. "
        f"Correspondências encontradas: {gemma_matches}. "
        "Ajuste IDENTIFICADOR_GEMMA no início do script."
    )

GEMMA = gemma_matches[0]
print("\nGerador isolado para teste:", GEMMA)

# Confirma codificação binária.
labels = set(df["Label"].dropna().unique())
if not labels.issubset({0, 1, 0.0, 1.0}):
    raise ValueError(
        f"Label deve estar codificado como 0/1. Valores encontrados: {labels}"
    )

# Hash para documentar a versão exata do arquivo utilizado.
with open(CAMINHO_FEATURES, "rb") as f:
    sha256_dataset = hashlib.sha256(f.read()).hexdigest()

print("SHA256 do dataset:", sha256_dataset)
print("Versão XGBoost:", xgb.__version__)
print("Versão pandas:", pd.__version__)
print("Versão NumPy:", np.__version__)

# ---------------- TREINAMENTO LOLO DA GEMMA ----------------
# Segue o script original: Gemma isolada para teste; demais
# geradores para treino e GridSearchCV com cv=5 e scoring='f1'.
# Não usa metadados como features.
# Não inclui mensagens da Gemma no treinamento.

print("\n" + "=" * 70)
print("2. TREINAMENTO E AVALIAÇÃO LOLO DA GEMMA")
print("=" * 70)

teste = df[df["Modelo_Gerador"] == GEMMA].copy()
treino = df[df["Modelo_Gerador"] != GEMMA].copy()

X_train = treino[features].copy()
y_train = treino["Label"].astype(int)

X_test = teste[features].copy()
y_test = teste["Label"].astype(int)

if y_train.nunique() != 2:
    raise ValueError("O conjunto de treinamento precisa conter as duas classes.")

if y_test.nunique() != 2:
    print(
        "ATENÇÃO: o conjunto de teste da Gemma não contém as duas classes. "
        "Algumas métricas podem não ser informativas."
    )

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

modelo_base = xgb.XGBClassifier(
    random_state=SEED,
    eval_metric="logloss"
)

grid = GridSearchCV(
    estimator=modelo_base,
    param_grid=PARAM_GRID,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    refit=True,
    verbose=1
)

grid.fit(X_train, y_train)
modelo = grid.best_estimator_

pred = modelo.predict(X_test).astype(int)
prob = modelo.predict_proba(X_test)[:, 1]

print("\nMelhores parâmetros:", grid.best_params_)
print("Melhor F1 médio na validação interna:", grid.best_score_)
print("Acurácia Gemma:", accuracy_score(y_test, pred))
print("Precisão Gemma:", precision_score(y_test, pred, zero_division=0))
print("Recall Gemma:", recall_score(y_test, pred, zero_division=0))
print("F1 Gemma:", f1_score(y_test, pred, zero_division=0))
print("\nRelatório de classificação:")
print(classification_report(y_test, pred, zero_division=0))

print("Matriz de confusão [[TN, FP], [FN, TP]]:")
cm = confusion_matrix(y_test, pred, labels=[0, 1])
print(cm)

# ---------------- EXTRAÇÃO DE ERROS E ACERTOS ----------------

# Preserva o índice original do CSV para rastreabilidade.
resultados = teste.copy()
resultados["Label_real"] = y_test.to_numpy()
resultados["Previsao_XGBoost"] = pred
resultados["Probabilidade_phishing"] = prob
resultados["Resultado"] = np.select(
    [
        (resultados["Label_real"] == 1) &
        (resultados["Previsao_XGBoost"] == 0),
        (resultados["Label_real"] == 0) &
        (resultados["Previsao_XGBoost"] == 1),
        resultados["Label_real"] == resultados["Previsao_XGBoost"],
    ],
    ["Falso negativo", "Falso positivo", "Acerto"],
    default="Verificar"
)

falsos_negativos = resultados[
    (resultados["Label_real"] == 1) &
    (resultados["Previsao_XGBoost"] == 0)
].copy()

phishing_acertos = resultados[
    (resultados["Label_real"] == 1) &
    (resultados["Previsao_XGBoost"] == 1)
].copy()

falsos_positivos = resultados[
    (resultados["Label_real"] == 0) &
    (resultados["Previsao_XGBoost"] == 1)
].copy()

print("\n" + "=" * 70)
print("3. CONTAGEM DOS RESULTADOS")
print("=" * 70)
print("Total de mensagens Gemma:", len(resultados))
print("Phishing reais:", int((y_test == 1).sum()))
print("Falsos negativos:", len(falsos_negativos))
print("Phishing corretamente identificados:", len(phishing_acertos))
print("Falsos positivos:", len(falsos_positivos))
print("Taxa de falsos negativos entre os phishing:",
      len(falsos_negativos) / max(1, int((y_test == 1).sum())))

salvar_csv(resultados, "gemma_todas_previsoes.csv")
salvar_csv(falsos_negativos, "gemma_falsos_negativos.csv")
salvar_csv(phishing_acertos, "gemma_phishing_acertos.csv")
salvar_csv(falsos_positivos, "gemma_falsos_positivos.csv")

# ---------------- MÉDIAS POR GERADOR ----------------

print("\n" + "=" * 70)
print("4. COMPARAÇÃO DOS ATRIBUTOS ENTRE GERADORES")
print("=" * 70)

phishing = df[df["Label"].astype(int) == 1].copy()

# Todas as mensagens de phishing da Gemma versus todos os demais.
phishing_gemma = phishing[
    phishing["Modelo_Gerador"] == GEMMA
].copy()

phishing_outros = phishing[
    phishing["Modelo_Gerador"] != GEMMA
].copy()

stats_gemma_todos = resumo_numerico(phishing_gemma, features)
stats_outros_todos = resumo_numerico(phishing_outros, features)

comparacao_todos = stats_gemma_todos.merge(
    stats_outros_todos,
    on="Atributo",
    suffixes=("_Gemma", "_Outros")
)
comparacao_todos["Diferenca_media_Gemma_menos_Outros"] = (
    comparacao_todos["media_Gemma"] -
    comparacao_todos["media_Outros"]
)

# Falsos negativos da Gemma versus phishing da Gemma classificados corretamente.
stats_fn = resumo_numerico(falsos_negativos, features)
stats_acertos = resumo_numerico(phishing_acertos, features)

comparacao_erros = stats_fn.merge(
    stats_acertos,
    on="Atributo",
    suffixes=("_FalsosNegativos", "_Acertos")
)
comparacao_erros["Diferenca_media_FN_menos_Acertos"] = (
    comparacao_erros["media_FalsosNegativos"] -
    comparacao_erros["media_Acertos"]
)

salvar_csv(comparacao_todos, "comparacao_gemma_vs_outros_phishing.csv")
salvar_csv(comparacao_erros, "comparacao_gemma_FN_vs_acertos.csv")

# Tabela compacta para os atributos prioritários.
prioritarios = features.copy()

tabela_principal = comparacao_todos[
    comparacao_todos["Atributo"].isin(prioritarios)
][[
    "Atributo", "media_Gemma", "desvio_padrao_Gemma",
    "media_Outros", "desvio_padrao_Outros",
    "Diferenca_media_Gemma_menos_Outros"
]].merge(
    comparacao_erros[
        comparacao_erros["Atributo"].isin(prioritarios)
    ][[
        "Atributo", "media_FalsosNegativos",
        "desvio_padrao_FalsosNegativos",
        "media_Acertos", "desvio_padrao_Acertos",
        "Diferenca_media_FN_menos_Acertos"
    ]],
    on="Atributo",
    how="left"
)

print("\nTabela de atributos prioritários:")
print(tabela_principal.round(4).to_string(index=False))
salvar_csv(tabela_principal, "tabela_principal_atributos_gemma.csv")

# ---------------- COMPARAÇÃO INDIVIDUAL DOS GERADORES ----------------

stats_por_gerador = (
    phishing.groupby("Modelo_Gerador")[features]
    .agg(["mean", "std", "median"])
)
stats_por_gerador.columns = [
    f"{atributo}_{estatistica}"
    for atributo, estatistica in stats_por_gerador.columns
]
stats_por_gerador = stats_por_gerador.reset_index()
salvar_csv(stats_por_gerador, "estatisticas_phishing_por_gerador.csv")

# ---------------- AMOSTRAS PARA ANÁLISE QUALITATIVA ----------------

# Colunas textuais comuns. Se o CSV não contiver alguma delas, ela será omitida.
candidatas_texto = [
    "Texto", "texto", "Email", "email", "Mensagem", "mensagem",
    "Conteudo", "Conteúdo", "conteudo", "Body", "body",
    "Assunto", "assunto", "Subject", "subject",
    "Tema", "Extensao_Classe", "Modelo_Gerador", "Label",
    "Label_real", "Previsao_XGBoost", "Probabilidade_phishing",
    "Resultado"
]

colunas_textuais = [
    c for c in candidatas_texto if c in resultados.columns
]

# As features também são mantidas para permitir auditoria dos casos.
colunas_exportacao = list(dict.fromkeys(
    colunas_textuais + features
))

# Extrai uma amostra aleatória reprodutível de falsos negativos e acertos.
# Exporta todos os casos quando houver menos que o limite solicitado.
N_AMOSTRAS_POR_GRUPO = 30

amostra_fn = falsos_negativos.sample(
    n=min(N_AMOSTRAS_POR_GRUPO, len(falsos_negativos)),
    random_state=SEED
) if len(falsos_negativos) else falsos_negativos.copy()

amostra_acertos = phishing_acertos.sample(
    n=min(N_AMOSTRAS_POR_GRUPO, len(phishing_acertos)),
    random_state=SEED
) if len(phishing_acertos) else phishing_acertos.copy()

salvar_csv(
    amostra_fn[colunas_exportacao],
    "amostra_qualitativa_falsos_negativos_gemma.csv"
)
salvar_csv(
    amostra_acertos[colunas_exportacao],
    "amostra_qualitativa_phishing_acertos_gemma.csv"
)

# ---------------- AMOSTRAS DOS OUTROS GERADORES ----------------

amostras_outros = []

for gerador, grupo in phishing_outros.groupby("Modelo_Gerador"):
    amostra = grupo.sample(
        n=min(N_AMOSTRAS_POR_GRUPO, len(grupo)),
        random_state=SEED
    ).copy()

    # Adiciona colunas de classificação para manter o mesmo formato
    # dos arquivos exportados da Gemma.
    amostra["Label_real"] = amostra["Label"].astype(int)
    amostra["Previsao_XGBoost"] = np.nan
    amostra["Probabilidade_phishing"] = np.nan
    amostra["Resultado"] = "Não avaliado nesta execução"

    amostras_outros.append(amostra)

if amostras_outros:
    amostras_outros_df = pd.concat(
        amostras_outros,
        ignore_index=True
    )

    # Exporta apenas colunas que realmente existem no DataFrame.
    colunas_exportacao_validas = [
        c for c in colunas_exportacao
        if c in amostras_outros_df.columns
    ]

    salvar_csv(
        amostras_outros_df[colunas_exportacao_validas],
        "amostra_qualitativa_phishing_outros_geradores.csv"
    )
# ---------------- IMPORTÂNCIA DO MODELO ----------------

importancia = pd.DataFrame({
    "Atributo": features,
    "Importancia_gain": modelo.feature_importances_
}).sort_values("Importancia_gain", ascending=False)

salvar_csv(importancia, "importancia_features_modelo_gemma_lolo.csv")

print("\n10 principais features segundo feature_importances_:")
print(importancia.head(10).round(6).to_string(index=False))

# ---------------- SALVAR MODELO E METADADOS ----------------

caminho_modelo = os.path.join(
    PASTA_SAIDA, "modelo_xgb_cego_para_gemma_reproduzido.json"
)
modelo.save_model(caminho_modelo)

metadados = {
    "arquivo_features": CAMINHO_FEATURES,
    "sha256_dataset": sha256_dataset,
    "gerador_teste": str(GEMMA),
    "seed": SEED,
    "n_splits_cv": N_SPLITS,
    "param_grid": PARAM_GRID,
    "melhores_parametros": grid.best_params_,
    "melhor_f1_cv": float(grid.best_score_),
    "n_treino": int(len(treino)),
    "n_teste": int(len(teste)),
    "n_phishing_gemma": int((y_test == 1).sum()),
    "n_falsos_negativos": int(len(falsos_negativos)),
    "n_phishing_acertos": int(len(phishing_acertos)),
    "acuracia": float(accuracy_score(y_test, pred)),
    "precisao": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
    "matriz_confusao_labels_0_1": cm.tolist(),
    "xgboost_version": xgb.__version__,
    "pandas_version": pd.__version__,
    "numpy_version": np.__version__,
}

caminho_metadados = os.path.join(PASTA_SAIDA, "metadados_execucao.json")
with open(caminho_metadados, "w", encoding="utf-8") as f:
    json.dump(metadados, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 70)
print("5. RESUMO FINAL")
print("=" * 70)
print(json.dumps(metadados, ensure_ascii=False, indent=2))
print("\nArquivos gerados em:", PASTA_SAIDA)
print("Concluído.")
