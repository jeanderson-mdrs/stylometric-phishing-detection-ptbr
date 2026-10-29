import re
import html
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path
from datasketch import MinHash, MinHashLSH
from collections import defaultdict
from IPython.display import display



BASE = Path("/content/drive/MyDrive/Colab Notebooks/")

ARQ_DADOS = BASE / "dataset_completo.csv"
ARQ_TREINO = Path("/content/drive/MyDrive/Colab Notebooks/ids_train.csv")
ARQ_VAL = Path("/content/drive/MyDrive/Colab Notebooks/ids_val.csv")
ARQ_TESTE = Path("/content/drive/MyDrive/Colab Notebooks/ids_test.csv")

ID_COL = "ID_original"
K_SHINGLE = 5
NUM_PERM = 128
LIMIAR_FINAL = 0.80
LIMIAR_LSH = 0.70


# CARREGAR DATASET E PRESERVAR A POSIÇÃO ORIGINAL


df = pd.read_csv(ARQ_DADOS)

colunas_necessarias = {
    "Assunto", "Texto_Limpo", "Modelo_Gerador",
    "Extensao_Classe", "Label"
}
faltantes = colunas_necessarias - set(df.columns)

if faltantes:
    raise ValueError(f"Colunas ausentes no dataset: {faltantes}")


df.insert(0, ID_COL, df.index)

print("Total de amostras:", len(df))
print("Intervalo de IDs:", df[ID_COL].min(), "-", df[ID_COL].max())


# CARREGAR OS IDS DOS PARTICIONAMENTOS


def carregar_ids(caminho):
    ids_df = pd.read_csv(caminho)

    if ID_COL not in ids_df.columns:
        raise ValueError(
            f"Coluna {ID_COL} não encontrada em {caminho}"
        )

    ids = pd.to_numeric(ids_df[ID_COL], errors="raise").astype(int)

    if ids.duplicated().any():
        raise ValueError(f"IDs duplicados em {caminho}")

    return set(ids.tolist())

ids_train = carregar_ids(ARQ_TREINO)
ids_val = carregar_ids(ARQ_VAL)
ids_test = carregar_ids(ARQ_TESTE)

# Validar limites e sobreposição.
ids_dataset = set(df[ID_COL])

for nome, ids in [
    ("treino", ids_train),
    ("validação", ids_val),
    ("teste", ids_test)
]:
    fora = ids - ids_dataset
    if fora:
        raise ValueError(
            f"{len(fora)} IDs de {nome} não existem no dataset. "
            f"Exemplos: {list(fora)[:10]}"
        )

print("\nTamanhos dos conjuntos:")
print("Treino:", len(ids_train))
print("Validação:", len(ids_val))
print("Teste:", len(ids_test))

print("\nInterseções de IDs:")
print("Treino ∩ validação:", len(ids_train & ids_val))
print("Treino ∩ teste:", len(ids_train & ids_test))
print("Validação ∩ teste:", len(ids_val & ids_test))

if (ids_train & ids_val) or (ids_train & ids_test) or (ids_val & ids_test):
    raise ValueError("Há IDs sobrepostos entre os particionamentos.")

if len(ids_train | ids_val | ids_test) != len(df):
    print(
        "\nATENÇÃO: os particionamentos não cobrem todas as linhas "
        "do dataset, ou há IDs não utilizados."
    )

particao = {}
for i in ids_train:
    particao[i] = "treino"
for i in ids_val:
    particao[i] = "validacao"
for i in ids_test:
    particao[i] = "teste"

df["particao"] = df[ID_COL].map(particao)

# Manter apenas as amostras que pertencem aos três conjuntos.
df = df[df["particao"].notna()].copy()


# NORMALIZAÇÃO DO ASSUNTO E DO CORPO


def normalizar(texto):
    texto = html.unescape(str(texto))
    texto = unicodedata.normalize("NFKC", texto).lower()

    # Normalizar URLs e e-mails variáveis.
    texto = re.sub(r"https?://\S+|www\.\S+", " URL ", texto)
    texto = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.\w+\b",
        " EMAIL ",
        texto
    )

    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

df["texto_completo"] = (
    "assunto: " + df["Assunto"].fillna("").astype(str)
    + " corpo: " + df["Texto_Limpo"].fillna("").astype(str)
).map(normalizar)

df = df[df["texto_completo"].str.len() > 0].copy()

# DUPLICATAS EXATAS ENTRE PARTIÇÕES

duplicatas_exatas = []

for texto, grupo in df.groupby("texto_completo", sort=False):
    particoes = set(grupo["particao"])

    if len(grupo) > 1 and len(particoes) > 1:
        registros = grupo.to_dict("records")

        for i in range(len(registros)):
            for j in range(i + 1, len(registros)):
                a, b = registros[i], registros[j]

                if a["particao"] != b["particao"]:
                    duplicatas_exatas.append({
                        "id_a": a[ID_COL],
                        "particao_a": a["particao"],
                        "id_b": b[ID_COL],
                        "particao_b": b["particao"],
                        "label_a": a["Label"],
                        "label_b": b["Label"],
                        "texto_a": a["texto_completo"],
                        "texto_b": b["texto_completo"]
                    })

print("\nPares de duplicatas exatas entre partições:",
      len(duplicatas_exatas))


# SHINGLES E MINHASH


def shingles_palavras(texto, k=K_SHINGLE):
    palavras = texto.split()

    if not palavras:
        return set()

    if len(palavras) < k:
        return {" ".join(palavras)}

    return {
        " ".join(palavras[i:i+k])
        for i in range(len(palavras) - k + 1)
    }

def criar_minhash(shingles):
    mh = MinHash(num_perm=NUM_PERM)
    for shingle in shingles:
        mh.update(shingle.encode("utf-8"))
    return mh

registros = []

for _, linha in df.iterrows():
    sh = shingles_palavras(linha["texto_completo"])

    registros.append({
        "id": int(linha[ID_COL]),
        "particao": linha["particao"],
        "label": linha["Label"],
        "modelo": linha["Modelo_Gerador"],
        "texto": linha["texto_completo"],
        "shingles": sh,
        "minhash": criar_minhash(sh)
    })


# LSH: BUSCA DE PARES CANDIDATOS


lsh = MinHashLSH(
    threshold=LIMIAR_LSH,
    num_perm=NUM_PERM
)

por_id = {}

for r in registros:
    chave = str(r["id"])
    por_id[chave] = r
    lsh.insert(chave, r["minhash"])

pares = set()

# Comparar teste contra treino e validação.
for r in registros:
    if r["particao"] != "teste":
        continue

    for chave in lsh.query(r["minhash"]):
        outro = por_id[chave]

        if outro["particao"] in {"treino", "validacao"}:
            pares.add(tuple(sorted([str(r["id"]), chave])))



resultados = []

for chave_a, chave_b in pares:
    a = por_id[chave_a]
    b = por_id[chave_b]

    if a["particao"] == b["particao"]:
        continue

    inter = len(a["shingles"] & b["shingles"])
    uniao = len(a["shingles"] | b["shingles"])
    jaccard = inter / uniao if uniao else 1.0

    if jaccard < LIMIAR_FINAL:
        continue

    if {a["particao"], b["particao"]} == {"treino", "teste"}:
        comparacao = "treino-teste"
    elif {a["particao"], b["particao"]} == {"validacao", "teste"}:
        comparacao = "validacao-teste"
    else:
        continue

    resultados.append({
        "comparacao": comparacao,
        "id_a": a["id"],
        "particao_a": a["particao"],
        "id_b": b["id"],
        "particao_b": b["particao"],
        "jaccard": round(jaccard, 4),
        "label_a": a["label"],
        "label_b": b["label"],
        "modelo_a": a["modelo"],
        "modelo_b": b["modelo"],
        "texto_a": a["texto"],
        "texto_b": b["texto"]
    })

colunas = [
    "comparacao", "id_a", "particao_a", "id_b", "particao_b",
    "jaccard", "label_a", "label_b", "modelo_a", "modelo_b",
    "texto_a", "texto_b"
]

resultado_df = pd.DataFrame(resultados, columns=colunas)

if not resultado_df.empty:
    resultado_df = resultado_df.sort_values(
        ["comparacao", "jaccard"],
        ascending=[True, False]
    )

print("\nPares suspeitos com Jaccard >=", LIMIAR_FINAL)

if resultado_df.empty:
    print("Nenhum par candidato acima do limiar.")
else:
    print(resultado_df.groupby("comparacao").size())
    display(resultado_df.head(30))


# EXPORTAR RESULTADOS


arq_pares = BASE / "auditoria_quase_duplicatas_minhash.csv"
arq_exatas = BASE / "auditoria_duplicatas_exatas.csv"

resultado_df.to_csv(
    arq_pares, index=False, encoding="utf-8-sig"
)
pd.DataFrame(duplicatas_exatas).to_csv(
    arq_exatas, index=False, encoding="utf-8-sig"
)

print("\nRelatórios salvos:")
print(arq_pares)
print(arq_exatas)
