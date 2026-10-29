import os
import time
import json
import random
import platform
import statistics
import numpy as np
import pandas as pd
import torch
import spacy
import textstat

from nltk import ngrams
from xgboost import XGBClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter


CAMINHO_DATASET = (
    "/content/drive/MyDrive/Colab Notebooks/"
    "dataset_completo.csv"
)
CAMINHO_IDS_TESTE = (
    "/content/drive/MyDrive/Colab Notebooks/ids_test.csv"
)
CAMINHO_XGBOOST = (
    "/content/drive/MyDrive/Colab Notebooks/"
    "modelo_xgboost_phishing.json"
)
CAMINHO_BERTIMBAU = (
    "/content/drive/MyDrive/Colab Notebooks/"
    "modelo_bertimbau_phishing"
)
CAMINHO_SAIDA = (
    "/content/drive/MyDrive/Colab Notebooks/"
    "benchmark_latencia"
)
os.makedirs(CAMINHO_SAIDA, exist_ok=True)

SEED = 42
MAX_LENGTH = 512
N_WARMUP = 10
N_REPETICOES = 3  # mede cada e-mail 3 vezes; 
BATCH_SIZE_BERT = 1  # batch 1 para medir latência por e-mail

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# VERIFICAÇÃO DO AMBIENTE E DOS CAMINHOS

try:
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
except ImportError:
    pass

for nome, caminho in {
    "Dataset": CAMINHO_DATASET,
    "IDs de teste": CAMINHO_IDS_TESTE,
    "Modelo XGBoost": CAMINHO_XGBOOST,
    "Modelo BERTimbau": CAMINHO_BERTIMBAU,
}.items():
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"{nome} não encontrado: {caminho}\n"
            "Corrija somente o caminho correspondente na seção 1."
        )

print("=== AMBIENTE ===")
print("Python:", platform.python_version())
print("Sistema:", platform.platform())
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("spaCy:", spacy.__version__)
print("textstat:", getattr(textstat, "__version__", "versão não exposta"))
print("PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Dispositivo de inferência BERT: CPU")


# CARREGAR E VALIDAR OS 874 IDS

df = pd.read_csv(CAMINHO_DATASET, dtype={"ID_original": str})
ids_df = pd.read_csv(CAMINHO_IDS_TESTE, dtype={"ID_original": str})

if "ID_original" not in df.columns:
    print(
        "AVISO: dataset sem ID_original. Recriando pelo índice; "
        "isso pressupõe que o arquivo manteve exatamente a ordem original."
    )
    df["ID_original"] = df.index.astype(str)

if "ID_original" not in ids_df.columns:
    # Compatibilidade com CSV contendo uma única coluna com outro cabeçalho.
    if ids_df.shape[1] == 1:
        ids_df.columns = ["ID_original"]
        ids_df["ID_original"] = ids_df["ID_original"].astype(str)
    else:
        raise ValueError("O arquivo de IDs precisa conter a coluna ID_original.")

df["ID_original"] = df["ID_original"].astype(str).str.strip()
ids_df["ID_original"] = ids_df["ID_original"].astype(str).str.strip()
ids = ids_df["ID_original"].tolist()

if len(ids) != 874:
    print(f"AVISO: esperado 874 IDs; encontrado(s) {len(ids)}.")
if len(set(ids)) != len(ids):
    raise ValueError("O arquivo ids_test.csv contém IDs duplicados.")
if df["ID_original"].duplicated().any():
    raise ValueError("O dataset contém ID_original duplicado.")

mapa = df.set_index("ID_original", drop=False)
faltantes = [x for x in ids if x not in mapa.index]
if faltantes:
    raise ValueError(
        f"{len(faltantes)} IDs não foram encontrados. Exemplos: {faltantes[:10]}. "
        "Verifique se o dataset é a mesma versão/ordem usada para gerar os IDs."
    )

# Preserva a ordem de ids_test.csv
df_test = mapa.loc[ids].reset_index(drop=True)

for col in ["Assunto", "Texto_Limpo"]:
    if col not in df_test.columns:
        raise KeyError(f"Coluna obrigatória ausente no dataset: {col}")
    df_test[col] = df_test[col].fillna("").astype(str)

if "Label" not in df_test.columns:
    raise KeyError("A coluna Label não foi encontrada no dataset.")

df_test["Label"] = pd.to_numeric(df_test["Label"], errors="raise").astype(int)
df_test["Texto_Completo"] = (
    df_test["Assunto"] + " - " + df_test["Texto_Limpo"]
)

print("\n=== VALIDAÇÃO DO CONJUNTO DE TESTE ===")
print("IDs:", len(ids))
print("Registros recuperados:", len(df_test))
print("IDs ausentes:", len(faltantes))
print("Classes:\n", df_test["Label"].value_counts().sort_index())
print("Duplicatas de ID no teste:", df_test["ID_original"].duplicated().sum())


# CARREGAR spaCy E FUNÇÃO ORIGINAL DE EXTRAÇÃO DAS 60 FEATURES

print("\nCarregando spaCy pt_core_news_lg...")
nlp = spacy.load("pt_core_news_lg")

def extrair_f60_personalizacao(doc):
    vocativos = ["prezado", "caro", "olá", "senhor", "senhora", "cliente", "usuário", "contribuinte"]
    pronomes_diretos = ["você", "seu", "sua", "vocês"]

    count = 0
    for t in doc:
        if t.lemma_.lower() in vocativos:
            count += 1
        if t.pos_ == "PRON" and t.lemma_.lower() in pronomes_diretos:
            count += 1
    return count

def extrair_60_features_literal(corpo, assunto):
    texto = f"{assunto} {corpo}".strip()
    if not texto: return None

    doc = nlp(corpo)
    sentences = list(doc.sents)
    words = [t for t in doc if not t.is_punct and not t.is_space]
    f = {}

    # --- 1. RECURSOS LEXICAIS (F1 - F12) ---
    f['F1_word_count'] = len(words)
    f['F2_char_count'] = len(texto)
    f['F3_avg_word_len'] = np.mean([len(t.text) for t in words]) if words else 0
    f['F4_sentence_count'] = len(sentences)
    f['F5_avg_sentence_len'] = len(words) / len(sentences) if sentences else 0
    f['F6_unique_word_count'] = len(set(t.lemma_.lower() for t in words))
    f['F7_lexical_diversity'] = f['F6_unique_word_count'] / f['F1_word_count'] if f['F1_word_count'] > 0 else 0
    f['F8_email_symbol_count'] = texto.count('@')
    f['F9_upper_word_count'] = sum(1 for t in words if t.text.isupper() and len(t.text) > 1)
    f['F10_upper_word_ratio'] = f['F9_upper_word_count'] / f['F1_word_count'] if f['F1_word_count'] > 0 else 0
    f['F11_complex_word_count'] = sum(1 for t in words if len(t.text) > 6)
    f['F12_avg_syllables_per_word'] = textstat.avg_syllables_per_word(corpo)

    # --- 2. RECURSOS SINTÁTICOS (F13 - F20) ---
    f['F13_comma_count'] = corpo.count(',')
    f['F14_semicolon_count'] = corpo.count(';')
    f['F15_colon_count'] = corpo.count(':')
    f['F16_complexity_ratio'] = sum(1 for t in doc if t.pos_ in ("CCONJ", "SCONJ")) / len(sentences) if sentences else 0
    f['F17_clause_density'] = sum(1 for t in doc if t.dep_ in ("ccomp", "xcomp", "advcl")) / len(sentences) if sentences else 0
    f['F18_pronoun_density'] = sum(1 for t in doc if t.pos_ == "PRON") / f['F1_word_count'] if f['F1_word_count'] > 0 else 0
    f['F19_preposition_density'] = sum(1 for t in doc if t.pos_ == "ADP") / f['F1_word_count'] if f['F1_word_count'] > 0 else 0
    f['F20_function_word_density'] = sum(1 for t in doc if t.pos_ in ("DET", "ADP", "PRON", "SCONJ")) / f['F1_word_count'] if f['F1_word_count'] > 0 else 0

    # --- 3. RECURSOS DE PONTUAÇÃO (F21 - F30) ---
    simbolos = ['.', ',', '!', ':', '-', '"', '(', ')', '/']
    for i, s in enumerate(simbolos, 21):
        f[f'F{i}_freq_{s}'] = corpo.count(s) / len(corpo) if len(corpo) > 0 else 0
    f['F30_punctuation_variety'] = len(set(c for c in corpo if c in simbolos))

    # --- 4. LEGIBILIDADE (F31 - F40) ---
    asl = f['F5_avg_sentence_len']
    asw = f['F12_avg_syllables_per_word']
    f['F31_flesch_reading_ease_PT'] = 248.835 - (1.015 * asl) - (84.6 * asw)
    f['F32_smog_index'] = textstat.smog_index(corpo)
    f['F33_dale_chall_score'] = textstat.dale_chall_readability_score(corpo)
    f['F34_coleman_liau_index'] = textstat.coleman_liau_index(corpo)
    f['F35_gunning_fog'] = textstat.gunning_fog(corpo)

    for i, tag in enumerate(['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON'], 36):
        f[f'F{i}_{tag.lower()}_ratio'] = sum(1 for t in doc if t.pos_ == tag) / len(doc) if len(doc) > 0 else 0

    # --- 5. PSICOLINGUÍSTICOS E PRAGMÁTICOS (F41 - F45) ---
    verbos_comando_phishing = ["clicar", "baixar", "acessar", "abrir", "conferir", "pagar", "atualizar", "verificar", "fazer"]
    count_f41 = 0
    for t in doc:
        if t.pos_ in ["VERB", "AUX"]:
            morph = str(t.morph)
            if "Mood=Imp" in morph:
                count_f41 += 1
            elif "Mood=Sub" in morph:
                if t.lemma_.lower() in verbos_comando_phishing or not any(child.dep_ == "nsubj" for child in t.children):
                    count_f41 += 1
    f['F41_imperative_verbs_count'] = count_f41
    f['F42_modal_verbs_count'] = sum(1 for t in doc if t.lemma_.lower() in ["poder", "dever", "precisar", "ter"])
    f['F43_uncertainty_adverbs_count'] = sum(1 for t in doc if t.text.lower() in ["talvez", "possivelmente"])
    f['F44_pron_1st_pers_count'] = sum(1 for t in doc if t.morph.get("Person") == ["1"])
    f['F45_pron_2nd_3rd_pers_count'] = sum(1 for t in doc if t.morph.get("Person") in [["2"], ["3"]])

    # --- 6. ESPECÍFICOS DE E-MAIL (F46 - F50) ---
    f['F46_attachment_mentions'] = sum(1 for t in doc if t.lemma_.lower() in ["anexo", "arquivo", "fatura"])
    f['F47_technical_jargon_count'] = sum(1 for t in doc if t.lemma_.lower() in ["segurança", "autenticação", "senha"])
    f['F48_promotional_words_count'] = sum(1 for t in doc if t.lemma_.lower() in ["oferta", "acordo", "promoção"])
    f['F49_subject_word_count'] = len(assunto.split())
    f['F50_subject_char_count'] = len(assunto)

    # --- 7. COMPLEXIDADE (F51 - F55) ---
    f['F51_bigram_count'] = len(list(ngrams([t.text for t in words], 2)))
    f['F52_trigram_count'] = len(list(ngrams([t.text for t in words], 3)))
    f['F53_word_length_variation'] = np.std([len(t.text) for t in words]) if words else 0
    f['F54_syllable_count_total'] = textstat.syllable_count(corpo)
    f['F55_polysyllable_count'] = textstat.polysyllabcount(corpo)

    # --- 8. ESTILÍSTICOS (F56 - F60) ---
    f['F56_politeness_markers'] = sum(1 for t in doc if t.lemma_.lower() in ["obrigado", "gentileza"])
    f['F57_aggressiveness_markers'] = sum(1 for t in doc if t.lemma_.lower() in ["obrigatório", "suspensão"])
    f['F58_urgency_markers_count'] = sum(1 for t in doc if t.lemma_.lower() in ["urgente", "imediatamente", "agora"])
    f['F59_conditional_phrases_count'] = sum(1 for t in doc if t.text.lower() in ["se", "caso"])
    f['F60_personalisation_markers_count'] = extrair_f60_personalizacao(doc)

    return f


# CARREGAR MODELOS TREINADOS (SEM MEDIR O TEMPO DE CARREGAMENTO)

print("\nCarregando XGBoost...")
modelo_xgb = XGBClassifier()
modelo_xgb.load_model(CAMINHO_XGBOOST)

print("Carregando tokenizer e BERTimbau fine-tuned...")
tokenizer = AutoTokenizer.from_pretrained(CAMINHO_BERTIMBAU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo_bert = AutoModelForSequenceClassification.from_pretrained(CAMINHO_BERTIMBAU)
modelo_bert.to(device)
modelo_bert.eval()


# EXTRAIR FEATURES DOS 874 EMAILS E MEDIR O TEMPO POR EMAIL

print("\nExtraindo features dos e-mails de teste...")
nomes_features = None
features_linhas = []
tempos_extracao = []
erros_extracao = []

for i, row in df_test.iterrows():
    inicio = time.perf_counter()
    try:
        resultado = extrair_60_features_literal(
            row["Texto_Limpo"], row["Assunto"]
        )
        if resultado is None:
            raise ValueError("E-mail sem conteúdo.")
        features_linhas.append(resultado)
        tempos_extracao.append(time.perf_counter() - inicio)
    except Exception as e:
        erros_extracao.append((i, str(row["ID_original"]), repr(e)))
        features_linhas.append(None)
        tempos_extracao.append(time.perf_counter() - inicio)

if erros_extracao:
    print("\nErros na extração (primeiros 10):")
    for erro in erros_extracao[:10]:
        print(erro)
    raise RuntimeError(
        f"Falha na extração de {len(erros_extracao)} e-mail(s). "
        "Não continue até resolver os erros."
    )

X_features = pd.DataFrame(features_linhas)
if X_features.shape[1] != 60:
    raise ValueError(f"Esperadas 60 features; obtidas {X_features.shape[1]}.")

nomes_modelo = getattr(modelo_xgb, "feature_names_in_", None)
if nomes_modelo is None:
    nomes_modelo = getattr(modelo_xgb, "feature_names", None)

if nomes_modelo is not None:
    nomes_modelo = list(nomes_modelo)
    ausentes = [c for c in nomes_modelo if c not in X_features.columns]
    extras = [c for c in X_features.columns if c not in nomes_modelo]
    if ausentes:
        raise ValueError(
            "O extrator não produziu colunas esperadas pelo XGBoost: "
            + ", ".join(ausentes)
        )
    if len(nomes_modelo) != 60:
        print(
            f"AVISO: modelo XGBoost indica {len(nomes_modelo)} features, "
            "não 60. Usando a ordem registrada no modelo."
        )
    X_features = X_features[nomes_modelo]
else:
    print(
        "AVISO: o modelo não expõe nomes das features. "
        "Será usada a ordem produzida pela função original. "
        "Confirme que coincide com a ordem do treinamento."
    )

if X_features.isna().any().any():
    raise ValueError("Há valores NaN nas features extraídas.")
if not np.isfinite(X_features.to_numpy(dtype=float)).all():
    raise ValueError("Há valores infinitos nas features extraídas.")


# AQUECIMENTO E LATÊNCIA DE CLASSIFICAÇÃO XGBOOST

print("\nMedindo inferência XGBoost...")
# Aquecimento fora da medição
for i in range(min(N_WARMUP, len(X_features))):
    _ = modelo_xgb.predict(X_features.iloc[[i]])

tempos_inferencia_xgb = []
predicoes_xgb = []

for i in range(len(X_features)):
    amostra = X_features.iloc[[i]]
    duracoes = []
    pred = None
    for _ in range(N_REPETICOES):
        inicio = time.perf_counter()
        pred = modelo_xgb.predict(amostra)
        duracoes.append(time.perf_counter() - inicio)
    tempos_inferencia_xgb.append(float(np.median(duracoes)))
    predicoes_xgb.append(int(pred[0]))


# TOKENIZAÇÃO BERTIMBAU E TEMPO DE INFERÊNCIA POR E-MAIL

print("\nTokenizando e medindo inferência BERTimbau...")
textos = df_test["Texto_Completo"].tolist()

# Tokenização individual medida separadamente
tempos_tokenizacao = []
entradas_tokenizadas = []

for texto in textos:
    inicio = time.perf_counter()
    tokens = tokenizer(
        texto,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    tempos_tokenizacao.append(time.perf_counter() - inicio)
    entradas_tokenizadas.append(tokens)

def sincronizar_dispositivo():
    if device.type == "cuda":
        torch.cuda.synchronize()

# Aquecimento fora da medição
with torch.inference_mode():
    for i in range(min(N_WARMUP, len(entradas_tokenizadas))):
        entradas = {
            k: v.to(device) for k, v in entradas_tokenizadas[i].items()
        }
        _ = modelo_bert(**entradas)
        sincronizar_dispositivo()

tempos_inferencia_bert = []
predicoes_bert = []

with torch.inference_mode():
    for tokens in entradas_tokenizadas:
        entradas = {k: v.to(device) for k, v in tokens.items()}
        duracoes = []
        logits = None
        for _ in range(N_REPETICOES):
            sincronizar_dispositivo()
            inicio = time.perf_counter()
            saida = modelo_bert(**entradas)
            sincronizar_dispositivo()
            duracoes.append(time.perf_counter() - inicio)
            logits = saida.logits
        tempos_inferencia_bert.append(float(np.median(duracoes)))
        predicoes_bert.append(int(torch.argmax(logits, dim=-1).item()))


# RESUMO ESTATÍSTICO

def resumo_tempos(nome, tempos_segundos):
    a = np.asarray(tempos_segundos, dtype=float)
    return {
        "Etapa": nome,
        "N_e-mails": len(a),
        "Média_ms": float(np.mean(a) * 1000),
        "Mediana_ms": float(np.median(a) * 1000),
        "P95_ms": float(np.percentile(a, 95) * 1000),
        "Desvio_padrão_ms": float(np.std(a, ddof=1) * 1000) if len(a) > 1 else 0.0,
        "Soma_tempos_s": float(np.sum(a)),
        "E-mails_por_segundo_estimado": (
            float(1 / np.mean(a)) if np.mean(a) > 0 else None
        ),
    }

# O total de XGBoost é soma da extração e inferência medida por amostra.
tempos_total_xgb = [
    a + b for a, b in zip(tempos_extracao, tempos_inferencia_xgb)
]
tempos_total_bert = [
    a + b for a, b in zip(tempos_tokenizacao, tempos_inferencia_bert)
]

resumos = [
    resumo_tempos("XGBoost - extração 60 features", tempos_extracao),
    resumo_tempos("XGBoost - inferência", tempos_inferencia_xgb),
    resumo_tempos("XGBoost - total por e-mail", tempos_total_xgb),
    resumo_tempos("BERTimbau - tokenização", tempos_tokenizacao),
    resumo_tempos("BERTimbau - inferência", tempos_inferencia_bert),
    resumo_tempos("BERTimbau - total por e-mail", tempos_total_bert),
]
df_resumo = pd.DataFrame(resumos)

# Predições e tempos por amostra
df_resultados = pd.DataFrame({
    "ID_original": df_test["ID_original"].astype(str),
    "Label_real": df_test["Label"].to_numpy(),
    "Predicao_XGBoost": predicoes_xgb,
    "Predicao_BERTimbau": predicoes_bert,
    "Extracao_XGBoost_ms": np.asarray(tempos_extracao) * 1000,
    "Inferencia_XGBoost_ms": np.asarray(tempos_inferencia_xgb) * 1000,
    "Total_XGBoost_ms": np.asarray(tempos_total_xgb) * 1000,
    "Tokenizacao_BERTimbau_ms": np.asarray(tempos_tokenizacao) * 1000,
    "Inferencia_BERTimbau_ms": np.asarray(tempos_inferencia_bert) * 1000,
    "Total_BERTimbau_ms": np.asarray(tempos_total_bert) * 1000,
})

# Métricas preditivas para confirmar que os modelos carregados produzem saída
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

y_true = df_test["Label"].to_numpy()
metricas = {
    "XGBoost": {
        "accuracy": float(accuracy_score(y_true, predicoes_xgb)),
        "f1_phishing": float(f1_score(y_true, predicoes_xgb, pos_label=1)),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_true, predicoes_xgb, labels=[0, 1]
        ).tolist(),
    },
    "BERTimbau": {
        "accuracy": float(accuracy_score(y_true, predicoes_bert)),
        "f1_phishing": float(f1_score(y_true, predicoes_bert, pos_label=1)),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_true, predicoes_bert, labels=[0, 1]
        ).tolist(),
    },
}

print("\n=== RESUMO DE LATÊNCIA ===")
print(df_resumo.round(4).to_string(index=False))

print("\n=== MÉTRICAS PREDITIVAS NO TESTE ===")
print(json.dumps(metricas, indent=2, ensure_ascii=False))



arquivo_resumo = os.path.join(CAMINHO_SAIDA, "resumo_latencia.csv")
arquivo_detalhado = os.path.join(CAMINHO_SAIDA, "latencia_por_email.csv")
arquivo_metricas = os.path.join(CAMINHO_SAIDA, "metricas_e_ambiente.json")
arquivo_features = os.path.join(CAMINHO_SAIDA, "features_extraidas_teste.csv")

df_resumo.to_csv(arquivo_resumo, index=False)
df_resultados.to_csv(arquivo_detalhado, index=False)
X_features.assign(ID_original=df_test["ID_original"].astype(str)).to_csv(
    arquivo_features, index=False
)

ambiente = {
    "seed": SEED,
    "n_emails": len(df_test),
    "distribuicao_classes": {
        str(k): int(v) for k, v in df_test["Label"].value_counts().sort_index().items()
    },
    "dataset": CAMINHO_DATASET,
    "ids_teste": CAMINHO_IDS_TESTE,
    "xgboost_model": CAMINHO_XGBOOST,
    "bertimbau_model": CAMINHO_BERTIMBAU,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "spacy": spacy.__version__,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device": str(device),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "max_length_bert": MAX_LENGTH,
    "warmup_emails": N_WARMUP,
    "repetitions_per_email": N_REPETICOES,
    "latency_method": (
        "Per-email latency; model loading excluded; warm-up excluded; "
        "median of repeated inference timings per email. "
        "BERT GPU synchronized before/after forward pass."
    ),
    "metrics": metricas,
}
with open(arquivo_metricas, "w", encoding="utf-8") as f:
    json.dump(ambiente, f, indent=2, ensure_ascii=False)

print("\n=== ARQUIVOS SALVOS ===")
print(arquivo_resumo)
print(arquivo_detalhado)
print(arquivo_features)
print(arquivo_metricas)
print("\nBenchmark concluído.")
