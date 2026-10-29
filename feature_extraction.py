import pandas as pd
import spacy
import textstat
import numpy as np
import os
from nltk import ngrams
from collections import Counter

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


caminho_in = "/content/drive/MyDrive/Colab Notebooks/dataset_completo.csv" 
caminho_out = "/content/drive/MyDrive/Colab Notebooks/dataset_completo_features.csv"

if os.path.exists(caminho_in):
    df = pd.read_csv(caminho_in)
    print(f"Iniciando extração para {len(df)} registros. Isso pode levar alguns minutos...")

    df['Assunto'] = df['Assunto'].fillna('')
    df['Texto_Limpo'] = df['Texto_Limpo'].fillna('') 


    features_list = df.apply(lambda row: extrair_60_features_literal(row['Texto_Limpo'], row['Assunto']), axis=1)
    df_features = pd.DataFrame(list(features_list))

    colunas_preservadas = ['Modelo_Gerador', 'Extensao_Classe', 'Label']

  
    if 'Tema' in df.columns:
        colunas_preservadas.append('Tema')

    df_final = pd.concat([df[colunas_preservadas], df_features], axis=1)


    df_final.to_csv(caminho_out, index=False)
    print(f"✅ Sucesso! Dataset salvo em: {caminho_out}")
    print(f"Estrutura Final: {df_final.shape[1]} colunas preservadas prontas para o XGBoost.")
else:
    print(f"Erro: Arquivo '{caminho_in}' não encontrado. Verifique o caminho.")