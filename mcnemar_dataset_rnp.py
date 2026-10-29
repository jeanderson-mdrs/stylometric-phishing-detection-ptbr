import pandas as pd
import numpy as np
import xgboost as xgb
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from statsmodels.stats.contingency_tables import mcnemar


caminho_features = "/content/drive/MyDrive/Colab Notebooks/dataset_organico_features_rnp.csv"
caminho_texto = "/content/drive/MyDrive/Colab Notebooks/dataset_organico_phishing_rnp.csv"
caminho_modelo_xgb = "/content/drive/MyDrive/Colab Notebooks/modelo_xgboost_phishing.json"
caminho_modelo_bert = "/content/drive/MyDrive/Colab Notebooks/modelo_bertimbau_phishing"

print("Carregando dados e modelos para o Teste de Significância...")

# Carregar Dados
df_feat = pd.read_csv(caminho_features).dropna(how='all')
df_text = pd.read_csv(caminho_texto).dropna(how='all')

# Inferência XGBoost
colunas_ignorar = ['Label', 'Message_ID']
X_org = df_feat.drop(columns=[col for col in colunas_ignorar if col in df_feat.columns])
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(caminho_modelo_xgb)
previsoes_xgb = xgb_model.predict(X_org)

# Inferência BERTimbau
col_assunto = 'Assunto_Anonimizado' if 'Assunto_Anonimizado' in df_text.columns else 'Assunto'
col_texto = 'Texto_Limpo_Anonimizado' if 'Texto_Limpo_Anonimizado' in df_text.columns else 'Texto_Limpo'
df_text[col_assunto] = df_text[col_assunto].fillna('')
df_text[col_texto] = df_text[col_texto].fillna('')
textos_org = (df_text[col_assunto] + " - " + df_text[col_texto]).tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(caminho_modelo_bert)
model = AutoModelForSequenceClassification.from_pretrained(caminho_modelo_bert)
model.to(device)
model.eval()

previsoes_bert = []
for texto in textos_org:
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        previsoes_bert.append(pred)

# Construção da Tabela de Contingência de McNemar
# Como y_true é sempre 1 (todos são phishing):
# Acerto = prever 1, Erro = prever 0
ambos_acertaram = sum((previsoes_xgb == 1) & (np.array(previsoes_bert) == 1))
so_xgb_acertou = sum((previsoes_xgb == 1) & (np.array(previsoes_bert) == 0))
so_bert_acertou = sum((previsoes_xgb == 0) & (np.array(previsoes_bert) == 1))
ambos_erraram = sum((previsoes_xgb == 0) & (np.array(previsoes_bert) == 0))

tabela_contingencia = [[ambos_acertaram, so_xgb_acertou],
                       [so_bert_acertou, ambos_erraram]]


resultado_mcnemar = mcnemar(tabela_contingencia, exact=True)


print(" RESULTADO DO TESTE DE McNEMAR (SIGNIFICÂNCIA ESTATÍSTICA)")

print(f"Ambos Acertaram: {ambos_acertaram}")
print(f"Ambos Erraram: {ambos_erraram}")
print(f"Apenas XGBoost Acertou: {so_xgb_acertou}")
print(f"Apenas BERTimbau Acertou: {so_bert_acertou}")

print(f"Estatística do Teste: {resultado_mcnemar.statistic}")
print(f"Valor-p (p-value): {resultado_mcnemar.pvalue:.6f}")


if resultado_mcnemar.pvalue < 0.05:
    print("\n CONCLUSÃO: Há diferença estatisticamente significativa entre os modelos (p < 0.05).")
else:
    print("\n CONCLUSÃO: Não há diferença estatisticamente significativa (p >= 0.05).")