import pandas as pd
import xgboost as xgb


caminho_features = "/content/drive/MyDrive/Colab Notebooks/dataset_organico_features_rnp.csv"
# Ajuste para o arquivo que contém os textos (pode ser o anonimizado ou o bruto)
caminho_textos = "/content/drive/MyDrive/Colab Notebooks/dataset_organico_phishing_rnp.csv"
caminho_saida_erros = "/content/drive/MyDrive/Colab Notebooks/falsos_negativos_xgboost.csv"


df_feat = pd.read_csv(caminho_features).dropna(how='all')
df_text = pd.read_csv(caminho_textos).dropna(how='all')


colunas_ignorar = ['Label', 'Message_ID']
X_org = df_feat.drop(columns=[col for col in colunas_ignorar if col in df_feat.columns])

# Inferência
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("/content/drive/MyDrive/Colab Notebooks/modelo_xgboost_phishing.json")
previsoes = xgb_model.predict(X_org)

mascara_erros = (previsoes == 0)
df_erros = df_text[mascara_erros].copy()


print(f" Encontrados {len(df_erros)} Falsos Negativos.")
print(f"Salvando o dossiê de erros em: {caminho_saida_erros}\n")
df_erros.to_csv(caminho_saida_erros, index=False)

# Mostra as 5 primeiras falhas diretamente na tela do Colab para análise rápida
print("--- Amostra Rápida dos Erros ---")
col_visualizar = 'Assunto_Anonimizado' if 'Assunto_Anonimizado' in df_erros.columns else 'Assunto'
col_texto = 'Texto_Limpo_Anonimizado' if 'Texto_Limpo_Anonimizado' in df_erros.columns else 'Texto_Limpo'

for idx, row in df_erros.head(5).iterrows():
    print(f"\n[Assunto]: {row.get(col_visualizar, '')}")
    print(f"[Corpo]: {row.get(col_texto, '')[:200]}...") 