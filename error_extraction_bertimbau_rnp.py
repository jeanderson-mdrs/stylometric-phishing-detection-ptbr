import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


caminho_texto_org = "/content/drive/MyDrive/Colab Notebooks/Datasets/Novo/dataset_organico_phishing_rnp.csv"
caminho_modelo_bert = "/content/drive/MyDrive/Colab Notebooks/Modelos/Novos/modelo_bertimbau_phishing"
caminho_saida_erros_bert = "/content/drive/MyDrive/Colab Notebooks/Datasets/Novo/falsos_negativos_bertimbau.csv"

# Carregamento dos dados
df_org = pd.read_csv(caminho_texto_org).dropna(how='all')
col_assunto = 'Assunto_Anonimizado' if 'Assunto_Anonimizado' in df_org.columns else 'Assunto'
col_texto = 'Texto_Limpo_Anonimizado' if 'Texto_Limpo_Anonimizado' in df_org.columns else 'Texto_Limpo'

df_org[col_assunto] = df_org[col_assunto].fillna('')
df_org[col_texto] = df_org[col_texto].fillna('')
textos_org = (df_org[col_assunto] + " - " + df_org[col_texto]).tolist()

# Carregamento do BERTimbau
print("Carregando o BERTimbau para extração dos erros...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(caminho_modelo_bert)
model = AutoModelForSequenceClassification.from_pretrained(caminho_modelo_bert)
model.to(device)
model.eval()

# Inferência para capturar as previsões
previsoes = []
for texto in textos_org:
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        previsoes.append(pred)

# Injeção das previsões no dataframe original e extração
df_org['Previsao_BERT'] = previsoes
df_erros_bert = df_org[df_org['Previsao_BERT'] == 0].copy() 


print(f" Encontrados {len(df_erros_bert)} Falsos Negativos do BERTimbau.")
print(f"Salvando o dossiê de erros neurais em: {caminho_saida_erros_bert}\n")
df_erros_bert.to_csv(caminho_saida_erros_bert, index=False)

for idx, row in df_erros_bert.head(3).iterrows():
    print(f"\n[Assunto]: {row.get(col_assunto, '')}")
    print(f"[Corpo]: {row.get(col_texto, '')[:150]}...")