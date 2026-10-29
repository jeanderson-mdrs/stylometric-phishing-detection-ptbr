import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score


caminho_texto_org = "/content/drive/MyDrive/dataset_organico_phishing_rnp.csv" 
caminho_modelo_bert = "/content/drive/MyDrive/Colab Notebooks/modelo_bertimbau_phishing"

try:
    
    df_org = pd.read_csv(caminho_texto_org)
    df_org = df_org.dropna(how='all')

   
    col_assunto = 'Assunto_Anonimizado' if 'Assunto_Anonimizado' in df_org.columns else 'Assunto'
    col_texto = 'Texto_Limpo_Anonimizado' if 'Texto_Limpo_Anonimizado' in df_org.columns else 'Texto_Limpo'

    # Prepara o texto completo concatenado (igual no treinamento)
    df_org[col_assunto] = df_org[col_assunto].fillna('')
    df_org[col_texto] = df_org[col_texto].fillna('')
    textos_org = (df_org[col_assunto] + " - " + df_org[col_texto]).tolist()

    # Tudo é phishing (Label 1)
    y_org = [1] * len(textos_org)

    print(f"Textos orgânicos carregados: {len(textos_org)} amostras válidas.")
    print("Carregando o BERTimbau (Tokenizador e Modelo)...")

    # Carrega Modelo e Tokenizador para a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(caminho_modelo_bert)
    model = AutoModelForSequenceClassification.from_pretrained(caminho_modelo_bert)
    model.to(device)
    model.eval() # Modo de avaliação (desliga atualizações de peso)

    print("Iniciando inferência neural nos dados da RNP...\n")

    # 4. Inferência Direta
    previsoes = []
    for texto in textos_org:
        # Prepara o texto para a rede neural
        inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()
            previsoes.append(pred)

    # 5. Cálculo das Métricas
    phishing_detectado = sum(p == 1 for p in previsoes)
    phishing_nao_detectado = sum(p == 0 for p in previsoes)
    taxa_captura = accuracy_score(y_org, previsoes)

  
    print(" RESULTADO DA INFERÊNCIA NEURAL (BERTIMBAU) NO MUNDO REAL")

    print(f"Total de E-mails Reais de Phishing: {len(y_org)}")
    print(f" Ataques Bloqueados (Verdadeiros Positivos): {phishing_detectado}")
    print(f" Ataques que Passaram (Falsos Negativos): {phishing_nao_detectado}")
    print(f"\n Taxa de Captura (Recall / Sensibilidade): {taxa_captura:.2%}")
    print("============================================================")

except FileNotFoundError as e:
    print(f"Erro: Arquivo não encontrado. Verifique o nome do seu CSV de textos. {e}")
except KeyError as e:
    print(f"Erro de coluna: {e}. Verifique os cabeçalhos do seu arquivo CSV.")