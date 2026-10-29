import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score


caminho_features_org = "/content/drive/MyDrive/Colab Notebooks/dataset_organico_features_rnp.csv"

try:
    df_org = pd.read_csv(caminho_features_org)


    df_org = df_org.dropna(how='all')


    df_org['Label'] = 1

    print(f"Dataset orgânico carregado: {len(df_org)} amostras válidas.")

    # Separação das variáveis
    colunas_ignorar = ['Label', 'Message_ID']
    X_org = df_org.drop(columns=[col for col in colunas_ignorar if col in df_org.columns])

    # Converte explicitamente para inteiro
    y_org = df_org['Label'].astype(int)

    # Carregar o Modelo XGBoost Global Treinado
    caminho_modelo = "/content/drive/MyDrive/Colab Notebooks/modelo_xgboost_phishing.json"

    print("Carregando o modelo XGBoost...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(caminho_modelo)

   
    print("Iniciando inferência nos dados do Zimbra...\n")
    previsoes = xgb_model.predict(X_org)

   
    total_emails = len(y_org)
    phishing_detectado = sum(previsoes == 1)      # Verdadeiros Positivos
    phishing_nao_detectado = sum(previsoes == 0)  # Falsos Negativos
    taxa_captura = accuracy_score(y_org, previsoes) # Acurácia = Recall neste cenário


    print(" RESULTADO DA INFERÊNCIA NO DATASET ORGÂNICO ")

    print(f"Total de E-mails Reais de Phishing: {total_emails}")
    print(f" Ataques Bloqueados (Verdadeiros Positivos): {phishing_detectado}")
    print(f" Ataques que Passaram (Falsos Negativos): {phishing_nao_detectado}")
    print(f"\n Taxa de Captura (Recall / Sensibilidade): {taxa_captura:.2%}")
    print("============================================================")

except FileNotFoundError as e:
    print(f"Erro ao localizar os arquivos. Verifique os caminhos: {e}")