import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
import xgboost as xgb
import time
import os
from sklearn.model_selection import StratifiedKFold

cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 1. IMPORTAÇÃO DOS IDs PARA PARIDADE ABSOLUTA COM O BERTIMBAU

if 'ID_original' not in df_final.columns:
    df_final['ID_original'] = df_final.index.astype(str)
else:
    df_final['ID_original'] = df_final['ID_original'].astype(str)

# Carrega os arquivos salvos na etapa do modelo neural
ids_train = pd.read_csv("ids_train.csv")['ID_original'].astype(str).tolist()
ids_val = pd.read_csv("ids_val.csv")['ID_original'].astype(str).tolist()
ids_test = pd.read_csv("ids_test.csv")['ID_original'].astype(str).tolist()

# O XGBoost fará Validação Cruzada (CV=5) usando Treino + Validação (80%)
ids_desenvolvimento = ids_train + ids_val

# 2. Filtragem e Particionamento Exato
df_dev = df_final[df_final['ID_original'].isin(ids_desenvolvimento)]
df_test = df_final[df_final['ID_original'].isin(ids_test)]

print(f"XGBoost Desenvolvimento (GridSearch): {len(df_dev)} amostras")
print(f"XGBoost Teste Intocado (Cofre Comum): {len(df_test)} amostras")


# 3. SEPARAÇÃO DE VARIÁVEIS (Features vs Target)

# Removemos todos os metadados para não vazar informações para as árvores
colunas_metadados = [
    'Label', 'Modelo_Gerador', 'Extensao_Classe', 'Tema', 'ID_original',
    'Assunto', 'Texto_Limpo', 'Texto_Completo', 'Texto_Normalizado', 'chave_estratificacao'
]

X_dev = df_dev.drop(columns=[col for col in colunas_metadados if col in df_dev.columns], errors='ignore')
y_dev = df_dev['Label']

X_test = df_test.drop(columns=[col for col in colunas_metadados if col in df_test.columns], errors='ignore')
y_test = df_test['Label']


# 4. OTIMIZAÇÃO E TREINAMENTO (GridSearchCV)

param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

print("\nIniciando a Otimização de Hiperparâmetros (GridSearchCV)...")
inicio_treino = time.time()


xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    eval_metric='logloss'
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_dev, y_dev)
tempo_treino = time.time() - inicio_treino

print(f"\n Treinamento concluído em {tempo_treino:.2f} segundos.")
print(f"Melhores hiperparâmetros encontrados: {grid_search.best_params_}")


# 5. INFERÊNCIA E AVALIAÇÃO CEGA

melhor_xgb = grid_search.best_estimator_
previsoes = melhor_xgb.predict(X_test)

f1_final = f1_score(y_test, previsoes)

print(f"\n--- Resultados do XGBoost Global Otimizado ---")
print(f"Acurácia: {accuracy_score(y_test, previsoes):.4f}")
print(f"F1-Score (Classe Positiva): {f1_final:.4f}\n")
print("Relatório de Classificação (Teste Final 20%):")
print(classification_report(y_test, previsoes))


caminho_diretorio = "/content/drive/MyDrive/Colab Notebooks/"
os.makedirs(caminho_diretorio, exist_ok=True)

caminho_modelo = f"{caminho_diretorio}/modelo_xgboost_phishing.json"
melhor_xgb.save_model(caminho_modelo)
print(f"\nModelo global validado e salvo com sucesso em: {caminho_modelo}")


# 6. EXPORTAÇÃO DAS PREDIÇÕES PARA O TESTE DE MCNEMAR
resultado_xgb = pd.DataFrame({
    'ID_original': df_test['ID_original'].values,
    'y_true': y_test.values,
    'y_pred_xgb': previsoes
})

caminho_predicoes = f"{caminho_diretorio}/predicoes_xgboost_test.csv"
resultado_xgb.to_csv(caminho_predicoes, index=False)
print(f"Tabela de predições exportada para McNemar em: {caminho_predicoes}")