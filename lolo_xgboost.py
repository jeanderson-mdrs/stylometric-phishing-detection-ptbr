import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import time


caminho_features = "/content/drive/MyDrive/Colab Notebooks/dataset_completo_features.csv"

df_feat = pd.read_csv(caminho_features)


param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

modelos_geradores = df_feat['Modelo_Gerador'].unique()
resultados_f1 = []

print("Iniciando Validação Cross-Domain Aninhada (Com GridSearch)...\n")
inicio_total = time.time()

for modelo_alvo in modelos_geradores:
    print(f"==================================================")
    print(f"🚀 ALVO DE TESTE (ISOLADO): {modelo_alvo.upper()}")
    print(f"==================================================")

    teste_df = df_feat[df_feat['Modelo_Gerador'] == modelo_alvo]
    treino_df = df_feat[df_feat['Modelo_Gerador'] != modelo_alvo]


    colunas_metadados = ['Label', 'Modelo_Gerador', 'Extensao_Classe', 'Tema']

    X_train = treino_df.drop(columns=colunas_metadados, errors='ignore')
    y_train = treino_df['Label']

    X_test = teste_df.drop(columns=colunas_metadados, errors='ignore')
    y_test = teste_df['Label']

  
    xgb_base = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    print(f"Treinando e otimizando com os modelos: {', '.join([m for m in modelos_geradores if m != modelo_alvo])}...")

    # Treinamento 
    grid_search.fit(X_train, y_train)
    melhor_modelo = grid_search.best_estimator_

    # Inferência no domínio isolado
    previsoes = melhor_modelo.predict(X_test)

    # Coleta de Métricas
    acc = accuracy_score(y_test, previsoes)
    f1 = f1_score(y_test, previsoes)
    resultados_f1.append({'Modelo_Teste': modelo_alvo, 'F1_Score': f1, 'Acuracia': acc})

    print(f"Melhores parâmetros encontrados para este fold: {grid_search.best_params_}")
    print(f"F1-Score obtido contra o {modelo_alvo}: {f1:.4f}")

    cm = confusion_matrix(y_test, previsoes)
    print(f"Matriz de Confusão:\n{cm}")


    nome_arquivo = f"/content/drive/MyDrive/Colab Notebooks/modelo_xgb_cego_para_{modelo_alvo}.json"
    melhor_modelo.save_model(nome_arquivo)
    print(f" Modelo salvo: {nome_arquivo}\n")

tempo_total = time.time() - inicio_total

# Resumo Final

print("RESUMO CROSS-DOMAIN")
df_resultados = pd.DataFrame(resultados_f1)
print(df_resultados)
print(f"\nF1-Score Médio Cross-Domain: {df_resultados['F1_Score'].mean():.4f}")
print(f"Tempo total de execução (Nested CV): {tempo_total:.2f} segundos")