import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


df_xgb = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/predicoes_xgboost_test.csv")
df_bert = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/resultados_bertimbau/predicoes_bertimbau_test.csv") 


df_comparacao = pd.merge(df_xgb, df_bert, on=['ID_original', 'y_true'])

# Classificar Acertos e Erros (True = Acertou, False = Errou)
df_comparacao['acerto_xgb'] = df_comparacao['y_pred_xgb'] == df_comparacao['y_true']
df_comparacao['acerto_bert'] = df_comparacao['y_pred_bert'] == df_comparacao['y_true']

# 4. Construir a Tabela de Contingência
# a = Ambos acertaram | b = XGBoost errou, BERT acertou
# c = XGBoost acertou, BERT errou | d = Ambos erraram
a = len(df_comparacao[(df_comparacao['acerto_xgb'] == True) & (df_comparacao['acerto_bert'] == True)])
b = len(df_comparacao[(df_comparacao['acerto_xgb'] == False) & (df_comparacao['acerto_bert'] == True)])
c = len(df_comparacao[(df_comparacao['acerto_xgb'] == True) & (df_comparacao['acerto_bert'] == False)])
d = len(df_comparacao[(df_comparacao['acerto_xgb'] == False) & (df_comparacao['acerto_bert'] == False)])

tabela_contingencia = [[a, b],
                       [c, d]]

print("--- Tabela de Contingência ---")
print(f"XGB Acertou | BERT Acertou: {a}")
print(f"XGB Errou   | BERT Acertou: {b}")
print(f"XGB Acertou | BERT Errou:   {c}")
print(f"XGB Errou   | BERT Errou:   {d}\n")

# Executar o Teste de McNemar 
resultado_mcnemar = mcnemar(tabela_contingencia, exact=True, correction=True)

print("--- Teste de McNemar ---")
print(f"Estatística (Qui-Quadrado): {resultado_mcnemar.statistic:.4f}")
print(f"P-Valor: {resultado_mcnemar.pvalue:.4e}")
