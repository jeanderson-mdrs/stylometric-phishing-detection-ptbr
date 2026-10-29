import matplotlib.pyplot as plt
import pandas as pd

# 1. Extração das importâncias do melhor modelo validado
importancias = melhor_xgb.feature_importances_

nomes_features = X_dev.columns

# 2. Estruturação dos dados para o gráfico
df_importancias = pd.DataFrame({
    'Atributo': nomes_features,
    'Ganho_de_Informacao': importancias
}).sort_values(by='Ganho_de_Informacao', ascending=True)

# 3. Plotagem das variáveis mais decisivas
plt.figure(figsize=(10, 8)) 

plt.barh(df_importancias['Atributo'].tail(15), df_importancias['Ganho_de_Informacao'].tail(15), color='#2c3e50')

plt.xlabel('Importância Relativa (Ganho de Informação)', fontsize=12)
plt.ylabel('Atributos Estilométricos', fontsize=12)
plt.title('Auditoria White-Box: Atributos Mais Relevantes na Detecção (XGBoost)', fontsize=14, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()