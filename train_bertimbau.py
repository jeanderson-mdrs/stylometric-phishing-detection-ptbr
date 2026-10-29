import pandas as pd
import numpy as np
import time
import torch
import random
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score



SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.cuda.empty_cache()


caminho_in = "/content/drive/MyDrive/Colab Notebooks/dataset_completo_com_tema.csv"
df = pd.read_csv(caminho_in)




df['ID_original'] = df.index.astype(str)

# Concatenação, Limpeza e Normalização
df['Assunto'] = df['Assunto'].fillna('')
df['Texto_Limpo'] = df['Texto_Limpo'].fillna('')
df['Texto_Completo'] = df['Assunto'].astype(str) + " - " + df['Texto_Limpo'].astype(str)

df['Texto_Normalizado'] = df['Texto_Completo'].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()

# Estratificação 
df['Modelo_Gerador'] = df['Modelo_Gerador'].fillna('Desconhecido')
df['Extensao_Classe'] = df['Extensao_Classe'].fillna('Desconhecida')
df['Tema'] = df['Tema'].fillna('Geral')
df['chave_estratificacao'] = (
    df['Modelo_Gerador'].astype(str) + "_" +
    df['Label'].astype(str) + "_" +
    df['Extensao_Classe'].astype(str) + "_" +
    df['Tema'].astype(str)
)


# PARTICIONAMENTO COM IDs

df_dev, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df['chave_estratificacao']
)

df_train, df_val = train_test_split(
    df_dev,
    test_size=0.1,
    random_state=SEED,
    stratify=df_dev['chave_estratificacao']
)

# Salvar os IDs para o XGBoost usar depois
df_train[['ID_original']].to_csv("ids_train.csv", index=False)
df_val[['ID_original']].to_csv("ids_val.csv", index=False)
df_test[['ID_original']].to_csv("ids_test.csv", index=False)
print(" Arquivos de IDs exportados com sucesso (ids_train.csv, ids_val.csv, ids_test.csv).")


# AUDITORIA DE DUPLICATAS (VAZAMENTO DE DADOS)

textos_train = set(df_train['Texto_Normalizado'])
textos_val = set(df_val['Texto_Normalizado'])
textos_test = set(df_test['Texto_Normalizado'])

print("\n--- Auditoria de Interseção Textual ---")
print(f"Duplicatas exatas no dataset inteiro: {df['Texto_Normalizado'].duplicated().sum()}")
print(f"Vazamento (Treino ∩ Validação): {len(textos_train & textos_val)}")
print(f"Vazamento (Treino ∩ Teste): {len(textos_train & textos_test)}")
print(f"Vazamento (Validação ∩ Teste): {len(textos_val & textos_test)}")
print("---------------------------------------\n")

# Extração final das listas para o Hugging Face
X_train, y_train = df_train['Texto_Completo'].tolist(), df_train['Label'].tolist()
X_val, y_val = df_val['Texto_Completo'].tolist(), df_val['Label'].tolist()
X_test, y_test = df_test['Texto_Completo'].tolist(), df_test['Label'].tolist()

# Formatação para Dataset Hugging Face
train_dataset = Dataset.from_dict({"text": X_train, "label": y_train})
val_dataset = Dataset.from_dict({"text": X_val, "label": y_val})
test_dataset = Dataset.from_dict({"text": X_test, "label": y_test})

# Carregamento do Tokenizador
modelo_nome = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(modelo_nome)

def tokenizar_funcao(exemplos):
    return tokenizer(exemplos["text"], padding="max_length", truncation=True, max_length=512)

print("Tokenizando os textos...")
tokenized_train = train_dataset.map(tokenizar_funcao, batched=True)
tokenized_val = val_dataset.map(tokenizar_funcao, batched=True)
tokenized_test = test_dataset.map(tokenizar_funcao, batched=True)

# Carregamento do Modelo Neural
model = AutoModelForSequenceClassification.from_pretrained(modelo_nome, num_labels=2)


# MÉTRICA EXPLÍCITA E SEEDS NO TRAINER

training_args = TrainingArguments(
    output_dir="./resultados_bertimbau",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",      
    greater_is_better=True,         
    seed=SEED,                      
    data_seed=SEED,                  
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# Treinamento
print("\nIniciando o Fine-Tuning do BERTimbau na GPU...")
inicio_treino = time.time()
trainer.train()
tempo_treino = time.time() - inicio_treino
print(f"Treinamento concluído em {tempo_treino:.2f} segundos.")

# Avaliação Cega
print("\nRealizando predições no conjunto de Teste Intocado (20%)...")
predictions_output = trainer.predict(tokenized_test)
previsoes_bert = np.argmax(predictions_output.predictions, axis=-1)

f1_final_bert = f1_score(y_test, previsoes_bert)

print("\n--- Resultados do Baseline Neural (BERTimbau) ---")
print(f"Tempo de Treinamento: {tempo_treino:.2f} segundos")
print(f"Acurácia: {accuracy_score(y_test, previsoes_bert):.4f}")
print(f"F1-Score (Classe Positiva): {f1_final_bert:.4f}\n")
print("Relatório de Classificação (Teste Final 20%):")
print(classification_report(y_test, previsoes_bert))


# EXPORTAÇÃO DAS PREDIÇÕES PARA O TESTE DE MCNEMAR

os.makedirs("/content/drive/MyDrive/Colab Notebooks/Modelos/Novos/resultados_bertimbau", exist_ok=True)

resultado_bert = pd.DataFrame({
    'ID_original': df_test['ID_original'].values,
    'y_true': y_test,
    'y_pred_bert': previsoes_bert
})

caminho_predicoes_bert = "/content/drive/MyDrive/Colab Notebooks/resultados_bertimbau/predicoes_bertimbau_test.csv"
resultado_bert.to_csv(caminho_predicoes_bert, index=False)
print(f"Tabela de predições exportada para McNemar em: {caminho_predicoes_bert}")