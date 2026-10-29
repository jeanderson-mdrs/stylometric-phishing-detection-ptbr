import os
import json
import pandas as pd
import httpx
from openai import OpenAI
import time

# CONFIGURAÇÃO DO SERVIDOR LOCAL (INTRANET)

cliente_http_inseguro = httpx.Client(verify=False)

client = OpenAI(
    api_key="",
    base_url="https:///v1",
    http_client=cliente_http_inseguro
)

# Modelos Utilizados
MODELOS = [
    "gemma-4-26B-A4B-it",
    "gpt-oss-120b",
    "qwen36-35b-fp8",
    "deepseek-v4-flash-0731"
]

TEMAS = [
    "ofertas promocionais", "oportunidades financeiras", "emprego",
    "atualizações de TI", "saúde", "notificações financeiras", "relacionamentos"
]

DIMENSOES = {
    "Curto": "O texto deve ter obrigatoriamente entre 30 e 70 palavras.",
    "Medio": "O texto deve ter obrigatoriamente entre 100 e 200 palavras.",
    "Longo": "O texto deve ter obrigatoriamente entre 250 e 400 palavras."
}

def gerar_dataset_local(amostras_por_combinacao=2):
    resultados = []

    for modelo in MODELOS:
        print(f"\n--- Iniciando geração no servidor local com: {modelo} ---")
        for tema in TEMAS:
            for rotulo_tam, regra_tam in DIMENSOES.items():
                for i in range(amostras_por_combinacao):

                    prompt_phishing = (
                        f"Gere um e-mail de phishing sobre {tema}. Escreva de forma funcional e direta, "
                        f"simulando tratativa corporativa. IMPORTANTE: 1. {regra_tam} 2. O texto deve conter a mecânica do ataque: instrua a vítima a interagir com algo (exemplo: orientar o acesso a um link, a abertura de um documento, etc). 3. Não inclua metadados "
                        f"como 'Assunto:', 'Corpo:', ou termos como 'Link de phishing', 'Link malicioso' e/ou 'link falso'."
                    )

                    prompt_legitimo = (
                        f"Gere um e-mail profissional legítimo sobre {tema}. O texto deve ser técnico, "
                        f"formal e seguir o padrão de uma organização respeitável. IMPORTANTE: 1. {regra_tam} "
                        f" 2. O email dever ser claro, "
                        f"amigável e confiável."
                    )

                    for tipo_prompt, label in [(prompt_phishing, 1), (prompt_legitimo, 0)]:
                        try:
                            resposta = client.chat.completions.create(
                                messages=[
                                    {"role": "system", "content": "Você é um assistente que retorna APENAS JSON válido com as chaves 'Assunto' e 'Texto_Limpo'."},
                                    {"role": "user", "content": tipo_prompt}
                                ],
                                model=modelo,
                                response_format={"type": "json_object"},
                                temperature=0.7 + (i * 0.1)
                            )

                            conteudo = json.loads(resposta.choices[0].message.content)
                            resultados.append({
                                "Assunto": conteudo.get("Assunto", "").strip(),
                                "Texto_Limpo": conteudo.get("Texto_Limpo", "").strip(),
                                "Modelo_Gerador": modelo,
                                "Extensao_Classe": rotulo_tam,
                                "Tema": tema,
                                "Label": label
                            })
                            print(f"✅ {modelo} | {tema} | {rotulo_tam} | Label: {label}")

                     
                            time.sleep(0.5)

                        except Exception as e:
                            print(f"Erro na comunicação com o servidor local ({modelo} | {tema}): {e}")

    return resultados

# Execução

print("Iniciando pipeline de geração...")
amostras = gerar_dataset_local(amostras_por_combinacao=10)
df_sintetico = pd.DataFrame(amostras)
df_sintetico.to_csv("novo_dataset_sintetico.csv", index=False, encoding="utf-8")
print(f" Dataset gerado com sucesso! {len(df_sintetico)} amostras salvas em novo_dataset_sintetico.csv")