import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Base simples de exemplo
dados = {
    'frase': [
        'Amei o produto', 'Ótimo atendimento', 'Muito ruim',
        'Veio estragado', 'Excelente qualidade', 'Péssimo serviço'
    ],
    'sentimento': [1, 1, 0, 0, 1, 0]  # 1 = positivo, 0 = negativo
}

df = pd.DataFrame(dados)

# Transformar texto em número
vetorizador = TfidfVectorizer()
X = vetorizador.fit_transform(df['frase'])
y = df['sentimento']

# Treinar modelo
modelo = LogisticRegression()
modelo.fit(X, y)

# Salvar modelo e vetorizador
joblib.dump(modelo, 'modelo.pkl')
joblib.dump(vetorizador, 'vetorizador.pkl')

print("Modelo treinado e salvo com sucesso!")
