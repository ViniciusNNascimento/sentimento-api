from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

modelo = joblib.load('modelo.pkl')
vetorizador = joblib.load('vetorizador.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analise-form', methods=['POST'])
def analise_form():
    frase = request.form['frase']
    entrada = vetorizador.transform([frase])
    resultado = modelo.predict(entrada)[0]
    sentimento = 'positivo' if resultado == 1 else 'negativo'
    return render_template('index.html', resultado=sentimento)

@app.route('/analise', methods=['POST'])
def analise_api():
    dados = request.json
    texto = dados['frase']
    entrada = vetorizador.transform([texto])
    resultado = modelo.predict(entrada)[0]
    sentimento = 'positivo' if resultado == 1 else 'negativo'
    return jsonify({'resultado': sentimento})

if __name__ == '__main__':
    app.run(debug=True)
