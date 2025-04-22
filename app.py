import sqlite3
from flask import Flask, request, jsonify, render_template
import joblib
from datetime import datetime
app = Flask(__name__)


DB_PATH = 'analises.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

conn = get_db()
conn.execute("""
  CREATE TABLE IF NOT EXISTS analises (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frase TEXT NOT NULL,
    resultado TEXT NOT NULL,
    timestamp TEXT NOT NULL
  )
""")
conn.commit()
conn.close()

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

    # ─── grava no histórico ───
    conn = get_db()
    conn.execute(
        "INSERT INTO analises (frase, resultado, timestamp) VALUES (?, ?, ?)",
        (frase, sentimento, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
    # ─────────────────────────

    return render_template('index.html', resultado=sentimento)

@app.route('/analise', methods=['POST'])
def analise_api():
    dados = request.json
    texto = dados['frase']
    entrada = vetorizador.transform([texto])
    resultado = modelo.predict(entrada)[0]
    sentimento = 'positivo' if resultado == 1 else 'negativo'
    return jsonify({'resultado': sentimento})


@app.route('/historico')
def historico():
    conn = get_db()
    rows = conn.execute(
        "SELECT frase, resultado, timestamp FROM analises ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return render_template('historico.html', historico=rows)


if __name__ == '__main__':
    app.run(debug=True)
