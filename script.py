from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Cargar el modelo entrenado
model_file_path = 'checkpoints/gb_model.pkl'
with open(model_file_path, 'rb') as file:
    gb_model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/realizar_prediccion', methods=['POST'])
def prediccion():
    datos = [
        float(request.form['monto_compras_nac']),
        float(request.form['monto_compras_int']),
        float(request.form['monto_avances_nac']),
        float(request.form['monto_avances_int']),
        float(request.form['pagos_nacional'])
    ]
    resultado = gb_model.predict([datos])
    return f'La predicción es: {"Cliente Moroso" if resultado == 1 else "Cliente No Moroso"}'

if __name__ == '__main__':
    app.run(debug=True)
