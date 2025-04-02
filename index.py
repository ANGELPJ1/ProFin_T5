from flask import Flask, render_template, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Cargar el modelo T5
nombre_modelo = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(nombre_modelo)
modelo = T5ForConditionalGeneration.from_pretrained(nombre_modelo)

@app.route('/')
def index():
    return render_template('menu.html')

@app.route('/procesar', methods=['POST'])
def procesar():
    datos = request.json
    tarea = datos['tarea']
    texto = datos['texto']
    
    if tarea == "resumir":
        entrada = f"summarize: {texto}"
    elif tarea == "traducir":
        entrada = f"translate English to French: {texto}"
    elif tarea == "preguntar":
        contexto = datos.get('contexto', '')
        entrada = f"question: {texto} context: {contexto}"
    else:
        return jsonify({'error': 'Tarea no válida'})
    
    input_ids = tokenizer(entrada, return_tensors="pt").input_ids
    output_ids = modelo.generate(input_ids, max_length=512)
    resultado = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return jsonify({'resultado': resultado})

if __name__ == '__main__':
    app.run(debug=True)
