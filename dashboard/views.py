from django.shortcuts import render
from .ml_logic.processor import load_arff_data, load_trec_emails
import joblib
import pandas as pd
import arff
import os
from django.shortcuts import render
from django.conf import settings

# Definimos la ruta base para encontrar los modelos .pkl en el servidor
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def index(request):
    # Inicializamos todas las variables del contexto vacías
    context = {
        'nsl_table': None,
        'chart_labels': None,
        'chart_values': None,
        'spam_result': None,
        'spam_prob_values': None,
        'file_name': None,
        'raw_snippet': None,
    }

    if request.method == 'POST':
        # --- BLOQUE 1: PROCESAR DATASET DE RED (.ARFF) ---
        arff_file = request.FILES.get('arff_file')
        if arff_file:
            try:
                # Leer el archivo .arff subido desde el teléfono o PC
                content = arff_file.read().decode('utf-8')
                data = arff.loads(content)
                
                # Extraer nombres de atributos y crear DataFrame
                attributes = [attr[0] for attr in data['attributes']]
                df_uploaded = pd.DataFrame(data['data'], columns=attributes)
                
                # Procesar datos para las gráficas (Protocolos)
                protocol_counts = df_uploaded['protocol_type'].value_counts().to_dict()
                
                context.update({
                    'nsl_table': df_uploaded.head(10).to_dict(orient='records'),
                    'chart_labels': list(protocol_counts.keys()),
                    'chart_values': list(protocol_counts.values()),
                })
            except Exception as e:
                context['error'] = f"Error al procesar el archivo ARFF: {e}"

        # --- BLOQUE 2: ANALIZADOR DE EMAILS (IA) ---
        raw_file = request.FILES.get('raw_file')
        if raw_file:
            try:
                # Leer el archivo de texto/email subido
                content_mail = raw_file.read().decode('utf-8', errors='ignore')
                
                # Rutas a los modelos guardados en la raíz del proyecto
                model_path = os.path.join(BASE_DIR, 'spam_model.pkl')
                vectorizer_path = os.path.join(BASE_DIR, 'vectorizer.pkl')
                
                # Cargar modelos
                model = joblib.load(model_path)
                vectorizer = joblib.load(vectorizer_path)
                
                # Realizar predicción y obtener probabilidades
                vector = vectorizer.transform([content_mail])
                prediction = model.predict(vector)[0]
                probabilities = model.predict_proba(vector)[0]
                
                # Convertir probabilidades a lista de floats para Chart.js
                prob_list = [round(float(p) * 100, 2) for p in probabilities]
                
                context.update({
                    'spam_result': "SPAM" if prediction == 1 else "HAM",
                    'spam_prob_values': prob_list,
                    'raw_snippet': content_mail[:800], # Snippet más largo para mejor visualización
                    'file_name': raw_file.name
                })
            except Exception as e:
                context['error'] = f"Error en el análisis de Email: {e}"

    return render(request, 'dashboard/index.html', context)