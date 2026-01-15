from django.shortcuts import render
from django.conf import settings
import joblib
import pandas as pd
import arff
import os

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
        'error': None,
    }

    if request.method == 'POST':
        # --- BLOQUE 1: PROCESAR DATASET DE RED (.ARFF) ---
        arff_file = request.FILES.get('arff_file')
        if arff_file:
            try:
                # Leer el archivo .arff
                content = arff_file.read().decode('utf-8')
                
                # Cargar la estructura ARFF
                data = arff.loads(content)
                
                # Extraer nombres de atributos
                attributes = [attr[0] for attr in data['attributes']]
                
                # OPTIMIZACIÓN CRÍTICA PARA RENDER (LIMITAR RAM)
                # Tu archivo KDDTrain+.arff tiene 125k filas. El plan gratuito no lo soporta.
                # Tomamos solo las primeras 1000 filas para que la gráfica cargue sin error 502.
                limited_data = data['data'][:1000]
                df_uploaded = pd.DataFrame(limited_data, columns=attributes)
                
                # Procesar datos para las gráficas (Protocolos)
                if 'protocol_type' in df_uploaded.columns:
                    protocol_counts = df_uploaded['protocol_type'].value_counts().to_dict()
                    context['chart_labels'] = list(protocol_counts.keys())
                    context['chart_values'] = list(protocol_counts.values())
                
                # Tabla para mostrar en el HTML (primeras 10 filas)
                context['nsl_table'] = df_uploaded.head(10).to_dict(orient='records')
                
            except Exception as e:
                context['error'] = f"Error en ARFF: {str(e)}"

        # --- BLOQUE 2: ANALIZADOR DE EMAILS (IA) ---
        raw_file = request.FILES.get('raw_file')
        if raw_file:
            try:
                # Leer el contenido del email (archivo inmail.1)
                content_mail = raw_file.read().decode('utf-8', errors='ignore')
                
                # Definir rutas a los modelos en la raíz del proyecto
                model_path = os.path.join(settings.BASE_DIR, 'spam_model.pkl')
                vectorizer_path = os.path.join(settings.BASE_DIR, 'vectorizer.pkl')
                
                # Verificar si los archivos existen antes de intentar cargarlos
                if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
                    raise FileNotFoundError("No se encontraron los archivos .pkl en la raíz del proyecto.")

                # Cargar modelos
                model = joblib.load(model_path)
                vectorizer = joblib.load(vectorizer_path)
                
                # Realizar predicción
                vector = vectorizer.transform([content_mail])
                prediction = model.predict(vector)[0]
                probabilities = model.predict_proba(vector)[0]
                
                # Convertir a porcentajes para la gráfica
                prob_list = [round(float(p) * 100, 2) for p in probabilities]
                
                context.update({
                    'spam_result': "ES SPAM" if prediction == 1 else "NO ES SPAM (HAM)",
                    'spam_prob_values': prob_list,
                    'raw_snippet': content_mail[:800],
                    'file_name': raw_file.name
                })
            except Exception as e:
                context['error'] = f"Error en Análisis de Email: {str(e)}"

    return render(request, 'dashboard/index.html', context)