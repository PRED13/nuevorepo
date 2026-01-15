import arff
import pandas as pd
import os
import glob

# RUTAS ABSOLUTAS (Directo a tus carpetas en Documentos)
PATH_ARFF = "/home/pred/Documentos/datasets/datasets/NSL-KDD/KDDTrain+.arff"
PATH_TREC = "/home/pred/Documentos/datasets/datasets/trec07p/data/" # Ajustado a la subcarpeta 'data'

def load_arff_data():
    """Carga el dataset NSL-KDD"""
    try:
        with open(PATH_ARFF, 'r') as f:
            data = arff.load(f)
            attributes = [attr[0] for attr in data['attributes']]
            df = pd.DataFrame(data['data'], columns=attributes)
            print(f"✅ NSL-KDD cargado: {df.shape}")
            return df
    except Exception as e:
        print(f"❌ Error en ARFF: {e}")
        return None

def load_trec_emails():
    import re
    inmail_data = []
    files = glob.glob(os.path.join(PATH_TREC, 'inmail.*'))
    
    for file_path in sorted(files)[:20]: # Solo 20 para la tabla del frontend
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Extraer el Asunto (Subject) usando Regex
                subject = re.search(r"Subject: (.*)", content)
                subject = subject.group(1) if subject else "Sin Asunto"
                
                inmail_data.append({
                    'file': os.path.basename(file_path),
                    'subject': subject[:50], # Limitamos largo
                    'snippet': content[content.find('\n\n'):][:100].strip() # Cuerpo del mensaje
                })
        except:
            continue
    return pd.DataFrame(inmail_data)

if __name__ == "__main__":
    df_nsl = load_arff_data()
    df_emails = load_trec_emails()