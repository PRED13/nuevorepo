import os
import glob
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

PATH_TREC = "/home/pred/Documentos/datasets/datasets/trec07p/data/"

def train_spam_model():
    print("🧠 Entrenando con soporte de probabilidades...")
    emails = []
    # Usamos 2000 archivos para un entrenamiento sólido
    files = glob.glob(os.path.join(PATH_TREC, 'inmail.*'))[:2000]
    
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Etiquetado simple para prueba (puedes mejorar esto con el archivo index)
            label = 1 if "subject: adv" in content.lower() or "click here" in content.lower() else 0
            emails.append({'text': content, 'label': label})

    df = pd.DataFrame(emails)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    model = MultinomialNB()
    model.fit(X, y)

    # GUARDAR
    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("✅ Modelos guardados correctamente.")

if __name__ == "__main__":
    train_spam_model()