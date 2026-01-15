import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from .processor import load_arff_data

def train_network_model():
    df = load_arff_data()
    if df is None: return
    
    # 1. Preprocesamiento: Convertir texto a números
    le = LabelEncoder()
    # Las columnas 1, 2, 3 de NSL-KDD suelen ser protocol_type, service, y flag
    text_cols = ['protocol_type', 'service', 'flag']
    for col in text_cols:
        df[col] = le.fit_transform(df[col])
    
    # La última columna es el 'class' (normal o anomaly)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    print("🚀 Entrenando modelo de red (NSL-KDD)...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X, y)
    
    # 2. Guardar el modelo y el encoder
    joblib.dump(model, 'network_model.pkl')
    print("✅ Modelo guardado como network_model.pkl")
    return model

if __name__ == "__main__":
    train_network_model()