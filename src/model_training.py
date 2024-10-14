from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import joblib

# Definir y Entrenar Modelos
def entrenar_modelo_logistico(X_train, y_train):
    """
    Entrena un modelo de Regresión Logística.
    Args:
        X_train (DataFrame): Conjunto de entrenamiento.
        y_train (Series): Etiquetas de entrenamiento.
    Returns:
        LogisticRegression: Modelo entrenado.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def entrenar_modelo_random_forest(X_train, y_train):
    """
    Entrena un modelo Random Forest.
    Args:
        X_train (DataFrame): Conjunto de entrenamiento.
        y_train (Series): Etiquetas de entrenamiento.
    Returns:
        RandomForestClassifier: Modelo entrenado.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def entrenar_modelo_xgboost(X_train, y_train):
    """
    Entrena un modelo XGBoost.
    Args:
        X_train (DataFrame): Conjunto de entrenamiento.
        y_train (Series): Etiquetas de entrenamiento.
    Returns:
        XGBClassifier: Modelo entrenado.
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluar Modelos
def evaluar_modelo(model, X_test, y_test):
    """
    Evalúa un modelo usando diversas métricas.
    Args:
        model: Modelo a evaluar.
        X_test (DataFrame): Características de prueba.
        y_test (Series): Etiquetas de prueba.
    Returns:
        dict: Diccionario con métricas de evaluación.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred)
    }

# Guardar Modelo
def guardar_modelo(model, nombre_archivo):
    """
    Guarda un modelo en un archivo.
    Args:
        model: Modelo a guardar.
        nombre_archivo (str): Nombre del archivo.
    """
    joblib.dump(model, nombre_archivo)