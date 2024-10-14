import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Funciones de Visualización
def graficar_distribucion(df, columna):
    """
    Grafica la distribución de una columna específica.
    Args:
        df (DataFrame): DataFrame que contiene la columna.
        columna (str): Columna a graficar.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[columna], kde=True)
    plt.title(f"Distribución de {columna}")
    plt.xlabel(columna)
    plt.ylabel("Frecuencia")
    plt.show()

def graficar_roc(model, X_test, y_test):
    """
    Grafica la curva ROC para el modelo proporcionado.
    Args:
        model: Modelo a evaluar.
        X_test (DataFrame): Características de prueba.
        y_test (Series): Etiquetas de prueba.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

# Funciones Auxiliares
def obtener_metricas(evaluacion):
    """
    Imprime las métricas de evaluación del modelo.
    Args:
        evaluacion (dict): Diccionario con métricas de evaluación.
    """
    for key, value in evaluacion.items():
        print(f"{key}: {value:.2f}")