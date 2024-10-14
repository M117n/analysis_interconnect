import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Funciones de Preprocesamiento de Datos
def cargar_datos(file_paths):
    """
    Carga los datos desde las rutas especificadas y los devuelve como DataFrames.
    Args:
        file_paths (dict): Diccionario con las rutas de los archivos.
    Returns:
        dict: Diccionario con los DataFrames cargados.
    """
    data = {}
    for key, path in file_paths.items():
        data[key] = pd.read_csv(path)
    return data

def limpiar_datos(df):
    """
    Limpia los datos realizando conversiones de tipos y manejando valores faltantes.
    Args:
        df (DataFrame): DataFrame a limpiar.
    Returns:
        DataFrame: DataFrame limpio.
    """
    # Convertir fechas y manejar valores faltantes
    df['BeginDate'] = pd.to_datetime(df['BeginDate'])
    df['EndDate'] = pd.to_datetime(df['EndDate'], errors='coerce')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df.fillna('No', inplace=True)
    return df

def codificar_variables(df, cols_to_encode):
    """
    Codifica variables categóricas en variables numéricas.
    Args:
        df (DataFrame): DataFrame con variables categóricas.
        cols_to_encode (list): Lista de columnas a codificar.
    Returns:
        DataFrame: DataFrame con variables codificadas.
    """
    le = LabelEncoder()
    for col in cols_to_encode:
        df[col] = le.fit_transform(df[col])
    return df

def dividir_datos(X, y, test_size=0.15, validation_size=0.15):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba.
    Args:
        X (DataFrame): Características.
        y (Series): Objetivo.
        test_size (float): Tamaño del conjunto de prueba.
        validation_size (float): Tamaño del conjunto de validación.
    Returns:
        tuple: Conjuntos de entrenamiento, validación y prueba.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + validation_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size / (test_size + validation_size), random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def balancear_datos(X, y):
    """
    Aplica SMOTE para balancear las clases.
    Args:
        X (DataFrame): Características de entrenamiento.
        y (Series): Objetivo de entrenamiento.
    Returns:
        tuple: Características y objetivo balanceados.
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res