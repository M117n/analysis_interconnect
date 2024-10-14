# Análisis Exploratorio y Modelado de Tasa de Cancelación de Clientes para Interconnect

## Descripción
Este proyecto tiene como objetivo analizar los datos de la compañía **Interconnect** para entender y predecir la tasa de cancelación de clientes. Se aplica un análisis exploratorio exhaustivo seguido de la construcción de varios modelos de clasificación con el fin de mejorar la retención de clientes.

## Estructura del Proyecto
- **data/raw**: Datos originales provistos por la compañía.
- **data/processed**: Datos preparados para el análisis.
- **notebooks**: Contiene cuadernos Jupyter para:
  - `eda.ipynb`: Análisis Exploratorio de Datos (EDA).
  - `modeling.ipynb`: Entrenamiento y evaluación de modelos.
  - `analysis_results.ipynb`: Análisis de resultados y conclusiones.
- **src**: Código fuente organizado en módulos para un desarrollo estructurado.
- **requirements.txt**: Lista de todas las dependencias requeridas para reproducir el análisis.
- **final.pdf**: Informe detallado del proyecto.

## Resultados
Se realizaron pruebas con diferentes modelos de Machine Learning, como:
- **Regresión Logística**: AUC-ROC de 0.84 en conjunto de pruebas.
- **Random Forest**: AUC-ROC de 0.89 en el mejor caso.
- **XGBoost**: Modelo con mejor rendimiento, AUC-ROC de 0.92 en pruebas.

## Uso
1. **Clonar el repositorio:**
    ```
    git clone https://github.com/tuusuario/ProyectoInterconnect.git
    ```
2. **Instalar las dependencias:**
    ```
    pip install -r requirements.txt
    ```
3. **Ejecutar los notebooks**: Los cuadernos Jupyter se encuentran en la carpeta `notebooks/` y contienen el análisis paso a paso.

## Contacto
Para preguntas, problemas o sugerencias, no dudes en contactarme en [tu.email@example.com].
