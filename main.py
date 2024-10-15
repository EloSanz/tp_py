# Importar las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Cargar el Dataset
def cargar_datos(ruta):
    # Cargar el dataset desde un archivo CSV
    data = pd.read_csv(ruta)
    return data

# 2. Análisis Exploratorio de Datos
def analisis_exploratorio(data):
    # Información básica del dataset
    print("Información del dataset:")
    print(data.info())

    # Estadísticas descriptivas
    print("\nEstadísticas descriptivas:")
    print(data.describe())

    # Verificar valores nulos
    print("\nValores nulos por columna:")
    print(data.isnull().sum())

# 3. Visualización de Datos
def visualizar_datos(data):
    # Histograma de la variable "Amount"
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Amount'], bins=50, kde=True)
    plt.title('Distribución de la Variable Amount')
    plt.show()

    # Matriz de correlación
    plt.figure(figsize=(15, 10))
    sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.show()

# 4. Preparación de los Datos
def preparar_datos(data):
    # Variables predictoras (X) y objetivo (y)
    X = data.drop(columns=['Class', 'Time'])  # Eliminamos las columnas Class y Time de las variables predictoras
    y = data['Class']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}")
    print(f"Tamaño del conjunto de prueba: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

# 5. Entrenamiento y Evaluación del Modelo
def entrenar_y_evaluar(X_train, X_test, y_train, y_test):
    # Crear el modelo RandomForest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Entrenar el modelo
    clf.fit(X_train, y_train)

    # Predicción
    y_pred = clf.predict(X_test)

    # Evaluación del modelo
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    # Guardar el modelo entrenado
    joblib.dump(clf, 'random_forest_model.pkl')
    print("\nModelo guardado como 'random_forest_model.pkl'.")

# 6. Función principal
if __name__ == '__main__':
    ruta_archivo = 'creditcard.csv'  # Reemplaza con la ruta de tu archivo CSV

    # Cargar los datos
    data = cargar_datos(ruta_archivo)

    # Realizar análisis exploratorio de datos
    analisis_exploratorio(data)

    # Visualizar los datos
    visualizar_datos(data)

    # Preparar los datos para el modelo
    X_train, X_test, y_train, y_test = preparar_datos(data)

    # Entrenar y evaluar el modelo
    entrenar_y_evaluar(X_train, X_test, y_train, y_test)
