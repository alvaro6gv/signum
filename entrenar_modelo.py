"""
############################################################

    ███████╗██╗ ██████╗ ███╗   ██╗██╗   ██╗███╗   ███╗
    ██╔════╝██║██╔════╝ ████╗  ██║██║   ██║████╗ ████║
    ███████╗██║██║  ███╗██╔██╗ ██║██║   ██║██╔████╔██║
    ╚════██║██║██║   ██║██║╚██╗██║██║   ██║██║╚██╔╝██║
    ███████║██║╚██████╔╝██║ ╚████║╚██████╔╝██║ ╚═╝ ██║
    ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝

    Proyecto Signum - TFG
    Autor: Álvaro José Gullón Vega

    pip install tensorflow pandas numpy scikit-learn

############################################################
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# =============================================================================
#  CONFIGURACION
# =============================================================================

FRAMES_POR_MUESTRA = 20
FEATURES           = 8
DIRECTORIO_DATASET = "dataset"
RUTA_MODELO_TFLITE = "modelo_signum.tflite"
ETIQUETAS_VOZ = {
    "buenos_dias"   : "Buenos días",
    "gracias"       : "Gracias",
    "cuanto_cuesta" : "¿Cuánto cuesta?",
    "captura"       : "__ACTIVACION__",
    "ninguno"       : "__RECHAZO__",
}

# Hiperparametros del modelo
LSTM_UNIDADES  = 64
DENSE_UNIDADES = 32
DROPOUT_RATE   = 0.3
EPOCHS         = 100
BATCH_SIZE     = 16
PATIENCE       = 15


# =============================================================================
#  CARGA Y PREPARACION DE DATOS
# =============================================================================
def cargar_datasets():
    # CSV Datasets -> Arrays NumPy -> Tensor para Tensorflow
    columnas_features = [
        "Meñique_%", "Pulgar_%", "Índice_%", "Corazón_%", "Anular_%",
        "Acel_X", "Acel_Y", "Acel_Z"
    ]

    muestras = []
    etiquetas = []

    archivos_csv = [f for f in os.listdir(DIRECTORIO_DATASET) if f.endswith(".csv")]

    if not archivos_csv:
        raise FileNotFoundError(
            f"No se encontraron CSV en '{DIRECTORIO_DATASET}/'. "
        )

    print(f"\n{'='*60}")
    print("CARGANDO DATASETS")
    print(f"{'='*60}")

    for archivo in sorted(archivos_csv):
        ruta = os.path.join(DIRECTORIO_DATASET, archivo)
        df   = pd.read_csv(ruta)

        num_muestras_csv = df["Num_Muestra"].max()
        gesto            = df["Gesto"].iloc[0]

        print(f"{archivo}  →  {num_muestras_csv} muestras  (gesto: '{gesto}')")

        # Reconstruir cada muestra como tensor (20, 8)
        for num in df["Num_Muestra"].unique():
            filas = df[df["Num_Muestra"] == num][columnas_features].values
            if filas.shape[0] != FRAMES_POR_MUESTRA:
                print(f"Muestra {num} tiene {filas.shape[0]} frames, se descarta.")
                continue
            muestras.append(filas)
            etiquetas.append(gesto)

    X = np.array(muestras, dtype=np.float32)
    y = np.array(etiquetas)

    print(f"\n  Total muestras cargadas : {len(X)}")
    print(f"  Shape del tensor X      : {X.shape}  → (muestras, frames, features)")
    print(f"  Clases detectadas       : {sorted(set(y))}")

    return X, y


def preparar_datos(X, y):
    # Normalización + division entrenamiento + validacion    
    X_flat  = X.reshape(-1, FEATURES)
    minimos = X_flat.min(axis=0)
    maximos = X_flat.max(axis=0)
    rango   = maximos - minimos
    rango[rango == 0] = 1 # Si dato es 0 constante, se dividie por 1 en vez de por 0

    X_norm = (X - minimos) / rango

    np.save("normalizacion_min.npy", minimos)
    np.save("normalizacion_max.npy", maximos)
    print("\nParametros de normalizacion guardados:")
    print(f"normalizacion_min.npy")
    print(f"normalizacion_max.npy")

    encoder = LabelEncoder()
    y_int   = encoder.fit_transform(y)
    y_cat   = tf.keras.utils.to_categorical(y_int)

    clases = list(encoder.classes_)
    print(f"\nMapeo de clases (índice → gesto):")
    for i, clase in enumerate(clases):
        voz = ETIQUETAS_VOZ.get(clase, clase)
        print(f"     [{i}] {clase}  →  '{voz}'")
    
    np.save("clases.npy", np.array(clases))
    print(f"     clases.npy guardado")

    # Division dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y_cat, test_size=0.2, random_state=42, stratify=y_int
    )

    print(f"\nMuestras entrenamiento : {len(X_train)}")
    print(f"Muestras validación    : {len(X_val)}")

    return X_train, X_val, y_train, y_val, clases


# =============================================================================
# DISEÑO DEL MODELO
# =============================================================================

def construir_modelo(num_clases):
    entrada = tf.keras.Input(shape=(FRAMES_POR_MUESTRA, FEATURES))

    x = tf.keras.layers.Flatten()(entrada)

    x = Dense(128, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(DROPOUT_RATE / 2)(x)
    x = Dense(32, activation='relu')(x)
    salida = Dense(num_clases, activation='softmax')(x)

    modelo = tf.keras.Model(inputs=entrada, outputs=salida)

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def entrenar(modelo, X_train, X_val, y_train, y_val):
    print(f"\n{'='*60}")
    print("ENTRENANDO EL MODELO")
    print(f"{'='*60}")
    modelo.summary()
    print()

    # EarlyStopping: para automáticamente si la validaciOn deja de mejorar
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=PATIENCE,
        restore_best_weights=True,   # Recupera los pesos de la mejor iteracion de la red
        verbose=1
    )

    historial = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    return historial


# =============================================================================
# EVALUACIÓN
# =============================================================================

def evaluar(modelo, X_val, y_val, clases):
    print(f"\n{'='*60}")
    print("EVALUACIÓN FINAL")
    print(f"{'='*60}")

    loss, acc = modelo.evaluate(X_val, y_val, verbose=0)
    print(f"\nAccuracy en validación : {acc*100:.1f}%")
    print(f"Loss en validación     : {loss:.4f}")

    y_pred = np.argmax(modelo.predict(X_val, verbose=0), axis=1)
    y_true = np.argmax(y_val, axis=1)

    print(f"\nReporte por gesto:\n")
    print(classification_report(y_true, y_pred, target_names=clases))

    print(f"Matriz de confusión (filas=real, columnas=predicho):")
    print(f"Clases: {clases}")
    print(confusion_matrix(y_true, y_pred))


# =============================================================================
# EXPORTAR A TFLITE
# =============================================================================

def exportar_tflite(modelo):
    print(f"\n{'='*60}")
    print("EXPORTANDO A TFLITE")
    print(f"{'='*60}")

    conversor = tf.lite.TFLiteConverter.from_keras_model(modelo)

    conversor.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = conversor.convert()

    with open(RUTA_MODELO_TFLITE, "wb") as f:
        f.write(tflite_model)

    tamanio_kb = os.path.getsize(RUTA_MODELO_TFLITE) / 1024
    print(f"\nModelo exportado: {RUTA_MODELO_TFLITE}")
    print(f"Tamaño           : {tamanio_kb:.1f} KB")

# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
============================================================
    PROYECTO SIGNUM — ENTRENAMIENTO DEL MODELO
============================================================
    TensorFlow : {}
    GPU         : {}
============================================================
""".format(
        tf.__version__,
        "Disponible" if tf.config.list_physical_devices('GPU') else "No disponible (CPU, normal en PC sin GPU)"
    ))

    # Cargar datos
    X, y = cargar_datasets()

    # Preparar (normalizar, codificar, split)
    X_train, X_val, y_train, y_val, clases = preparar_datos(X, y)

    # Construir modelo
    modelo = construir_modelo(num_clases=len(clases))

    # Entrenar
    entrenar(modelo, X_train, X_val, y_train, y_val)

    # Evaluar
    evaluar(modelo, X_val, y_val, clases)

    # Exportar a TFLite
    exportar_tflite(modelo)

    print(f"\n{'='*60}")
    print("Proceso completado.")
    print(f"{'='*60}\n")