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

############################################################
"""
# =============================================================================
# CARGA DE LIBRERIAS
# =============================================================================

import sys
import os
import time
import numpy as np
from mpu6050 import mpu6050
from gpiozero import MCP3008

try:
    import smbus2 as smbus
    sys.modules['smbus'] = smbus
except ImportError:
    print("Error: No se pudo cargar smbus2.")
    sys.exit(1)

# =============================================================================
#  CONFIGURACIÓN
# =============================================================================

FRAMES_POR_MUESTRA = 20
FEATURES           = 8
PERIODO_CAPTURA    = 0.10
VENTANA_ACTIVACION = 2.0
UMBRAL_CONFIANZA   = 0.80
GESTO_ACTIVACION   = "captura"
GESTO_RECHAZO      = "ninguno"

RUTA_PESOS  = "pesos_signum.npz"
RUTA_MIN    = "normalizacion_min.npy"
RUTA_MAX    = "normalizacion_max.npy"
RUTA_CLASES = "clases.npy"

ETIQUETAS_VOZ = {
    "buenos_dias"  : "Buenos días",
    "gracias"      : "Gracias",
    "cuanto_cuesta": "¿Cuánto cuesta?",
}

nombres_dedos = ["Meñique", "Pulgar ", "Índice ", "Corazón", "Anular "]
CALIBRACION = {
    "Meñique": {"plano": 0.449, "doblado": 0.140},
    "Pulgar ": {"plano": 0.333, "doblado": 0.164},
    "Índice ": {"plano": 0.389, "doblado": 0.157},
    "Corazón": {"plano": 0.387, "doblado": 0.157},
    "Anular ": {"plano": 0.457, "doblado": 0.240},
}

# =============================================================================
#  HARDWARE
# =============================================================================

canales_flex = [MCP3008(channel=i) for i in range(5)]

try:
    sensorAcelerometro = mpu6050(0x68)
    print("MPU6050 detectado")
except Exception as e:
    sensorAcelerometro = None
    print(f"MPU6050 no disponible ({e})")

# =============================================================================
#  MODELO USANDO NUMPY
# =============================================================================

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def forward(pesos, x):
    """
    Forward pass MLP con numpy puro.
    x: array aplanado (160,)
    Arquitectura: Dense(128,relu) → Dense(64,relu) → Dense(32,relu) → Dense(4,softmax)
    """
    x = relu(pesos['W1'] @ x + pesos['b1'])
    x = relu(pesos['W2'] @ x + pesos['b2'])
    x = relu(pesos['W3'] @ x + pesos['b3'])
    x = softmax(pesos['W4'] @ x + pesos['b4'])
    return x

def cargar_modelo():
    for ruta in [RUTA_PESOS, RUTA_MIN, RUTA_MAX, RUTA_CLASES]:
        if not os.path.exists(ruta):
            print(f"Error: No se encuentra '{ruta}'.")
            sys.exit(1)

    npz     = np.load(RUTA_PESOS)
    pesos   = {k: npz[k] for k in npz.files}
    minimos = np.load(RUTA_MIN)
    maximos = np.load(RUTA_MAX)
    clases  = np.load(RUTA_CLASES)

    print(f"Pesos cargados: {RUTA_PESOS}")
    print(f"Clases: {list(clases)}")

    return pesos, minimos, maximos, clases

# =============================================================================
#  SENSORES
# =============================================================================

def obtener_flexion(valor_actual, nombre_dedo):
    conf = CALIBRACION[nombre_dedo]
    if valor_actual >= conf["plano"]:
        return 0
    
    if valor_actual <= conf["doblado"]:
        return 100
    
    try:
        return int((valor_actual - conf["plano"]) * 100 / (conf["doblado"] - conf["plano"]))
    except ZeroDivisionError:
        return 0

def leer_frame():
    dedos = [obtener_flexion(canales_flex[i].value, nombres_dedos[i]) for i in range(5)]
    if sensorAcelerometro:
        a = sensorAcelerometro.get_accel_data()
        ax, ay, az = round(a['x'], 4), round(a['y'], 4), round(a['z'], 4)
    else:
        ax, ay, az = 0.0, 0.0, 0.0
    return dedos + [ax, ay, az]


# =============================================================================
#  NORMALIZACION E PREDICCION DEL GESTO
# =============================================================================

def normalizar(ventana, minimos, maximos):
    rango = maximos - minimos
    rango[rango == 0] = 1
    return (ventana - minimos) / rango

def predecir(pesos, ventana_norm, clases):
    x              = ventana_norm.flatten().astype(np.float32)
    probabilidades = forward(pesos, x)
    idx_max        = int(np.argmax(probabilidades))
    confianza      = float(probabilidades[idx_max])
    gesto          = str(clases[idx_max])

    return gesto, confianza, probabilidades


# =============================================================================
#  VOZ
# =============================================================================

def voz(texto):
    print(f"{texto}")
    os.system(f"espeak -ves -s 140 -p 50 -a 200 '{texto}' 2>/dev/null")

# =============================================================================
#  BUCLES DE DETECCION
# =============================================================================

def monitorizar_activacion(pesos, minimos, maximos, clases):
    # Bucle de espera activa del gesto de activacion
    buffer = []
    while True:
        t0    = time.monotonic()
        buffer.append(leer_frame())
        if len(buffer) > FRAMES_POR_MUESTRA:
            buffer.pop(0)
        if len(buffer) == FRAMES_POR_MUESTRA:
            ventana_norm    = normalizar(np.array(buffer, dtype=np.float32), minimos, maximos)
            gesto, confianza, _ = predecir(pesos, ventana_norm, clases)
            if gesto == GESTO_ACTIVACION and confianza >= UMBRAL_CONFIANZA:
                return True
        elapsed = time.monotonic() - t0
        restante = PERIODO_CAPTURA - elapsed
        if restante > 0:
            time.sleep(restante)

def esperar_gesto(pesos, minimos, maximos, clases):
    # Tras activacion se captura el mejor gesto en VENTANA_ACTIVACION segundos
    t_inicio        = time.monotonic()
    buffer          = []
    mejor_gesto     = None
    mejor_confianza = 0.0
    mejor_probs     = None

    while (time.monotonic() - t_inicio) < VENTANA_ACTIVACION:
        t0    = time.monotonic()
        buffer.append(leer_frame())
        if len(buffer) > FRAMES_POR_MUESTRA:
            buffer.pop(0)
        if len(buffer) == FRAMES_POR_MUESTRA:
            ventana_norm        = normalizar(np.array(buffer, dtype=np.float32), minimos, maximos)
            gesto, confianza, probs = predecir(pesos, ventana_norm, clases)
            # Ignoramos activación y rechazo solo nos interesan gestos traducibles
            if gesto not in (GESTO_ACTIVACION, GESTO_RECHAZO) and confianza > mejor_confianza:
                mejor_gesto     = gesto
                mejor_confianza = confianza
                mejor_probs     = probs
            restante_s = VENTANA_ACTIVACION - (time.monotonic() - t_inicio)
            print(f"\r{restante_s:.1f}s | {gesto} ({confianza*100:.0f}%)   ", end="", flush=True)
        elapsed  = time.monotonic() - t0
        sleep_t  = PERIODO_CAPTURA - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    print()
    return mejor_gesto, mejor_confianza, mejor_probs


# =============================================================================
#  MAIN
# =============================================================================

def ejecutar():
    print("\n" + "="*60)
    print("TRADUCTOR SIGNUM")
    print("="*60)

    pesos, minimos, maximos, clases = cargar_modelo()

    print(f"\nUmbral de confianza : {UMBRAL_CONFIANZA*100:.0f}%")
    print(f"Ventana activación  : {VENTANA_ACTIVACION} s")
    print(f"Gesto activador     : '{GESTO_ACTIVACION}'")
    print(f"Gesto de rechazo    : '{GESTO_RECHAZO}'")
    print(f"\n{'='*60}")
    print("Cargado correctamente. Haz el gesto de activacion para empezar a detectar")
    print(f"{'='*60}\n")

    voz("Traductor funcionando")

    try:
        while True:
            print("Esperando activación...")
            monitorizar_activacion(pesos, minimos, maximos, clases)

            print("\nActivación detectada. Haz tu gesto ahora.")
            voz("OK")

            gesto, confianza, probs = esperar_gesto(pesos, minimos, maximos, clases)

            if gesto is None or confianza < UMBRAL_CONFIANZA:
                print(f"No se detectó ningún gesto con suficiente confianza "
                      f"(confianza={confianza*100:.1f}%).")
                voz("No entendí el gesto")
            else:
                texto_voz = ETIQUETAS_VOZ.get(gesto, gesto)
                print(f"\nGesto: '{gesto}'  |  Confianza: {confianza*100:.1f}%")
                print(f"Probabilidades: { {str(clases[i]): f'{probs[i]*100:.1f}%' for i in range(len(clases))} }")
                voz(texto_voz)

            print("\n" + "-"*60)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nSistema Signum detenido.")

if __name__ == "__main__":
    ejecutar()