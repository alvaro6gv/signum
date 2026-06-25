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

import sys
import os
import csv
import time
import tty
import termios
import select
from mpu6050 import mpu6050
from gpiozero import MCP3008

try:
    import smbus2 as smbus
    sys.modules['smbus'] = smbus
except ImportError:
    print("Error: No se pudo cargar smbus2. Asegúrate de estar en el (env).")
    sys.exit(1)

# =============================================================================
#  PARAMETROS CAPTURA
# =============================================================================
FRAMES_POR_MUESTRA = 20       # Numero de lecturas por repeticion
PERIODO_CAPTURA    = 0.10     # Segundos entre frames
DIRECTORIO_DATASET = "dataset" # Carpeta donde se guardan los CSV


# =============================================================================
#  CALIBRACION
# =============================================================================
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
    print("MPU6050 detectado en 0x68")
except Exception as e:
    sensorAcelerometro = None
    print(f"MPU6050 no disponible ({e}). Los ejes del acelerometro se guardaran como 0.0")


# =============================================================================
#  FUNCIONES AUXILIARES
# =============================================================================

def obtener_flexion(valor_actual, nombre_dedo):
    conf      = CALIBRACION[nombre_dedo]
    v_plano   = conf["plano"]
    v_doblado = conf["doblado"]

    if valor_actual >= v_plano:
        return 0
    if valor_actual <= v_doblado:
        return 100
    try:
        return int((valor_actual - v_plano) * 100 / (v_doblado - v_plano))
    except ZeroDivisionError:
        return 0


def leer_frame():
    dedos = [
        obtener_flexion(canales_flex[i].value, nombres_dedos[i])
        for i in range(5)
    ]

    if sensorAcelerometro:
        a  = sensorAcelerometro.get_accel_data()
        ax, ay, az = round(a['x'], 4), round(a['y'], 4), round(a['z'], 4)
    else:
        ax, ay, az = 0.0, 0.0, 0.0

    return dedos + [ax, ay, az]


def esperar_espacio():
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                ch = sys.stdin.read(1)
                if ch == ' ':
                    return
                # Ctrl+C / Ctrl+D → salida limpia
                if ch in ('\x03', '\x04'):
                    raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def ruta_csv(nombre_gesto):
    os.makedirs(DIRECTORIO_DATASET, exist_ok=True)
    # Nombre de archivo seguro: espacios → guiones bajos, todo minúsculas
    nombre_limpio = nombre_gesto.strip().lower().replace(" ", "_")
    return os.path.join(DIRECTORIO_DATASET, f"dataset_{nombre_limpio}.csv")


CABECERA_CSV = [
    "Num_Muestra",
    "Meñique_%", "Pulgar_%", "Índice_%", "Corazón_%", "Anular_%",
    "Acel_X", "Acel_Y", "Acel_Z",
    "Gesto",
]


def guardar_muestra(ruta, num_muestra, frames, etiqueta):
    archivo_nuevo = not os.path.exists(ruta)

    with open(ruta, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if archivo_nuevo:
            writer.writerow(CABECERA_CSV)

        for frame in frames:
            # frame = [m, p, i, c, a, ax, ay, az]
            writer.writerow([num_muestra] + frame + [etiqueta])


# =============================================================================
#  BUCLE PRINCIPAL DE CAPTURA
# =============================================================================

def limpiar_muestras_incompletas(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        cabecera = next(reader, None)
        if cabecera is None:
            return 0

        grupos = {}
        for fila in reader:
            if not fila:
                continue
            try:
                num = int(fila[0])
            except (ValueError, IndexError):
                continue
            grupos.setdefault(num, []).append(fila)

    completas = {k: v for k, v in grupos.items() if len(v) == FRAMES_POR_MUESTRA}
    incompletas = len(grupos) - len(completas)

    if incompletas > 0:
        print(f"\nSe encontraron {incompletas} muestra(s) incompleta(s), se eliminaran automáticamente.")

    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cabecera)
        for nuevo_num, (_, filas) in enumerate(sorted(completas.items()), start=1):
            for fila in filas:
                fila[0] = str(nuevo_num)
                writer.writerow(fila)

    return len(completas)


def ejecutar_captura():
    print("\n" + "=" * 60)
    print("PROYECTO SIGNUM — CAPTURA DE GESTOS")
    print("=" * 60)
    print(f"Frames por muestra : {FRAMES_POR_MUESTRA}")
    print(f"Frecuencia         : ({FRAMES_POR_MUESTRA * PERIODO_CAPTURA:.1f} s por muestra)")
    print(f"Directorio de salida: ./{DIRECTORIO_DATASET}/")
    print("=" * 60)

    nombre_gesto = input("\nNombre del gesto a capturar: ").strip()
    if not nombre_gesto:
        print("Error: el nombre del gesto no puede estar vacío.")
        sys.exit(1)

    ruta = ruta_csv(nombre_gesto)

    num_muestra_inicial = 1
    if os.path.exists(ruta):
        muestras_completas = limpiar_muestras_incompletas(ruta)
        print(f"\nArchivo encontrado: {ruta}")
        print(f"Muestras completas: {muestras_completas}")
        print(f"\nQue quieres hacer?")
        print(f"    [C]  Continuar desde la muestra #{muestras_completas + 1}")
        print(f"    [R]  Resetear y empezar desde cero (borra el CSV)")

        while True:
            opcion = input("\n  Tu elección (C/R): ").strip().upper()
            if opcion == "C":
                num_muestra_inicial = muestras_completas + 1
                print(f"Continuando desde la muestra #{num_muestra_inicial}")
                break
            elif opcion == "R":
                os.remove(ruta)
                num_muestra_inicial = 1
                print(f"CSV eliminado. Empezando desde la muestra #1")
                break
            else:
                print("Escribe C para continuar o R para resetear.")

    print(f"\n  Gesto     : '{nombre_gesto}'")
    print(f"  Archivo   : {ruta}")
    print(f"\n  Controles :")
    print(f"    [ESPACIO]  →  Capturar siguiente muestra")
    print(f"    [Ctrl+C]   →  Finalizar y salir")
    print("\n" + "-" * 60)

    num_muestra = num_muestra_inicial

    try:
        while True:
            print(f"\nColocate y pulsa [BARRA ESPACIADORA] para registrar "
                  f"la muestra #{num_muestra}...", end="", flush=True)

            esperar_espacio()

            for cuenta in ("3", "2", "1", "YA"):
                print(f"\r  {cuenta}  ", end="", flush=True)
                time.sleep(0.25)

            print(f"\rGrabando muestra #{num_muestra}...      ", flush=True)

            # ── Captura de los N frames ───────────────────────────────────────
            frames = []
            for f_idx in range(FRAMES_POR_MUESTRA):
                t0    = time.monotonic()
                frame = leer_frame()
                frames.append(frame)

                # Mostrar progreso en la misma línea
                barra = "█" * (f_idx + 1) + "░" * (FRAMES_POR_MUESTRA - f_idx - 1)
                print(f"\r  [{barra}] frame {f_idx+1:02d}/{FRAMES_POR_MUESTRA}", end="", flush=True)

                # Compensar el tiempo de cómputo para mantener la frecuencia
                elapsed = time.monotonic() - t0
                restante = PERIODO_CAPTURA - elapsed
                if restante > 0:
                    time.sleep(restante)
        
            guardar_muestra(ruta, num_muestra, frames, nombre_gesto)

            print(f"\rMuestra #{num_muestra} guardada ({FRAMES_POR_MUESTRA} frames → {ruta})")

            num_muestra += 1

    except KeyboardInterrupt:
        total = num_muestra - num_muestra_inicial
        print(f"\n\n{'='*60}")
        print(f"  Sesión finalizada.")
        print(f"  Muestras capturadas en esta sesión : {total}")
        print(f"  Archivo de salida                  : {ruta}")
        print(f"{'='*60}\n")


# =============================================================================
if __name__ == "__main__":
    ejecutar_captura()