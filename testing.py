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

# Carga librerias
import sys
import os
import time
from mpu6050 import mpu6050
from gpiozero import MCP3008

try:
    import smbus2 as smbus
    sys.modules['smbus'] = smbus
except ImportError:
    print("Error: No se pudo cargar smbus2. Asegúrate de estar en el (env).")

# Deteccion MPU6050
try:
    sensorAcelerometro = mpu6050(0x68)
except:
    sensorAcelerometro = None

# Configuración Flex (4 Largos + 1 Corto en canal 4)
canales_flex = [MCP3008(channel=i) for i in range(5)]
nombres_dedos = ["Meñique", "Pulgar ", "Índice ", "Corazón", "Anular "]

# CALIBRACION SENSORES FLEX
CALIBRACION = {
    "Meñique": {"plano": 0.449, "doblado": 0.140},
    "Pulgar ": {"plano": 0.333, "doblado": 0.164},
    "Índice ": {"plano": 0.398, "doblado": 0.157},
    "Corazón": {"plano": 0.471, "doblado": 0.251},
    "Anular ": {"plano": 0.457, "doblado": 0.238}
}

# Funcion para usar el eSpeak y reproducir el audio por el altavoz
def voz(texto):
    print(f"Hablando: {texto}")    
    os.system(f"espeak -ves -s 140 -p 50 -a 200 '{texto}' 2>/dev/null")

# Funcion para detectar la flexion de los dedos
def obtener_flexion(valor_actual, nombre_dedo):
    conf = CALIBRACION[nombre_dedo]
    v_plano = conf["plano"]
    v_doblado = conf["doblado"]
    
    if valor_actual >= v_plano:
        return 0
    
    if valor_actual <= v_doblado:
        return 100
    
    try:
        porcentaje = (valor_actual - v_plano) * 100 / (v_doblado - v_plano)
        return int(porcentaje)
    except ZeroDivisionError:
        return 0

def ejecutar_lecturas():
    print("\n" + "="*80)
    print("LECTURAS SENSORES")
    print("="*80)
    header = " | ".join(nombres_dedos) + " |  ACEL_X  |  ACEL_Y  |  ACEL_Z  |"
    print(header)
    print("-" * len(header))
    
    voz("Buenos días") 
    
    print("\n" + "="*80)

    try:
        while True:
            lecturas_f = []
            for i, canal in enumerate(canales_flex):
                f = obtener_flexion(canal.value, nombres_dedos[i])
                lecturas_f.append(f"{f:3}%")
            
            if sensorAcelerometro:
                a = sensorAcelerometro.get_accel_data()
                g = sensorAcelerometro.get_gyro_data()
                
                eje_x_accel = round(a['x'], 2)
                eje_y_accel = round(a['y'], 2)
                eje_z_accel = round(a['z'], 2)
                
                eje_x_gir = round(g['x'], 2)
                eje_y_gir = round(g['y'], 2)
                eje_z_gir = round(g['z'], 2)

                accel_str = f" {eje_x_accel} | {eje_y_accel} | {eje_z_accel} |"
                gir_str = f" {eje_x_gir} | {eje_y_gir} | {eje_z_gir} |"
            else:
                accel_str = "N/A | N/A | N/A |"
                gir_str = "N/A | N/A | N/A |"

            # Imprimimos y volvemos al inicio de la línea
            print("  |  ".join(lecturas_f) + "  |" + accel_str + " |" + gir_str, end='\r')
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nMonitor finalizado.")

if __name__ == "__main__":
    ejecutar_lecturas()