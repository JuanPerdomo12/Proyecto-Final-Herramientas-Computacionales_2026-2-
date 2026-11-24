import numpy as np
import matplotlib.pyplot as plt
import os
import re

#Se leen los parametros del archivo acqus que contiene toda la información de cómo la maquina de resonancia tomo los datos. Este es un formato conocido como Bruker. 
#Aqui tuvimos ayuda de la IA para entender cada parametro y su significado
def leer_acqus(ruta_carpeta):
    ruta_acqus = os.path.join(ruta_carpeta, "acqus")
    params = {}

    with open(ruta_acqus, "r", encoding="latin-1") as f:
        contenido = f.read()

    claves = [
        "TD", #Número de puntos en el dominio del tiempo
        "SW_h", #Ancho espectral en Hz
        "SW", #Ancho espectral en ppm
        "SFO1", #Frecuencia de resonancia
        "BF1", #Frecuencia base del espectrometro
        "O1", #Frecuencia de
        "NC", #Factor de escala, se usa para evitar overflow
        "GRPDLY", #Retardo de grupo, usado para corregir el desfase general de la señal
        "BYTORDA", #Orden de los bytes (Little-endian o Big-endian)
    ]
    for clave in claves:
        match = re.search(
            rf"##\${clave}=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", contenido #Esta expresión busca el valor de cada parametro dentro del archivo dado que cada uno inicia con ##$ seguido del nombre y un igual 
        )
        if match:
            valor = float(match.group(1))
            params[clave] = (
                int(valor) if clave in ["TD", "NC", "BYTORDA"] else valor
            )

    return params

ruta_proyecto = os.path.dirname(os.path.abspath(__file__))
ruta_exp = os.path.join(
    ruta_proyecto, "Tomate_raw_data", "nmrTom290114", "C-0-1", "1" #Aqui se selecciona el corte (C-#-#) y el tipo de adquisicion (1: Calibracion, 2: Experimento filtrado)
)

p = leer_acqus(ruta_exp)

TD = p["TD"]
SW_h = p["SW_h"]
SW_ppm = p["SW"]
BF1 = p["BF1"]
O1 = p["O1"]
NC = p.get("NC", -2)

endian = ">i4" if p.get("BYTORDA", 1) == 1 else "<i4"

grpdly_raw = p.get("GRPDLY", 68)
grpdly = int(round(grpdly_raw)) if 0 < grpdly_raw < 500 else 68 #Aqui se redondea en caso de que no sea entero y en caso de que sea un numero fuera de rango

raw_data = np.fromfile(os.path.join(ruta_exp, "fid"), dtype=endian)
raw_data = raw_data * (2**NC) #Aqui se aplica el factor de escala para restaurar la escala de intensidad real

fid = raw_data[0::2] - 1j * raw_data[1::2] #Aqui se indexan los datos para separar la parte real e imaginaria (reales son los pares, imaginarios son los impares)

fid = np.roll(fid, -grpdly) #Aqui se eliminan los puntos iniciales y se desplaza la señal a la izquierda para corregir el desfase general
fid[-grpdly:] = 0

dw = 1.0 / SW_h #Esto es el intervalo de tiempo entre toma de puntos
tiempo = np.arange(len(fid)) * dw

espectro = np.fft.fftshift(np.fft.fft(fid, n=65536)) #Se realiza fourier para pasar al dominio de frecuencias (espacio-k)

centro_ppm = O1 / BF1 #Calcula el centro del espectro en ppm
eje_ppm = np.linspace(
    centro_ppm + SW_ppm / 2, centro_ppm - SW_ppm / 2, len(espectro) #Aqui se genera el eje de desplazamiento quimico en ppm
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(tiempo, np.real(fid), color="darkblue", linewidth=0.8)
ax1.set_title(
    f"Señal FID (Dominio del tiempo)"
)
ax1.set_xlabel("Tiempo (s)")
ax1.set_ylabel("Intensidad (u.a.)")
ax1.grid(True, alpha=0.3)

ax2.plot(eje_ppm, np.abs(espectro), color="crimson", linewidth=0.9)
ax2.set_title("Espectro de RMN de $^1$H - Tomate (Magnitud)")
ax2.set_xlabel("Desplazamiento químico (ppm)")
ax2.set_ylabel("Intensidad (u.a.)")
ax2.set_xlim(10, 0)
ax2.set_ylim(0, 0.1e9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#Se leen los parametros del archivo procs que contiene toda la información de cómo la maquina de resonancia tomo los datos. Este es un formato conocido como Bruker. 
def leer_procs(ruta_pdata):
    ruta_procs = os.path.join(ruta_pdata, "procs")
    params = {}
    with open(ruta_procs, "r", encoding="latin-1") as f:
        contenido = f.read()

    claves = [
        "SI",
        "SW_p", 
        "SF", 
        "OFFSET", 
        "NC_proc", 
        "BYTORDP"
    ]
    for clave in claves:
        match = re.search(
            rf"##\${clave}=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", contenido
        )
        if match:
            valor = float(match.group(1))
            params[clave] = (
                int(valor) if clave in ["SI", "NC_proc", "BYTORDP"] else valor
            )
    return params

ruta_pdata = os.path.join(ruta_exp, "pdata", "1")

pp = leer_procs(ruta_pdata)

SI = pp["SI"]
OFFSET = pp["OFFSET"]
SW_ppm = pp["SW_p"] / pp["SF"]
NC_proc = pp.get("NC_proc", 0)
endian_p = ">i4" if pp.get("BYTORDP", 1) == 1 else "<i4"

spec_1r = np.fromfile(os.path.join(ruta_pdata, "1r"), dtype=endian_p) * (
    2**NC_proc
)
spec_1i = np.fromfile(os.path.join(ruta_pdata, "1i"), dtype=endian_p) * (
    2**NC_proc
)

eje_ppm_pdata = np.linspace(OFFSET, OFFSET - SW_ppm, SI)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(eje_ppm_pdata, spec_1r, color="darkgreen", linewidth=0.9)
ax1.set_title("Espectro Procesado Oficial - Parte Real (1r)")
ax1.set_ylabel("Intensidad (u.a.)")
ax1.set_xlim(10, 0)
ax1.set_ylim(-0.05e8, 0.6e8)
ax1.grid(True, alpha=0.3)

ax2.plot(eje_ppm_pdata, spec_1i, color="purple", linewidth=0.9)
ax2.set_title("Espectro Procesado Oficial - Parte Imaginaria (1i)")
ax2.set_xlabel("Desplazamiento químico (ppm)")
ax2.set_ylabel("Intensidad (u.a.)")
ax2.set_ylim(-0.3e8, 0.3e8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
