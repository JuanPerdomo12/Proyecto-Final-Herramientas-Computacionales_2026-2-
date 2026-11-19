import os
import re
import matplotlib.pyplot as plt
import numpy as np


def leer_acqus(ruta_carpeta):
    ruta_acqus = os.path.join(ruta_carpeta, "acqus")
    params = {}

    with open(ruta_acqus, "r", encoding="latin-1") as f:
        contenido = f.read()

    claves = [
        "TD",
        "SW_h",
        "SW",
        "SFO1",
        "BF1",
        "O1",
        "NC",
        "GRPDLY",
        "BYTORDA",
    ]
    for clave in claves:
        match = re.search(
            rf"##\${clave}=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", contenido
        )
        if match:
            valor = float(match.group(1))
            params[clave] = (
                int(valor) if clave in ["TD", "NC", "BYTORDA"] else valor
            )

    return params

ruta_exp = (
    r"C:\Users\jujo1\OneDrive\Documentos\Proyecto-Final-Herramientas-Computacionales_2026-2-"
    r"\Tomate_raw_data\nmrTom290114\C-0-1\2"
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
grpdly = int(round(grpdly_raw)) if 0 < grpdly_raw < 500 else 68

raw_data = np.fromfile(os.path.join(ruta_exp, "fid"), dtype=endian)
raw_data = raw_data * (2**NC)

fid = raw_data[0::2] - 1j * raw_data[1::2]

fid = np.roll(fid, -grpdly)
fid[-grpdly:] = 0

dw = 1.0 / SW_h
tiempo = np.arange(len(fid)) * dw

espectro = np.fft.fftshift(np.fft.fft(fid, n=65536))

centro_ppm = O1 / BF1
eje_ppm = np.linspace(
    centro_ppm + SW_ppm / 2, centro_ppm - SW_ppm / 2, len(espectro)
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(tiempo, np.real(fid), color="darkblue", linewidth=0.8)
ax1.set_title(
    f"Señal FID en el dominio del tiempo"
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

def leer_procs(ruta_pdata):
    ruta_procs = os.path.join(ruta_pdata, "procs")
    params = {}
    with open(ruta_procs, "r", encoding="latin-1") as f:
        contenido = f.read()

    claves = ["SI", "SW_p", "SF", "OFFSET", "NC_proc", "BYTORDP"]
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

ruta_pdata = (
    r"C:\Users\jujo1\OneDrive\Documentos\Proyecto-Final-Herramientas-Computacionales_2026-2-"
    r"\Tomate_raw_data\nmrTom290114\C-0-1\2\pdata\1"
)

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