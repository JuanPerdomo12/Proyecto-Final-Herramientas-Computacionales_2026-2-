import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pydicom
from scipy.optimize import curve_fit
import glob

#Se leen los parametros del archivo acqus que contiene toda la información de cómo la maquina de resonancia tomo los datos. Este es un formato conocido como Bruker. 
#Aqui tuvimos ayuda de la IA para entender cada parametro, su significado y algunas formulas de corrección de los datos
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
        "O1", #Frecuencia de offset
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

#Se leen los parametros del archivo procs que contiene toda la información de cómo el software del espectrometro (TopSpin) procesó la señal original para obtener el espectro final
def leer_procs(ruta_pdata):
    ruta_procs = os.path.join(ruta_pdata, "procs")
    params = {}
    with open(ruta_procs, "r", encoding="latin-1") as f:
        contenido = f.read()

    claves = [
        "SI", #Número de puntos en frecuencia
        "SW_p", #Ancho espectral en Hz
        "SF", #Frecuencia del espectrometro
        "OFFSET", #Valor máximo en ppm desde donde empieza el eje horizontal del espectro
        "NC_proc", #Factor de escala
        "BYTORDP" #Orden de los bytes (Little-endian o Big-endian)
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

ruta_pdata = os.path.join(ruta_exp, "pdata", "1") #Aqui se entra a la subcarpeta donde Bruker guarda los espectros ya procesados
pp = leer_procs(ruta_pdata)

SI = pp["SI"]
OFFSET = pp["OFFSET"]
SW_ppm_proc = pp["SW_p"] / pp["SF"] #Calcula el ancho espectral en ppm
NC_proc = pp.get("NC_proc", 0)
endian_p = ">i4" if pp.get("BYTORDP", 1) == 1 else "<i4"

#Se leen los archivos binarios 1r (parte real) y 1i (parte imaginaria) y se escalan con NC_proc para recuperar su magnitud real
spec_1r = np.fromfile(os.path.join(ruta_pdata, "1r"), dtype=endian_p) * (2**NC_proc)
spec_1i = np.fromfile(os.path.join(ruta_pdata, "1i"), dtype=endian_p) * (2**NC_proc)
magnitud_oficial = np.hypot(spec_1r, spec_1i)

#Aqui se construye el eje en ppm empezando desde el OFFSET
eje_ppm_pdata = np.linspace(OFFSET, OFFSET - SW_ppm_proc, SI)

#Primera gráfica: FID + Comparación entre espectro manual y oficial
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(tiempo, np.real(fid), color="darkblue", linewidth=0.8)
ax1.set_title("Señal FID (Dominio del tiempo)")
ax1.set_xlabel("Tiempo (s)")
ax1.set_ylabel("Intensidad (u.a.)")
ax1.grid(True, alpha=0.3)

#Aqui superponemos nuestro espectro calculado con el espectro oficial para verificar que los picos coincidan en frecuencia
ax2.plot(eje_ppm, np.abs(espectro), color="crimson", linewidth=0.9, label="Manual")
ax2.plot(eje_ppm_pdata, magnitud_oficial, color="darkgreen", linewidth=0.9, alpha=0.8, label="Oficial")
ax2.set_title("Comparación Espectro de RMN de $^1$H - Tomate")
ax2.set_xlabel("Desplazamiento químico (ppm)")
ax2.set_ylabel("Intensidad (u.a.)")
ax2.set_xlim(10, 0)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("FID_CompareMagnitudes.png")
plt.show()

#Segunda gráfica: Detalle de las componentes oficial real (1r) e imaginaria (1i)
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax3.plot(eje_ppm_pdata, spec_1r, color="darkgreen", linewidth=0.9)
ax3.set_title("Espectro Procesado Oficial - Parte Real (1r)")
ax3.set_ylabel("Intensidad (u.a.)")
ax3.set_xlim(10, 0)
ax3.grid(True, alpha=0.3)

ax4.plot(eje_ppm_pdata, spec_1i, color="purple", linewidth=0.9)
ax4.set_title("Espectro Procesado Oficial - Parte Imaginaria (1i)")
ax4.set_xlabel("Desplazamiento químico (ppm)")
ax4.set_ylabel("Intensidad (u.a.)")
ax4.set_ylim(-0.3e8, 0.3e8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Oficial_Components.png")
plt.show()

ruta_img = os.path.join(
    ruta_proyecto, "Tomate_corte_tiempos" #Aqui se selecciona la carpeta con todos los archivos de las imagenes del Tomate en formato DICOM
)
archivos = sorted(glob.glob(os.path.join(ruta_img, "PvMRIm*"))) #Aqui se guardan y ordenan todos los archivos DICOM

volumen_imgs = []
tiempos_eco = []

#Se recorre cada archivo DICOM para extraer su matriz de pixeles y el tiempo de eco (TE)
#Aqui tambien usamos IA para entender cada parametro de DICOM y como convertir los valores a algo real y fisico
for arch in archivos:
    ds = pydicom.dcmread(arch) #Lee todos los datos de imagen del archivo DICOM
    slope = float(getattr(ds, "RescaleSlope", 1.0)) #Factor de escala de DICOM
    intercept = float(getattr(ds, "RescaleIntercept", 0.0)) #Desplazamiento base (offset) de intensidad

    img_real = ds.pixel_array.astype(np.float64) * slope + intercept #Convierte los valores crudos a valores físicos reales de intensidad

    volumen_imgs.append(img_real)
    tiempos_eco.append(float(ds.EchoTime)) #Guarda el Tiempo de Eco (TE) en milisegundos de cada corte

volumen_imgs = np.array(volumen_imgs) #Junta todas las imágenes en un arreglo 3D (tiempo, alto, ancho)
tiempos_eco = np.array(tiempos_eco)

#Función para mejorar la nitidez de la imagen aplicando un filtro pasa-altos gaussiano en el dominio de frecuencias (espacio-k)
def filtro_nitidez(imagen, ganancia, sigma):
    fils, cols = imagen.shape
    u = np.arange(fils) - fils // 2 #Coordenadas de frecuencia vertical centradas en cero
    v = np.arange(cols) - cols // 2 #Coordenadas de frecuencia horizontal centradas en cero
    U, V = np.meshgrid(u, v, indexing="ij")
    D2 = U**2 + V**2 #Distancia al cuadrado desde el centro del espacio-k

    filtro_high_pass = 1.0 - np.exp(-D2 / (2 * sigma**2)) #Filtro que atenúa las frecuencias bajas

    f_shift = np.fft.fftshift(np.fft.fft2(imagen)) #Fourier
    f_filtrada = f_shift * (1.0 + ganancia * filtro_high_pass) #Refuerza los detalles según la ganancia dada
    img_nitida = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtrada))) #Transformada inversa
    return img_nitida


img_base = volumen_imgs[0] #Toma la primera imagen (pues tiene el menor TE y por tanto mayor señal) como referencia
img_nitida = filtro_nitidez(img_base, 0.7, 10.0) #Aqui se aplica filtro

#Separación de las gotas de agua de referencia ubicados en la parte inferior de la imagen
franja_inf = img_base[110:120, :]
umbral_agua = 0.5 * np.max(franja_inf) #Aqui se define un umbral del 50% del brillo máximo para separar el agua del fondo
mascara_agua = np.zeros_like(img_base, dtype=bool)
mascara_agua[110:120, :] = franja_inf > umbral_agua

curva_agua = np.array([img[mascara_agua].mean() for img in volumen_imgs]) #Calcula la intensidad promedio del agua a lo largo de todos los TE

#Aqui se calculan las curvas de decaimiento promedio en tres regiones de interés (ROIs) del tomate
roi_pericarpio = volumen_imgs[:, 7:17, 55:65].mean(
    axis=(1, 2) #Pared externa del tomate
)
roi_gel_semillas = volumen_imgs[:, 22:30, 53:61].mean(
    axis=(1, 2) #Zona gelatinosa que rodea las semillas
)
roi_placenta = volumen_imgs[:, 63:73, 58:68].mean(
    axis=(1, 2) #Tejido central del tomate
)

#Modelo físico de relajación transversal T2: sigue un decaimiento exponencial de la señal más un nivel base de ruido (c)
def modelo_t2(te, s0, t2, c):
    return s0 * np.exp(-te / t2) + c

#Ajusta los datos experimentales al modelo exponencial usando mínimos cuadrados no lineales para hallar s0, T2 y c
def calcular_t2(te, senal):
    p0 = [senal[0] - senal[-1], 60.0, senal[-1]] #Aqui se toman algunos valores iniciales estimados para ayudar a que el ajuste converja
    popt, _ = curve_fit(
        modelo_t2, te, senal, p0=p0, bounds=(0, [np.inf, 2000.0, np.inf]) #Restringe T2 entre 0 y 2000 ms para evitar resultados sin sentido físico
    )
    return popt

#Calcula los parámetros para el agua y cada tejido del tomate
params_agua = calcular_t2(tiempos_eco, curva_agua)
params_pericarpio = calcular_t2(tiempos_eco, roi_pericarpio)
params_gel = calcular_t2(tiempos_eco, roi_gel_semillas)
params_placenta = calcular_t2(tiempos_eco, roi_placenta)

f = open("Tiempos_Relajacion_T2.txt", "w")
f.write("   TIEMPOS DE RELAJACION T2\n")
f.write(f"\nAgua (Referencia): T2 = {params_agua[1]:.2f} ms \nGel locular (Semillas): T2 = {params_gel[1]:.2f} ms \nPericarpio (Pared externa): T2 = {params_pericarpio[1]:.2f} ms \nPlacenta (Centro): T2 = {params_placenta[1]:.2f} ms")
f.close()

#Aqui se va a construir el mapa T2
mapa_t2 = np.zeros_like(img_base)
umbral_fondo = 0.08 * np.max(img_base) #Aqui se toma un umbral del 8% para no tomar en cuenta el aire alrededor del tomate

te_sub = tiempos_eco[:15] #Aqui se usa solo los primeros 15 tiempos de eco donde la señal aún tiene buena relación señal-ruido
for y in range(128):
    for x in range(128):
        if img_base[y, x] > umbral_fondo:
            perfil = np.maximum(volumen_imgs[:15, y, x], 1.0) #Aqui se evitan ceros o números negativos antes de aplicar el logaritmo
            pendiente, _ = np.polyfit(te_sub, np.log(perfil), 1) #Aqui se linealiza el decaimiento exponencial
            t2_val = -1.0 / pendiente
            mapa_t2[y, x] = np.clip(t2_val, 0, 350)

#Tercera gráfica: Imagen original, filtro de nitidez, curvas T2 y mapa cuantitativo
fig, ax = plt.subplots(2, 2, figsize=(13, 10))

ax[0, 0].imshow(img_base, cmap="gray")
ax[0, 0].set_title("MRI Original (TE = 10 ms) + ROIs")
ax[0, 0].add_patch(
    plt.Rectangle((55, 7), 10, 10, edgecolor="cyan", facecolor="none", lw=1.5) #ROI Pericarpio
)
ax[0, 0].add_patch(
    plt.Rectangle((53, 22), 8, 8, edgecolor="lime", facecolor="none", lw=1.5) #ROI Gel de semillas
)
ax[0, 0].add_patch(
    plt.Rectangle((58, 63), 10, 10, edgecolor="yellow", facecolor="none", lw=1.5) #ROI Placenta
)
ax[0, 0].contour(
    mascara_agua, levels=[0.5], colors="blue", linewidths=1.5
)

ax[0, 1].imshow(img_nitida, cmap="gray")
ax[0, 1].set_title("MRI con Filtro de Nitidez")

te_fino = np.linspace(10, 320, 200)
norm = curva_agua[0]

ax[1, 0].plot(
    tiempos_eco, curva_agua / norm, "bo", markersize=4, label="Gotas de Agua"
)
ax[1, 0].plot(
    te_fino,
    modelo_t2(te_fino, *params_agua) / norm,
    "b-",
    label=f"Agua ($T_2$={params_agua[1]:.1f} ms)",
)
ax[1, 0].plot(
    tiempos_eco,
    roi_gel_semillas / norm,
    "go",
    markersize=4,
    label="Gel Semillas",
)
ax[1, 0].plot(
    te_fino,
    modelo_t2(te_fino, *params_gel) / norm,
    "g-",
    label=f"Gel ($T_2$={params_gel[1]:.1f} ms)",
)
ax[1, 0].plot(
    tiempos_eco,
    roi_pericarpio / norm,
    "co",
    markersize=4,
    label="Pericarpio",
)
ax[1, 0].plot(
    te_fino,
    modelo_t2(te_fino, *params_pericarpio) / norm,
    "c-",
    label=f"Pericarpio ($T_2$={params_pericarpio[1]:.1f} ms)",
)
ax[1, 0].plot(
    tiempos_eco, roi_placenta / norm, "yo", markersize=4, label="Placenta"
)
ax[1, 0].plot(
    te_fino,
    modelo_t2(te_fino, *params_placenta) / norm,
    "y-",
    label=f"Placenta ($T_2$={params_placenta[1]:.1f} ms)",
)
ax[1, 0].set_title("Decaimiento $T_2$ Calibrado al Agua")
ax[1, 0].set_xlabel("Tiempo de Eco TE (ms)")
ax[1, 0].set_ylabel("Intensidad Normalizada (u.a.)")
ax[1, 0].legend(fontsize=8)
ax[1, 0].grid(True, alpha=0.3)

im_t2 = ax[1, 1].imshow(mapa_t2, cmap="inferno", vmin=0, vmax=250)
ax[1, 1].set_title("Mapa Cuantitativo de $T_2$ (ms)")
plt.colorbar(im_t2, ax=ax[1, 1], label="$T_2$ (ms)")

plt.tight_layout()
plt.savefig("MRI_Analysis.png")
plt.show()