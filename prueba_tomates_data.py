import glob
import os
import pydicom

ruta_carpeta = r"C:\Users\jujo1\OneDrive\Documentos\Proyecto-Final-Herramientas-Computacionales_2026-2-\Tomate_corte_tiempos"

# Buscar todos los archivos PvMRIm* ordenados
archivos = sorted(glob.glob(os.path.join(ruta_carpeta, "PvMRIm*")))

print(f"Total de imágenes encontradas: {len(archivos)}\n")
print(
    f"{'Archivo':<12} | {'TE (ms)':<10} | {'TR (ms)':<10} | {'Slice (mm)':<12} | {'Fecha/Hora':<18}"
)
print("-" * 72)

for arch in archivos:
    ds = pydicom.dcmread(arch, stop_before_pixels=True)
    nombre = os.path.basename(arch)
    te = getattr(ds, "EchoTime", "N/A")
    tr = getattr(ds, "RepetitionTime", "N/A")
    sl = getattr(ds, "SliceLocation", "N/A")
    hora = getattr(ds, "AcquisitionTime", "N/A")
    print(
        f"{nombre:<12} | {str(te):<10} | {str(tr):<10} | {str(sl):<12} | {str(hora):<18}"
    )

# Imprimir además todas las etiquetas disponibles de la primera imagen para explorar qué más trae
print("\n--- Metadatos completos de PvMRIm001 ---")
ds0 = pydicom.dcmread(archivos[0], stop_before_pixels=True)
print(ds0)