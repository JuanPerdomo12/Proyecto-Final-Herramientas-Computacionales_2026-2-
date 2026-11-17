import glob
import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon

# ==========================================
# CONFIGURACIÓN DEL CORTE
corte = 1          # Puedes elegir 1, 2 o 3
adquisicion = 2    # 2 = adquisición con gradientes espaciales
# ==========================================

# 1. Buscar y ordenar numéricamente los ángulos del corte elegido
patron = f"nmrTom290114/C-*-{corte}/{adquisicion}"
rutas = sorted(
    glob.glob(patron),
    key=lambda ruta: int(ruta.split("C-")[1].split("-")[0])
)

print(f"Procesando Corte {corte} completo con {len(rutas)} ángulos...")
proyecciones = []

for carpeta in rutas:
    # Leer señal cruda y quitar retardo digital de Bruker
    dic, data = ng.bruker.read(carpeta)
    data_limpia = ng.bruker.remove_digital_filter(dic, data)
    
    # Transformada Rápida de Fourier 1D completa (todos los 32,695 puntos)
    perfil_completo = np.abs(np.fft.fftshift(np.fft.fft(data_limpia)))
    
    # Normalizar cada proyección para evitar que variaciones de ganancia deformen la imagen
    perfil_completo = perfil_completo / np.max(perfil_completo)
    proyecciones.append(perfil_completo)

# 2. Construir el sinograma completo (Filas = todos los puntos, Columnas = ángulos)
sinograma_completo = np.column_stack(proyecciones)
print("Dimensiones del sinograma completo:", sinograma_completo.shape)

# 3. Reconstrucción tomográfica 2D mediante Retroproyección Filtrada (Radon Inversa)
num_angulos = sinograma_completo.shape[1]
angulos = np.linspace(0., 180., num_angulos, endpoint=False)

# output_size=512 evita crear una matriz gigante de 32695x32695 que sature la memoria RAM
imagen_reconstruida = iradon(
    sinograma_completo,
    theta=angulos,
    filter_name="cosine",
    output_size=512
)

# 4. Visualizar el sinograma completo y la reconstrucción 2D
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].imshow(sinograma_completo, aspect="auto", cmap="gray")
axes[0].set_title(f"1. Sinograma Completo - Corte {corte} ({sinograma_completo.shape[0]} pts)")
axes[0].set_xlabel("Índice de ángulo")
axes[0].set_ylabel("Posición espacial completa")

axes[1].imshow(imagen_reconstruida, cmap="gray")
axes[1].set_title(f"2. Imagen 2D Reconstruida (Corte {corte} completo)")
axes[1].axis("off")

plt.tight_layout()
plt.show()

