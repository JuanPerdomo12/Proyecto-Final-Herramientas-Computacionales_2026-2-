import matplotlib.pyplot as plt
import numpy as np
import pydicom

rut_arch = "C:\\Users\\jujo1\\OneDrive\\Documentos\\Proyecto-Final-Herramientas-Computacionales_2026-2-\\Tomate\\PvMRIm001"
ds = pydicom.dcmread(rut_arch)

imagen = ds.pixel_array

print(f"{imagen.shape}, {imagen.dtype}")

f_imagen = np.fft.fft2(imagen)
f_shift = np.fft.fftshift(f_imagen)

magnitud = 20 * np.log(np.abs(f_shift))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].imshow(imagen, cmap="gray")
ax[0].set_title("MRI Original")
ax[0].axis("off")
ax[1].imshow(magnitud, cmap="gray")
ax[1].set_title("Espectro de Fourier")
ax[1].axis("off")

plt.tight_layout()
plt.show()