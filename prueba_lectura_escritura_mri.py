import matplotlib.pyplot as plt
import numpy as np

ruta_exp = (
    "/home/juperdomof/Proyecto-Final-Herramientas-Computacionales-2026-II/"
    "Tomate_raw_data/nmrTom290114/C-0-3/2/"
)

raw_fid = np.fromfile(ruta_exp + "fid", dtype="<i4")

k_1d = raw_fid[0::2] + 1j * raw_fid[1::2]

filas, columnas = 128, 128
k_2d = k_1d.reshape((filas, columnas))

img_reconstruida = np.abs(
    np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(k_2d)))
)

ref_real = np.fromfile(ruta_exp + "pdata/1/1r", dtype="<i4")
ref_imag = np.fromfile(ruta_exp + "pdata/1/1i", dtype="<i4")

img_referencia = np.abs(ref_real + 1j * ref_imag).reshape((256, 128))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(np.log1p(np.abs(k_2d)), cmap="gray", aspect="auto")
axes[0].set_title("Espacio-k (Log Magnitud)")

axes[1].imshow(img_reconstruida, cmap="gray", aspect="auto")
axes[1].set_title(f"Tu Reconstrucción ({filas}x{columnas})")

axes[2].imshow(img_referencia, cmap="gray")
axes[2].set_title(f"Referencia Bruker ({256}x{128})")

plt.tight_layout()
plt.show()
