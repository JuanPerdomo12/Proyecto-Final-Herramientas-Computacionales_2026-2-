import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from scipy.optimize import curve_fit

ruta_carpeta = r"C:\Users\jujo1\OneDrive\Documentos\Proyecto-Final-Herramientas-Computacionales_2026-2-\Tomate_corte_tiempos"
archivos = sorted(glob.glob(os.path.join(ruta_carpeta, "PvMRIm*")))

volumen_imgs = []
tiempos_eco = []

for arch in archivos:
    ds = pydicom.dcmread(arch)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    img_real = ds.pixel_array.astype(np.float64) * slope + intercept

    volumen_imgs.append(img_real)
    tiempos_eco.append(float(ds.EchoTime))

volumen_imgs = np.array(volumen_imgs)
tiempos_eco = np.array(tiempos_eco)

def filtro_nitidez(imagen, ganancia, sigma):
    fils, cols = imagen.shape
    u = np.arange(fils) - fils // 2
    v = np.arange(cols) - cols // 2
    U, V = np.meshgrid(u, v, indexing="ij")
    D2 = U**2 + V**2

    filtro_high_pass = 1.0 - np.exp(-D2 / (2 * sigma**2))

    f_shift = np.fft.fftshift(np.fft.fft2(imagen))
    f_filtrada = f_shift * (1.0 + ganancia * filtro_high_pass)
    img_nitida = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtrada)))
    return img_nitida


img_base = volumen_imgs[0]
img_nitida = filtro_nitidez(img_base, 0.7, 10.0)

franja_inf = img_base[105:125, :]
umbral_agua = 0.5 * np.max(franja_inf)
mascara_viales = np.zeros_like(img_base, dtype=bool)
mascara_viales[105:125, :] = franja_inf > umbral_agua

curva_agua = np.array([img[mascara_viales].mean() for img in volumen_imgs])

roi_pericarpio = volumen_imgs[:, 22:32, 58:70].mean(
    axis=(1, 2)
)
roi_gel_semillas = volumen_imgs[:, 40:50, 42:52].mean(
    axis=(1, 2)
)
roi_placenta = volumen_imgs[:, 60:70, 58:70].mean(
    axis=(1, 2)
)

def modelo_t2(te, s0, t2, c):
    return s0 * np.exp(-te / t2) + c


def calcular_t2(te, senal):
    p0 = [senal[0] - senal[-1], 60.0, senal[-1]]
    popt, _ = curve_fit(
        modelo_t2, te, senal, p0=p0, bounds=(0, [np.inf, 2000.0, np.inf])
    )
    return popt


params_agua = calcular_t2(tiempos_eco, curva_agua)
params_pericarpio = calcular_t2(tiempos_eco, roi_pericarpio)
params_gel = calcular_t2(tiempos_eco, roi_gel_semillas)
params_placenta = calcular_t2(tiempos_eco, roi_placenta)

print("=== TIEMPOS DE RELAJACIÓN T2 ===")
print(f"Agua (Referencia): T2 = {params_agua[1]:.2f} ms")
print(f"Gel locular (Semillas): T2 = {params_gel[1]:.2f} ms")
print(f"Pericarpio (Pared externa): T2 = {params_pericarpio[1]:.2f} ms")
print(f"Placenta (Centro): T2 = {params_placenta[1]:.2f} ms")

mapa_t2 = np.zeros_like(img_base)
umbral_fondo = 0.08 * np.max(img_base)

te_sub = tiempos_eco[:15]
for y in range(128):
    for x in range(128):
        if img_base[y, x] > umbral_fondo:
            perfil = np.maximum(volumen_imgs[:15, y, x], 1.0)
            pendiente, _ = np.polyfit(te_sub, np.log(perfil), 1)
            if pendiente < -1e-4:
                t2_val = -1.0 / pendiente
                mapa_t2[y, x] = np.clip(t2_val, 0, 350)

fig, ax = plt.subplots(2, 2, figsize=(13, 10))

ax[0, 0].imshow(img_base, cmap="gray")
ax[0, 0].set_title("MRI Original (TE = 10 ms) + ROIs")
ax[0, 0].add_patch(
    plt.Rectangle((58, 22), 12, 10, edgecolor="cyan", facecolor="none", lw=1.5)
)
ax[0, 0].add_patch(
    plt.Rectangle((42, 40), 10, 10, edgecolor="lime", facecolor="none", lw=1.5)
)
ax[0, 0].add_patch(
    plt.Rectangle(
        (58, 60), 12, 10, edgecolor="yellow", facecolor="none", lw=1.5
    )
)
ax[0, 1].imshow(img_nitida, cmap="gray")
ax[0, 1].set_title("MRI con Filtro de Nitidez (Espacio-k)")
te_fino = np.linspace(10, 320, 200)
norm = curva_agua[0]
ax[1, 0].plot(
    tiempos_eco, curva_agua / norm, "bo", markersize=4, label="Viales Agua"
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
ax[1, 0].set_title("Decaimiento $T_2$ Calibrado Agua")
ax[1, 0].set_xlabel("Tiempo de Eco TE (ms)")
ax[1, 0].set_ylabel("Intensidad Normalizada (u.a.)")
ax[1, 0].legend(fontsize=8)
ax[1, 0].grid(True, alpha=0.3)
im_t2 = ax[1, 1].imshow(mapa_t2, cmap="inferno", vmin=0, vmax=250)
ax[1, 1].set_title("Mapa Cuantitativo de $T_2$ (ms)")
plt.colorbar(im_t2, ax=ax[1, 1], label="$T_2$ (ms)")

plt.tight_layout()
plt.show()