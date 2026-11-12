import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image

prueba1 = Image.open('prueba1_B&W.jpg')

matriz = np.array(prueba1)

matriz_fft = fft2(matriz)
matriz_fft_shift = fftshift(matriz_fft)
magnitud = 20*np.log(np.abs(matriz_fft_shift))

plt.figure(figsize=(10, 10), dpi=120)
plt.imshow(magnitud, cmap='gray')
plt.axis('off')
plt.show()

matriz_fft_unshift = ifftshift(matriz_fft_shift)
matriz_nueva = np.real(ifft2(matriz_fft))

plt.figure(figsize=(10, 10), dpi=120)
plt.imshow(matriz_nueva, cmap='gray')
plt.axis('off')
plt.show()

prueba2 = Image.open('Prueba2.jpg')

matriz2 = np.array(prueba2)

matriz2_fft = fft2(matriz2)
matriz2_fft_shift = fftshift(matriz2_fft)
magnitud2 = np.abs(matriz2_fft_shift)

plt.figure(figsize=(10, 10), dpi=120)
plt.imshow(magnitud2, cmap='gray')
plt.axis('off')
plt.show()

matriz2_fft_unshift = ifftshift(matriz2_fft_shift)
matriz2_nueva = np.real(ifft2(matriz2_fft))

plt.figure(figsize=(10, 10), dpi=120)
plt.imshow(matriz2_nueva)
plt.axis('off')
plt.show()