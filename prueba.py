import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from PIL import Image

#imagen = Image.open('prueba1_B&W.jpg').convert('L')
imagen = Image.open('Prueba2.jpg').convert('L')

imagen = imagen.resize((100, 100))

Z = np.array(imagen)


alto, ancho = Z.shape
x = np.arange(0, ancho, 1)
y = np.arange(0, alto, 1)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')


ax.invert_yaxis()

plt.axis('on')
plt.show()

matriz_img = np.array(imagen)
f_transform = np.fft.fft2(matriz_img)
f_shift = np.fft.fftshift(f_transform)
Z_fourier = 20 * np.log(np.abs(f_shift) + 1)


alto, ancho = Z_fourier.shape
x = np.arange(-ancho // 2, ancho // 2)
y = np.arange(-alto // 2, alto // 2)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z_fourier, cmap='plasma', edgecolor='none')
ax.set_xlabel('Frecuencia X')
ax.set_ylabel('Frecuencia Y')

plt.show()

sand_fft_unshift = ifftshift(f_shift)
sand_nueva = np.real(ifftn(f_transform))
plt.figure(figsize=(10, 10), dpi=120)
plt.imshow(sand_nueva)
plt.axis('off')
plt.show()