#CODIGO APOYADO COMPLETAMENTE EN IA PUES SOLO SE USA PARA ESCRIBIR EL ARCHIVO HTML QUE RECONSTRUYE LA IMAGEN 3D DE LA RODILLA SOBREPONIENDO LOS CORTES 2D DE LA MISMA. EL PROCESO DE RECONSTRUCCION 3D SE HACE CON LA LIBRERIA PLOTLY Y EL ALGORITMO MARCHING CUBES DE SKIMAGE.
import glob
import os
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from skimage import measure #Se debe instalar previamente esta libreria con pip install scikit-image

carpeta_imagenes = "Cortes_rodilla"
archivo_salida_html = "rodilla_3d.html"

# 1. Cargar todas las imágenes 2D en orden
rutas = sorted(glob.glob(os.path.join(carpeta_imagenes, "corte_*.png")))
print(f"Cargando {len(rutas)} cortes para apilar en 3D...")

capas = list()
for ruta in rutas:
  # Abrir en escala de grises ('L')
  img = Image.open(ruta).convert("L")
  # Reducir resolución a la mitad (ej. 160x160) para que el 3D sea fluido en el navegador
  img_pequena = img.resize((img.width // 2, img.height // 2))
  capas.append(np.array(img_pequena, dtype=np.float32))

# Apilar en un arreglo 3D: (Z = cortes, Y = alto, X = ancho)
volumen = np.stack(capas, axis=0)
print(f"Forma del volumen 3D: {volumen.shape}")

# 2. Suavizar ligeramente para unir las capas de forma continua
volumen_suavizado = gaussian_filter(volumen, sigma=1.2)

# 3. Extraer la superficie 3D (Marching Cubes)
# El umbral (level) define qué tejido se reconstruye:
# - Valores bajos (ej. 35-50): contorno externo de la pierna/piel y músculos
# - Valores altos (ej. 90-130): estructuras internas más brillantes (tendones/grasa)
umbral = 45.0
print(f"Generando malla 3D con umbral de intensidad = {umbral}...")

vertices, caras, normales, valores = measure.marching_cubes(
    volumen_suavizado, level=umbral, step_size=2
)

# Separar coordenadas X, Y, Z de los vértices y los índices de los triángulos
z, y, x = vertices.T
i, j, k = caras.T

# 4. Crear visualización 3D interactiva con Plotly
fig = go.Figure(
    data=[
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            color="lightgray",
            opacity=0.85,
            flatshading=False,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3, roughness=0.5),
            name="Rodilla 3D",
        )
    ]
)

fig.update_layout(
    title="Reconstrucción 3D de la Rodilla (Capa por Capa)",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Corte (Z)",
        aspectmode="data",  # Mantiene las proporciones reales sin deformar la rodilla
    ),
    margin=dict(l=0, r=0, b=0, t=40),
)

fig.write_html(archivo_salida_html)

