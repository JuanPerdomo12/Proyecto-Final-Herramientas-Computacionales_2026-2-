#CODIGO APOYADO COMPLETAMENTE EN IA PUES SOLO SE USA PARA ESCRIBIR EL ARCHIVO HTML QUE RECONSTRUYE LA IMAGEN 3D DE LA RODILLA SOBREPONIENDO LOS CORTES 2D DE LA MISMA. EL PROCESO DE RECONSTRUCCION 3D SE HACE CON LA LIBRERIA PLOTLY Y EL ALGORITMO MARCHING CUBES DE SKIMAGE.
import glob
import os
import numpy as np
from PIL import Image
import plotly.graph_objects as go #Se debe instalar previamente esta libreria con pip install plotly
from scipy.ndimage import gaussian_filter
from skimage import measure #Se debe instalar previamente esta libreria con pip install scikit-image

carpeta_imagenes = "Cortes_rodilla"
archivo_salida_html = "rodilla_3d.html"

#Carga de todas las imágenes 2D en orden
rutas = sorted(glob.glob(os.path.join(carpeta_imagenes, "corte_*.png")))

capas = list()
for ruta in rutas:
  #Abrir en escala de grises
  img = Image.open(ruta).convert("L")
  #Reducir resolución a la mitad para que el 3D sea fluido en el navegador
  img_pequena = img.resize((img.width // 2, img.height // 2))
  capas.append(np.array(img_pequena, dtype=np.float32))

#Apilar en un arreglo 3D
volumen = np.stack(capas, axis=0)

#Suavizar ligeramente para unir las capas de forma continua
volumen_suavizado = gaussian_filter(volumen, sigma=1.2)

#El umbral define qué tejido se reconstruye
umbral = 45.0

vertices, caras, normales, valores = measure.marching_cubes(
    volumen_suavizado, level=umbral, step_size=2
)

#Separar coordenadas X, Y, Z de los vértices y los índices de los triángulos
z, y, x = vertices.T
i, j, k = caras.T

#Visualización 3D interactiva con Plotly
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
    title="Reconstrucción 3D de la Rodilla",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Corte (Z)",
        aspectmode="data",  #Mantiene las proporciones reales sin deformar la rodilla
    ),
    margin=dict(l=0, r=0, b=0, t=40),
)

fig.write_html(archivo_salida_html)