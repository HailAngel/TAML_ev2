# Clasificación de Imágenes - Intel Image Classification

Este proyecto entrena y evalua un modelo cnn utilizando TensorFlow/Keras para clasificar imágenes de paisajes urbanos y naturales basándose en el dataset de Intel, predice las clases ciudad, naturaleza y mar. Al final se compara con un modelo mlp previamente entrenado.

##  Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

* `data/`: Carpeta destinada a alojar el dataset crudo y descomprimido (ignorado en GitHub por peso).
* `images/`: Gráficos, metricas y resultados visuales del entrenamiento.
* `models/`: Modelo entrenado y exportado `modelo_final.keras`.
* `notebooks/`: Jupyter Notebooks con el análisis exploratorio y el entrenamiento del modelo.

## Dataset y Modelos

Debido a las restricciones de tamaño de GitHub, el dataset original y el modelo entrenado no están incluidos en este repositorio. 
* **Dataset:** Puedes descargar el dataset *Intel Image Classification* desde https://www.kaggle.com/datasets/puneet6060/intel-image-classification.
* **Modelo:** El archivo `modelo_final.keras` se genera automáticamente al ejecutar el notebook, o puedes descargarlo desde la carpeta models.

## Cómo ejecutar el proyecto

Puedes correr este proyecto de dos maneras: en la nube usando Google Colab o en tu entorno local.

### Opción 1: En Google Colab (Recomendado)
1. Sube el notebook a Google Colab.
2. Ve a la **Celda 0** y descoméntala (elimina los `#`). Esta celda se encargará de descomprimir el dataset (hay que subir el zip previamente).
3. Ejecuta todas las celdas en orden.

### Opción 2: En Entorno Local
Si prefieres correr el notebook en tu propio computador (Jupyter Notebook, VSCode, etc.), sigue estos pasos:

1. Descarga el dataset *Intel Image Classification* desde el enlace mencionado arriba.
2. Descomprime el archivo descargado y coloca los datos directamente dentro de la carpeta `data/` de este proyecto.
3. Abre el notebook y **asegúrate de que la Celda 0 esté comentada** para evitar descargas innecesarias.
4. En la **Celda 1**, cambia las rutas de los directorios para que apunten correctamente a tu carpeta `data/` local.
5. Ejecuta las celdas secuencialmente. 
6. Al llegar a la predicción y guardado del modelo en el **Punto 7**, verifica y ajusta la ruta para asegurarte de que el modelo se guarde correctamente como `models/modelo_final.keras`.

## Resultados y Conclusiones

* **Exactitud del modelo (Accuracy):** 92%
* **Observaciones:** Tal y como esta le falta un poco de mejora ya que clasifica muchas imagenes de `naturaleza` como `mar`.
