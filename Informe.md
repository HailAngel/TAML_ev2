# **Informe Técnico: Clasificación de Escenarios Naturales mediante Redes Neuronales Convolucionales (CNN) y Perceptrón Multicapa (MLP)**

Asignatura: Técnicas avanzadas de Machine Learning

Sección: 001D

Profesor: Ruben Claveria Vega

Integrantes: Angel Pimiento y Ayline Ríos


## **1. Descripción del Problema de Negocio.**

* **Empresa:** Agencia de viajes y turismo digital.

* **Problema:** La plataforma recibe miles de fotografías subidas por usuarios y sitios asociados sin etiquetar. La clasificación manual es inviable en costo y tiempo, impidiendo personalizar la oferta turística en tiempo real.
* **Solución propuesta:** Sistema automatizado de visión computacional mediante una red convolucional (CNN) que clasifique las imágenes en tres entornos clave: **Ciudad**, **Mar** y **Naturaleza**, permitiendo perfilar preferencias y recomendar panoramas acordes al usuario.



### Objetivos del Proyecto
* **Objetivo General:** Desarrollar e implementar un modelo de clasificación multiclase basado en una Red Neuronal Convolucional (CNN) para categorizar imágenes de destinos turísticos con un flujo de trabajo reproducible.
* **Objetivos Específicos:**
  1. Preprocesar y adaptar imágenes RGB (150x150 px) al formato requerido por la CNN.
  2. Entrenar y validar la red neuronal monitoreando convergencia y sobreajuste (loss/accuracy).
  3. Evaluar el desempeño mediante métricas de clasificación (Accuracy, Precision, Recall, F1-Score y Matriz de Confusión).
  4. Analizar casos de error y explicar por que erró.


### Definición de KPIs
* **KPIs de Negocio:**
  * **Tiempo de catalogación:** Reducción del 95% en el tiempo de etiquetado frente al proceso manual.
  * **Tasa de cobertura de recomendación:** Clasificar al menos el 80% de las fotos entrantes sin intervención humana.
* **KPIs Técnicos (Modelo):**
  * **Accuracy global:** $\ge 85\%$ en el conjunto de prueba.
  * **Macro F1-Score:** $\ge 0.85$.


### Dataset y Justificación de la Variante
* **Fuente:** *Intel Image Classification Dataset*.
* **Variante del grupo:**
  * **Subconjunto de 3 clases:** `ciudad` (Street/Buildings), `mar` (Sea) y `naturaleza` (Forest/Mountain/Glacier). Justificado en los 3 pilares de destino del negocio turístico.
  * **Resolución:** $150 \times 150$ píxeles en 3 canales (RGB).
  * **Semilla fija:** `SEED = 42` para garantizar trazabilidad y reproducibilidad.
  * **Partición de datos:** `seg_train`(80% entrenamiento, 20% validación) 14000 img aprox. y `seg_test` test independiente 3000 img aprox.

### ¿Por que resulta pertinente comparar un mlp con una cnn?
  Comparar ambos modelos ilustra por qué las CNN son superiores para procesar imágenes. Mientras el MLP destruye la estructura espacial al aplanar los datos y sufre de una cantidad ineficiente de parámetros, la CNN preserva la geometría original y utiliza filtros compartidos para reconocer patrones visuales en cualquier posición, las cnn son mas costosas en cuanto a procesamiento pero son mas eficientes en el costo en memoria.


## **2. Definición de Variante Propia y Justificación Técnica**

Para garantizar la autenticidad del trabajo y cumplir con el requerimiento de continuidad respecto a la Evaluación Parcial 1:

**Variante Seleccionada:** Agrupación y reducción de 6 clases originales a 3 macro-categorías clave:


1. **ciudad:** Fusión de las clases originales buildings y street.

2. **mar:** Conservación de la clase original sea.

3. **naturaleza:** Fusión de las clases forest, glacier y mountain.

**Semilla de Aleatoriedad (Random State):** 42 para la partición de datos.


###**Justificación Técnica:**

Reducción de sobrelape semántico: bulding y street comparten características visuales (hormigón, estructuras, vehiculos). Del mismo modo, glacier y mountain presentan un alto grado de similitud cromática y morfológica (roca, nieve, contornos angulares).


## **3. Objetivos del Proyecto y KPIs**

1. Diseñar, entrenar y evaluar una Red Neuronal Convolucional (CNN) en TensorFlow/Keras para la clasificación de las 3 macro-categorías.

2. Comparar cuantitativa y cualitativamente el desempeño de la CNN contra el modelo Perceptrón Multicapa (MLP) desarrollado en la Evaluación Parcial 1.

3. Extraer e interpretar los mapas de características (Feature Maps) y mapas de activación para comprender el proceso de extracción automática de atributos.

###**KPIs de Negocio y Métricas Técnicas**

- **Accuracy Global (KPI Principal):** Meta $\ge 85\%$ en el conjunto de prueba.

- **F1-SCore Macro (KPI de Balance):** Meta $\ge 85\%$ para segurar un rendimiento equitativo en las 3 clases.
- **Tiempo de Inferencia por Imagen:** $< 15\text{ ms}$ por muestra en CPU/GPU.

- **Tasa de Error de Clasificación Cruzada:** Minimizar las confusiones entre mar y naturaleza.


## **4. Descripción de las Fuentes de Datos y EDA**


**Dataset Base:** Intel Image Classification Dataset.

### **Volumen:**

**Entrenamiento original:** ~14,034 imágenes.

**Test original:** ~3,000 imágenes.

**Resolución:** $150 \times 150$ píxeles promedio, formato RGB (3 canales).

###**Resultados del Análisis Exploratorio de Datos (EDA)**

**Distribución de Clases (Split de Entrenamiento):**

Tras el mapeo a las 3 macro-clases, la clase naturaleza representa aproximadamente el 50% de las imágenes (al agrupar 3 clases originales), mientras que ciudad y mar representan el ~33% y ~17% respectivamente. Se aplica estratificación (stratify=train_df['etiqueta']) para mantener la proporción exacta en los splits de train (80%) y validación (20%).

**Dimensiones:** La mayoría de las imágenes conservan una relación de aspecto $150 \times 150$, confirmando que un redimensionamiento estandarizado a $128 \times 128$ o $150 \times 150$ es óptimo.


## **5. Metodología (CRISP-DM) y Preparación de Datos**

Para garantizar un flujo de trabajo estructurado, trazable y orientado a resultados de negocio, el proyecto se desarrolla bajo la metodología estándar de la industria **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*):

1. **Comprensión del Negocio:** Definición del problema de etiquetado en la plataforma turística, estableciendo los KPIs técnicos ($\text{Accuracy} \ge 85\%$, $\text{Macro F1-Score} \ge 0.85$) y de negocio (reducción del 95% en tiempo de procesamiento).

2. **Comprensión de Datos:** Carga e inspección de las imágenes mediante `pathlib` y `pandas`. Se analizó la distribución de las clases originales y se estructuró la estrategia de agrupación.

3. **Preparación de Datos:**

   * **Mapeo de Clases:** Conversión de las 6 categorías originales a las 3 macro-categorías clave (`ciudad`, `mar`, `naturaleza`).

   * **Ajuste Dimensional:** Estandarización del tamaño de las imágenes a un formato tensorial uniforme de $(150, 150, 3)$ píxeles, preservando los 3 canales de color (RGB) y la topología espacial.

   * **Escalado y Normalización:** Aplicación de un reescalado dinámico de píxeles dividiendo cada valor por $255.0$ (`rescale=1./255` mediante `ImageDataGenerator` / `Rescaling`), transformando la intensidad de los píxeles del rango entero $[0, 255]$ al rango flotante $[0.0, 1.0]$. Esto acelera la convergencia del algoritmo de optimización (Gradient Descent).

   * **Técnicas de Aumento de Datos (Data Augmentation):** Debido a la dominancia de la clase `naturaleza` (~50% del conjunto total), se implementaron transformaciones aleatorias durante el entrenamiento (tales como giros horizontales `horizontal_flip=True`, variaciones de zoom `zoom_range=0.2` y pequeñas rotaciones `rotation_range=15`) aplicadas únicamente al split de *train*. Esto expande la diversidad del dataset, evita que la CNN memorice características estáticas y mitiga significativamente el sobreajuste (*overfitting*).

   * **Estratificación y Partición de Datos:** Creación de los generadores de datos con una división estratificada del 80% para entrenamiento y 20% para validación, manteniendo un conjunto de prueba (*test*) completamente independiente de aproximadamente 3,000 imágenes.

4. **Modelado:** Configuración del Perceptrón Multicapa (MLP) base y diseño de la arquitectura de la Red Neuronal Convolucional (CNN).

5. **Evaluación:** Comparación de métricas globales, análisis de la matriz de confusión e interpretación de los mapas de características.

6. **Despliegue:** Formulación de recomendaciones de integración para la plataforma web de la agencia de viajes.

