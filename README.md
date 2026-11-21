# Detección de imágenes generadas por IA y comparación con el juicio de estudiantes universitarios guatemaltecos

Trabajo de graduación · Licenciatura en Ingeniería en Ciencias de la Computación y Tecnologías de la Información · Universidad del Valle de Guatemala, 2026

## Descripción

Este proyecto compara el desempeño de un clasificador de aprendizaje profundo con el de estudiantes universitarios al distinguir fotografías reales de imágenes generadas por inteligencia artificial (IA). Además, evalúa si la mirada de las personas coincide con las regiones de la imagen que el modelo usa para decidir.

El repositorio contiene el modelo usado en la comparación (`src/ModeloClasificacion.py`), que:

- Audita el conjunto de datos para medir si las clases se pueden separar solo por metadatos de archivo (formato, tamaño, compresión).
- Entrena una red **ResNet-50** con aprendizaje por transferencia, aplicando el mismo preprocesamiento a ambas clases (remuestreo, recompresión JPEG y desenfoque) para que el modelo no aprenda el formato del archivo.
- Evalúa el modelo en la partición de prueba interna y en los 40 estímulos del experimento del Neurolab (20 reales y 20 generados con ChatGPT).
- Genera mapas de **Grad-CAM** y de **sensibilidad por oclusión** que se compararon con las fijaciones registradas por *eye tracking*.

Resultados principales:

| Medida | Valor |
|---|---|
| AUC-ROC en la partición de prueba interna | 0.969 |
| AUC-ROC con los estímulos del experimento | 0.573 (nivel del azar) |
| Exactitud de los participantes (20 estudiantes, 800 respuestas) | 64.5 % |
| AUC-ROC de un clasificador que solo usa metadatos de archivo | 0.913 |

El análisis completo está en el [informe final](docs/informe_final.pdf).

## Tecnologías utilizadas

- Python 3.12
- PyTorch y torchvision (ResNet-50 preentrenada en ImageNet)
- NumPy y Pillow (procesamiento de imágenes)
- scikit-learn (métricas y prueba de atajos con regresión logística)
- Matplotlib (curvas ROC y matrices de confusión)
- tqdm (barras de progreso)
- iMotions con *eye tracker* Smart Eye (registro de la mirada en el Neurolab; los datos se exportaron para el análisis)

## Requisitos previos

- Python 3.12 y `pip`
- Git
- Aproximadamente 2 GB de espacio libre para las dependencias
- Opcional: GPU con CUDA o Mac con Apple Silicon (MPS). El modelo también funciona en CPU; solo el entrenamiento completo es lento sin GPU.
- Para reentrenar: se puede utilizar el conjunto de datos público [DALL-E Recognition Dataset](https://www.kaggle.com/datasets/superpotato9/dalle-recognition-dataset) de Kaggle (requiere una cuenta gratuita).

## Instalación

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/GraduacionesUVG/2026-21471-juarez-payes-sebastian.git
   cd 2026-21471-juarez-payes-sebastian
   ```

2. Crear un entorno virtual e instalar las dependencias:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate          # En Windows: .venv\Scripts\activate
   pip install -r src/requirements.txt
   ```

3. Configurar las variables de entorno (opcionales; el modelo no usa credenciales):

   ```bash
   cp src/.env.example src/.env
   set -a; source src/.env; set +a
   ```

4. Ejecutar el proyecto. Todos los comandos se corren desde `src/`:

   ```bash
   cd src
   python ModeloClasificacion.py -h
   ```

### Uso con el modelo entrenado

El repositorio incluye los pesos del modelo usado en la tesis (`src/runs/exp1/best.pt`) y su configuración (`src/runs/exp1/config.json`), con el umbral de decisión calibrado en validación. Con ellos se puede clasificar cualquier carpeta de imágenes y generar sus mapas sin reentrenar:

```bash
# Probabilidad de que cada imagen sea generada por IA
python ModeloClasificacion.py predict --out runs/exp1 --images <carpeta_de_imagenes>

# Mapas de Grad-CAM (layer3, layer4 y fusión) y de oclusión
python ModeloClasificacion.py gradcam --out runs/exp1 --images <carpeta_de_imagenes> --occlusion
```

Los resultados se guardan en `runs/exp1/` (`predictions_folder.csv` y la carpeta `gradcam/`, con un `.npy` y una superposición `.jpg` por mapa).

Para los estímulos del experimento, que se mostraron a pantalla completa en 3840 × 2160, se agrega `--display_size 3840x2160 --display_fit fill` para que el modelo vea los mismos píxeles que vieron los participantes. La evaluación pareada se ejecuta con:

```bash
python ModeloClasificacion.py testext --out runs/exp1 --images <carpeta_de_estimulos> --display_size 3840x2160 --display_fit fill
```

`testext` deduce la etiqueta del nombre de archivo, así que cada imagen debe llevar `_Real` o `_IA` en el nombre (por ejemplo, `Ave_1_Real.jpg` y `Ave_1_IA.png`). Los 40 estímulos del experimento están en [`/capturas`](capturas/) (`captura-01` a `captura-40`, en pares real/IA, en el orden de la sección [Capturas](#capturas)).

### Reproducir el entrenamiento completo

1. Descargar el conjunto de Kaggle y dejarlo en `src/fakeV2/` con esta estructura:

   ```
   src/fakeV2/
   ├── real/       # fotografías reales
   └── fake-v2/    # imágenes generadas por IA
   ```

2. Ejecutar, en orden:

   ```bash
   python ModeloClasificacion.py audit --data fakeV2 --out runs/nuevo           # auditoría de metadatos
   python ModeloClasificacion.py split --data fakeV2 --out runs/nuevo           # partición 80/10/10, semilla 42
   python ModeloClasificacion.py train --data fakeV2 --out runs/nuevo           # entrenamiento en dos fases
   python ModeloClasificacion.py eval  --out runs/nuevo --split test            # métricas de la prueba interna
   ```

Los hiperparámetros por defecto son los de la tesis (cuadro de hiperparámetros del capítulo de Metodología) y se pueden cambiar por línea de comandos, por ejemplo `--epochs 20` o `--lr 1e-4`.

## Capturas

Las 48 capturas están en la carpeta [`/capturas`](capturas/): los 40 estímulos del experimento y 8 resultados del modelo.

### Estímulos del experimento (capturas 1 a 40)

Las 40 imágenes que vieron los participantes en el Neurolab y que se usaron para evaluar el modelo: 20 fotografías reales y 20 imágenes generadas con ChatGPT, en pares del mismo animal (cinco categorías, cuatro pares por categoría).

| Par | Real | Generada por IA |
|---|---|---|
| Ave 1 | <img src="capturas/captura-01.jpg" width="260" alt="Captura 1"> <br> `captura-01.jpg` | <img src="capturas/captura-02.png" width="260" alt="Captura 2"> <br> `captura-02.png` |
| Ave 2 | <img src="capturas/captura-03.jpg" width="260" alt="Captura 3"> <br> `captura-03.jpg` | <img src="capturas/captura-04.png" width="260" alt="Captura 4"> <br> `captura-04.png` |
| Ave 3 | <img src="capturas/captura-05.jpg" width="260" alt="Captura 5"> <br> `captura-05.jpg` | <img src="capturas/captura-06.png" width="260" alt="Captura 6"> <br> `captura-06.png` |
| Ave 4 | <img src="capturas/captura-07.jpg" width="260" alt="Captura 7"> <br> `captura-07.jpg` | <img src="capturas/captura-08.png" width="260" alt="Captura 8"> <br> `captura-08.png` |
| Insecto 1 | <img src="capturas/captura-09.jpg" width="260" alt="Captura 9"> <br> `captura-09.jpg` | <img src="capturas/captura-10.png" width="260" alt="Captura 10"> <br> `captura-10.png` |
| Insecto 2 | <img src="capturas/captura-11.jpg" width="260" alt="Captura 11"> <br> `captura-11.jpg` | <img src="capturas/captura-12.png" width="260" alt="Captura 12"> <br> `captura-12.png` |
| Insecto 3 | <img src="capturas/captura-13.jpg" width="260" alt="Captura 13"> <br> `captura-13.jpg` | <img src="capturas/captura-14.png" width="260" alt="Captura 14"> <br> `captura-14.png` |
| Insecto 4 | <img src="capturas/captura-15.jpg" width="260" alt="Captura 15"> <br> `captura-15.jpg` | <img src="capturas/captura-16.png" width="260" alt="Captura 16"> <br> `captura-16.png` |
| Mamífero 1 | <img src="capturas/captura-17.jpg" width="260" alt="Captura 17"> <br> `captura-17.jpg` | <img src="capturas/captura-18.png" width="260" alt="Captura 18"> <br> `captura-18.png` |
| Mamífero 2 | <img src="capturas/captura-19.jpg" width="260" alt="Captura 19"> <br> `captura-19.jpg` | <img src="capturas/captura-20.png" width="260" alt="Captura 20"> <br> `captura-20.png` |
| Mamífero 3 | <img src="capturas/captura-21.jpg" width="260" alt="Captura 21"> <br> `captura-21.jpg` | <img src="capturas/captura-22.png" width="260" alt="Captura 22"> <br> `captura-22.png` |
| Mamífero 4 | <img src="capturas/captura-23.jpg" width="260" alt="Captura 23"> <br> `captura-23.jpg` | <img src="capturas/captura-24.png" width="260" alt="Captura 24"> <br> `captura-24.png` |
| Marino 1 | <img src="capturas/captura-25.jpg" width="260" alt="Captura 25"> <br> `captura-25.jpg` | <img src="capturas/captura-26.png" width="260" alt="Captura 26"> <br> `captura-26.png` |
| Marino 2 | <img src="capturas/captura-27.jpg" width="260" alt="Captura 27"> <br> `captura-27.jpg` | <img src="capturas/captura-28.png" width="260" alt="Captura 28"> <br> `captura-28.png` |
| Marino 3 | <img src="capturas/captura-29.jpg" width="260" alt="Captura 29"> <br> `captura-29.jpg` | <img src="capturas/captura-30.png" width="260" alt="Captura 30"> <br> `captura-30.png` |
| Marino 4 | <img src="capturas/captura-31.jpg" width="260" alt="Captura 31"> <br> `captura-31.jpg` | <img src="capturas/captura-32.png" width="260" alt="Captura 32"> <br> `captura-32.png` |
| Reptil 1 | <img src="capturas/captura-33.jpg" width="260" alt="Captura 33"> <br> `captura-33.jpg` | <img src="capturas/captura-34.png" width="260" alt="Captura 34"> <br> `captura-34.png` |
| Reptil 2 | <img src="capturas/captura-35.jpg" width="260" alt="Captura 35"> <br> `captura-35.jpg` | <img src="capturas/captura-36.png" width="260" alt="Captura 36"> <br> `captura-36.png` |
| Reptil 3 | <img src="capturas/captura-37.jpg" width="260" alt="Captura 37"> <br> `captura-37.jpg` | <img src="capturas/captura-38.png" width="260" alt="Captura 38"> <br> `captura-38.png` |
| Reptil 4 | <img src="capturas/captura-39.jpg" width="260" alt="Captura 39"> <br> `captura-39.jpg` | <img src="capturas/captura-40.png" width="260" alt="Captura 40"> <br> `captura-40.png` |

### Resultados del modelo (capturas 41 a 48)

| | |
|---|---|
| <img src="capturas/captura-41.png" width="400" alt="Captura 41"> <br> 41. Desempeño en la partición de prueba interna: a) curva ROC, b) matriz de confusión | <img src="capturas/captura-42.png" width="400" alt="Captura 42"> <br> 42. Desempeño en los 40 estímulos del experimento: a) curva ROC, b) matriz de confusión con umbral de 0.60 |
| <img src="capturas/captura-43.png" width="400" alt="Captura 43"> <br> 43. Probabilidad asignada por el clasificador frente a la respuesta de los participantes, por estímulo | <img src="capturas/captura-44.png" width="400" alt="Captura 44"> <br> 44. Exactitud de cada participante sobre los 40 estímulos |
| <img src="capturas/captura-45.jpg" width="400" alt="Captura 45"> <br> 45. Mapas sobre cuatro estímulos: a) densidad de fijaciones de los 20 participantes, b) Grad-CAM combinado, c) sensibilidad por oclusión | <img src="capturas/captura-46.png" width="400" alt="Captura 46"> <br> 46. NSS medio de las fijaciones humanas sobre cada mapa |
| <img src="capturas/captura-47.png" width="400" alt="Captura 47"> <br> 47. Salida de `eval` con las métricas de la partición de prueba interna | <img src="capturas/captura-48.png" width="400" alt="Captura 48"> <br> 48. Salida de `testext` con las métricas sobre los 40 estímulos |

## Demo

El video demostrativo, con narración, muestra el funcionamiento del modelo: su ejecución desde la terminal, las métricas que produce y los mapas de interpretabilidad sobre los estímulos del experimento.

▶️ [Ver video demostrativo (`demo/demo.mp4`)](demo/demo.mp4)

Formato MP4 (H.264 con audio AAC), resolución de 1512 × 982. Al abrir el enlace en GitHub se puede reproducir o descargar con el botón *Download raw file*.

## Documentación

- [Informe final del trabajo de graduación (`docs/informe_final.pdf`)](docs/informe_final.pdf): documento completo con marco teórico, metodología, resultados, discusión, conclusiones y anexos.
- [Consentimiento de uso de foto y video firmado (`docs/consentimiento_firmado.pdf`)](docs/consentimiento_firmado.pdf)

## Autores

| Nombre | Carné |
|---|---|
| Sebastián Juárez Payes | 21471 |

Asesora: M.Sc. Lynette García Pérez

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulte el archivo [LICENSE](LICENSE).
