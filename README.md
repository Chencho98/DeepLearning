# Proyecto CVAE HAM10000

## Cómo correrlo en Colab

1. Sube `balanced_image.zip` a tu Google Drive en `MyDrive/`.
2. Abre `deepLearning.ipynb` en Colab.
3. Ejecuta las celdas en orden.
4. Por defecto `RUN_TRAINING = False`, así el notebook carga el checkpoint ya guardado en Drive y genera una demo.
5. Si quieres reentrenar desde cero, cambia `RUN_TRAINING = True`.

## Estructura

- `model_bce_bn.py`: arquitectura del CVAE.
- `train_bce_bn.py`: entrenamiento.
- `deepLearning.ipynb`: notebook principal.
- `requirements.txt`: dependencias.

## Rutas esperadas

- Dataset ZIP: `/content/drive/MyDrive/balanced_image.zip`
- Resultados: `/content/drive/MyDrive/output_cvae_bce_bn_fp32/`
