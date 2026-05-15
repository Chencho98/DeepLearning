# Proyecto CVAE HAM10000

Hugging Face Demo:
https://huggingface.co/spaces/Chencho98/DeepLearning

## Cómo correrlo en Colab


1. Abre `deepLearning.ipynb` en Google Colab.
2. Monta Google Drive cuando el notebook lo solicite.
3. Descarga el dataset desde el siguiente link compartido de Google Drive:

  https://drive.google.com/file/d/1gbZqsRYmzb-dfARSBqCnTP8aDU6aby5v/view?usp=drivesdk

4. Guarda `balanced_image.zip` dentro de:
   
   `/content/drive/MyDrive/`

6. Por defecto `RUN_TRAINING = False`, así el notebook carga el checkpoint ya guardado en Drive y genera una demo.
7. Si quieres reentrenar desde cero, cambia `RUN_TRAINING = True`.

## Estructura

- `model_bce_bn.py`: arquitectura del CVAE.
- `train_bce_bn.py`: entrenamiento.
- `deepLearning.ipynb`: notebook principal.
- `requirements.txt`: dependencias.

## Rutas esperadas

- Dataset ZIP: `/content/drive/MyDrive/balanced_image.zip`
- Resultados: `/content/drive/MyDrive/output_cvae_bce_bn_fp32/`
