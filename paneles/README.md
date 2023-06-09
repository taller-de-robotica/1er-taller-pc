# Instalar dependencias

```
pip3 install --upgrade pip
pip3 install tensorflow==2.10.0
pip3 install patchify
pip3 install segmentation_models
```

# Pesos
Los pesos para la red preentrenada se pueden descargar de:
[sm_unet4_03.hdf5](https://quetzalcoatl.fciencias.unam.mx/taller/1erTaller/sm_unet4_03.hdf5)

# Ejecutar prueba

Para verificar rápidamente el funcionamiento del código, se analizará la imagen ```IMG_20221012_13227_DRO.png``` que se incluye aquí como muestra.  Para ello ejecutar:
```
cd 1er-taller-pc/paneles/
python3 poly_dust_detector.py
```

Para analizar las imágenes enviadas por la raspberry a través de la red se pasa como parámetro extra su dirección ip, por ejemplo:
```
python3 poly_dust_detector.py -ip 192.168.16.107
```