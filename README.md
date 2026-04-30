# Modifiering av Sifferkartor
Denna repo innehåller olika pythonscript som justerar sifferkartor från [https://sifferkartan.isof.se/](https://sifferkartan.isof.se/).
De används för att beskära, georefererara och komprimera kartor.

# Användning
Generell användning
Pre-reqs: Python3; gdal, cv2(openCV) and numpy for python.

```python <script.py> <input-folder> <output-folder> (<json-input-eko-geo> <json-cutmaps>)```

| input-folder | output-folder | json-input-eko-geo | json-cutmaps |
|---|---|---|
| Kartfiler i formatet jpg eller tif. | Här hamnar resultaten av skripten. | Används vid georeference, innehåller koordinater för kartfilerna i input-folder | Används vid georeference av oklippta kartor, output från cutmaps.py som innehåller information om hörnen i kartorna |


# Resultat
Exempel för hur det kan bli när man beskär en karta.

| input | output |
|---|---|
| ![output](/examples/4_13D1g.jpg) | ![output1](/examples/4_13D1g-out.jpg) |
