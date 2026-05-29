# Modifiering av Sifferkartor
Denna repo innehåller olika pythonscript som justerar sifferkartor från [https://sifferkartan.isof.se/](https://sifferkartan.isof.se/).
De används för att beskära, georefererara och komprimera kartor.

# Användning
Pre-reqs: Python3; gdal, cv2(openCV) and numpy for python.

Generell användning

```python <script.py> <input-folder> <output-folder> (<json-input-eko-geo> <json-cutmaps>)```

| input-folder | output-folder | json-input-eko-geo | json-cutmaps |
|---|---|---|---|
| Kartfiler i formatet jpg eller tif. | Här hamnar resultaten av skripten. | Används vid georeference, innehåller koordinater för kartfilerna i input-folder | Används vid georeference av oklippta kartor, output från cutmaps.py som innehåller information om hörnen i kartorna |

| Skript-namn | Funktion |
|---|---|
| batch_rename.py | Byter namn på alla filer i input-folder till xxx_modified_auto.tiff |
| compress-quality25.py | För komprimering vid användning för GeoServer ImageMosaic när man behöver klippa filerna med GIMP |
| compressmaps.py | Komprimerar kartfiler till jpeg in tiff, 25% quality |
| cutmaps.py | Beskär ekonomiska kartor för använding i rastermosaik |
| find-handwriting.py | Ett test att automatiskt hitta siffror på kartor, i dagsläget hittar den endast röda siffror |
| georeference.py | Automatisk referering av alla kartor i input-folder, kan användas för ekonomiska kartor men troligen andra också vid lite omskrivning. | 

# Resultat
Exempel för hur det kan bli när man beskär en karta.

| input | output |
|---|---|
| ![output](/examples/4_13D1g.jpg) | ![output1](/examples/4_13D1g-out.jpg) |
