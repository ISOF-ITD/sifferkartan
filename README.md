# Modifiering av Sifferkartor
Denna repo innehåller pythonscript som justerar sifferkartor från [https://sifferkartan.isof.se/](https://sifferkartan.isof.se/).
Skripten används för att beskära och georefererara kartor.

# Användning
Pre-reqs: Python3; gdal, cv2(openCV) and numpy for python.

```python <script.py> <input-folder> <output-folder> (<json-input>)```

| input-folder | output-folder | json-input |
|---|---|---|
| Kartfiler i formatet jpg eller tif. | Här hamnar resultaten av skripten. | Används vid georeference, innehåller koordinater för kartfilerna i input-folder |


# Resultat
Exempel för hur det kan bli när man beskär en karta.

| input | output |
|---|---|
| ![output](/examples/4_13D1g.jpg) | ![output1](/examples/4_13D1g-out.jpg) |
