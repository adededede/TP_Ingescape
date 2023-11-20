# TP_Ingescape
Realisation pour le tp sandbox ingescape (avec le whiteboard) : ~~Génération d'image via une IA~~ Utilisation d'url d'images puis affichage d'image via chat selon les couleurs, le nombre de tableaux  mais aussi l'action indiqué.

## Commande pour lancer le projet :
> (attention il faut etre placer dans le dossier Ingescape/sandbox)

> lancer aussi la plateforme fournis dans le dossier plateform pour ainsi pouvoir communiquer via le writer dans la chat
```
.\Whiteboard\Whiteboard\Whiteboard.exe --device Wi-Fi --port 15670
```
```
python .\Chat\src\main.py Chat --device Wi-Fi --port 15670
```
```
python .\IA_tableau\src\main.py IA_tableau --device Wi-Fi --port 15670 
```

### Il faut modifier les chemins dans le code du generateur
> pas nécessaire vu que dans cette dernière version on ne génère plus les images via IA
