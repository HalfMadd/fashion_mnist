# Fashion MNIST Machine Learning

## Installation
Les librairies nécessaires sont dans le fichier requirements.txt et s'installent dans le projet via la commande :

`pip install -r requirements.txt --no index`

( Installer les modules manuellement en cas d'échec avec `pip install [nom_module]` ) 

Le csv d'entrainement du modèle est disponible ici : https://www.kaggle.com/zalando-research/fashionmnist?select=fashion-mnist_train.csv
Il est supposé comme placé à la racine du projet

## Execution
Deux méthodes pour lancer le serveur :
 - `python app.py`
 - `export FLASK_APP=/route/vers/app.py && flask run`
 
## Utilisation

### Organisation

Pages Web: 
- Adresse de base: http://127.0.0.1:5000
- Routes: 
 - / : Accueil, le formulaire où renseigner l'url d'une image
 - /prediction : le résultat du formulaire

### Prédictions
Pour obtenir une prédiction et une réponse en retour, faire une requête **POST** sur l'adresse 127.0.0.1:5000/prediction avec pour donnée { 'url':' [url]' } 



