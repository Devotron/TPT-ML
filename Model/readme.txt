
# readme.txt

Pour l'entrainement/test du modèle, il faut lancer le fichier script.py

Ayant modifier l'organisation du projet pour le rendu, 
il ce peut que certain chemin vers les datasets sont erronés (j'ai corrigé mais je préviens au cas où).

Les datasets :
 - full_dataset : dataset complet
 - partial : version édulcoré du dataset complet (200 images contre 5208) 
 à utiliser pour les machines ayant peu de mémoire vive.

 Experiments :
 - Test d'une autre implementation de réseau de neurone pour la colorisation
 - Pas de résultat concluant, je suppose que la raison pourrait être un manque d'entrainement 
 ( 100 epoch de 1 step sur un daraset de 400 images dure 3h30 sur ma machine sans compter le 
 fait que les checkpoints sont skipper/ignorer ) 