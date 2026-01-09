# Moteur de recherche

Ce repository a été créé par : 
- Adrien Louis
- Timothée Robert
- Quentin Potiron

Il a pour but la mise en place d'un moteur de recherche dans le cadre d'un cours d'introduction à la data science.

## Installation

**Attention :** Il est nécessaire d'ajouter le fichier corpus.jsonl dans le dossier `data` comme celui-ci est trop lourd pour être partagé sur GitHub.


## Structure

Vous trouverez dans le fichier `approche_creuse.ipynb` un premier notebook explicitant notre première étude sur les données et la mise en place d'une approche creuse simple.

Dans le fichier `add_new_encoder.ipynb`, vous trouverez le fichier qui met en place de nouveaux encoders. Il est fortement déconseillé de lancer les fonctions comme celles-ci peuvent prendre énnormément de temps. Dans le reste de notre projet, nous utiliserons les encoders qui se trouvent dans le fichier `encoder`.

Le fichier `approche_dense.ipynb` s'occupe de la mise en place de l'approche dense avec l'ajout de nouveaux indicateurs afin d'évaluer la pertinence de notre moteur de recherche.

Enfin le fichier `approche_graph.ipynb` vient ajouter à l'approche dense des informations complémentaires avec l'étude d'un graphe de citation, d'un graphe d'auteurs et la prise en compte des citations et références dans les requêtes de notre moteur de recherche.
