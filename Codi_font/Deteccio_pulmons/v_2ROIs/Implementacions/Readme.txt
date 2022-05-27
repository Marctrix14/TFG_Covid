Model entrenat el 11/5/22

El model carrega les imatges de "train" i "valid" tal i com estan guardades 
en el dataset "subsetKaggleCOVID", és a dir, que no s'aplica, p.e. un RandomSplitter en el qual es processa un grup aleatori 
d'imatges com a "train" i un altre grup com a "valid".

ENTRENAMENT DEL MODEL

- 1r entrenament de 200 epochs després de congelar totes les capes del model CNN
excepte la última, és a dir, que la única capa entrenable és la última

Funció utilitzada per entrenar el model:
learn.fit_one_cycle(200, lr_max=lr_valley, div=12, pct_start=0.2) 


- 2n entrenament de 300 epochs després de descongelar totes les capes del model CNN
En aquest entrenament totes les capes són entrenables, és a dir, que s'actualitzen els weights (paràmetres del model)
en totes les capes.

Funció utilitzada per entrenar el model:
%time learn.fit_one_cycle(300, lr_max=slice(1e-5,1e-3)) 

Dades últim epoch #299:
train_loss: 0.05
valid_loss: 0.04
Accuracy: 0.83 

Com el valid_loss i el train_loss són gairebé idèntics es pot concloure que el model aprèn característiques genèriques
de les imatges en lloc de característiques únicament de les imatges d'entrenament, ja que sinó el valid_loss
seria molt més alt que el train_loss i es produiria  "Overfit", que és quan el model només aprèn característiques específiques pròpies
de les imatges d'entrenament.

En aquest model no es pot obtenir una matriu de confusió, ja que he fet que es predissin els bboxes sense classes.

TEST DEL MODEL

He pogut visualitzar els bounding boxes predits en les imatges de test set.
No he aconseguit mostrar l'accuracy / IoU (Intersection over Union) del test set ni el loss associat (veure última cel·la 
notebook "test_model...")



