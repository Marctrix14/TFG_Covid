v1: Quadricular imatges de radiografia prenent de referència que la columna estigui al centre

v2: Quadricular imatges de radiografia segons els bounding boxes dels pulmons.
Aquesta implementació l'he combinada amb el model de detecció de pulmons que he implementat de 2 ROIs (1 bbox per pulmó).

v3: Quadricular imatges centrades en els pulmons. 
Aquesta implementació l'he realitzada fent servir com a imatges de test el dataset 
que em va passar en Robert d'imatges de pulmons, que va generar
a través del seu algoritme d'1 ROI (1 bbox pels 2 pulmons) amb el qual va predir les ROIs dels pulmons.
El dataset que em va passar processat només amb imatges de pulmons prové del dataset d'imatges senceres de radiografia,
que vaig obtenir del repositori de Github de referència per a la classificació de "pleural effusion".  
