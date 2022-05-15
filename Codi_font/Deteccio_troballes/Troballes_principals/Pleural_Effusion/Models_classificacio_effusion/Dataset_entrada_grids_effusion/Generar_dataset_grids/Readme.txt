En la carpeta "Llibreries" es poden trobar els fitxers ".py" que necessiten utilitzar els notebooks
d'aquesta carpeta per generar els respectius datasets de quadrícules.

Els 3 notebooks generen datasets de quadrícules dels pulmons on pot haver "pleural effusion".

Notebook v1: genera un dataset de quadrícules on la quadrícula del pulmó esquerre està invertida per tal que totes les quadrícules estiguin orientades de la mateixa manera.
Per a cada imatge de pulmons del dataset original es guarden 2 imatges, 1 per a cada quadrícula, en el nou dataset. 

Notebook v2 (NO FUNCIONA): genera un dataset d'imatges de quadrícules combinades a partir de predir els bounding boxes dels pulmons
del dataset original del repositori de Github de referència, el qual conté imatges senceres de radiografia.

Notebook v3: genera un dataset d'imatges de quadrícules combinades a partir del dataset d'imatges de radiografia centrades en els pulmons.
