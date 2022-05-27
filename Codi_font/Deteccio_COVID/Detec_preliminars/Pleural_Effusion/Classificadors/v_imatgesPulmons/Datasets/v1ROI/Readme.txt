El dataset "1ROI_original" l'ha creat el meu tutor Robert.
Ha generat imatges només de pulmons a través de predir els bounding boxes dels pulmons amb el seu algorisme de 
detecció de pulmons d'1 ROI. 
El dataset original, que conté imatges senceres en lloc de retallades dels pulmons, es troba dins de la ruta:
E:\Marc\Documents\GitHub\TFG_Covid\Codi_font\Deteccio_COVID\Detec_preliminars\Pleural_Effusion\Datasets_originals\GitReferencia

El dataset "1ROI_resized" és el dataset final, que faig servir d'entrada al classificador de "effusion" de Fastai.
Genero aquest dataset a través del script "split&resize_dataset" que divideix el dataset "1_ROI_original" en les carpetes train_val i test,
que és com necessita el classificador que estigui organitzat el dataset d'entrada.
A part el script també fa que totes les imatges del dataset d'entrada del classificador es guardin amb la mateixa mida (300x300), perquè sinó el
classificador no pot processar bé el dataset.