Les dues implementacions amb millor % d'accuracy de classificació s'anomenen "pleural_effusion_classification_resnet50_ds_git_valid_20pct" i 
"pleural_effusion_classification_resnet34_ds_git".

La ruta de la primera implementació citada (des del directori d'aquest Readme) és: "resnet50/v2_fixant_learning_rate".
La ruta de la segona implementació és "resnet34/".

S'ha de tenir en compte que en totes les implementacions que he fet dins de la carpeta "Meva_adaptacio_Fastai" 
s'aplica un RandomSpliter, que fa que en cada execució nova del notebook corresponent es faci un repartiment aleatori de les imatges en train/validation.
Llavors, no es té un control de quines imatges són de train i quines de validation i això fa que els resultats d'accuracy no siguin fiables
perquè en cada execució nova es canvien les imatges d'entrada de train/validation i les implementacions no es poden comparar perquè els repartiments en train/val
són diferents. 