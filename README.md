# TFG_Covid

## TREBALL DE FINAL DE GRAU

<br/>

Autor: Marc Padrós Jiménez

Tutors: Robert Martí i Xavier Lladó

Curs: 2022/23

DESCRIPCIÓ: Implementació d'algorismes per col·laborar en la detecció de COVID-19 a través de l'anàlisi d'imatges de radiografia de tòrax.

<br/>


Els algorismes implementats ordenats per data de creació són:

1. Algorisme de segmentació de pulmons a través de
generar quadrícules. 

2. Models CNN de detecció de pulmons. 

3.  Models CNN de classificació de vessament pleural / "pleural effusion".

<br/><br/>



### Models CNN de classificació de "pleural effusion"

* Classificador 1: dataset d'entrada conté imatges de radiografia de cossos humans  
    - [VERSIO VIGENT](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_troballes/Troballes_principals/Pleural_Effusion/Models_classificacio_effusion/Dataset_entrada_imatges_senceres/Meva_adaptacio_fastai/resnet50/v3/v2/Model_a_entrenar/effusion_classification_resnet50_v2.ipynb)
        - RESULTATS MODEL ENTRENAT:
            - 2 entrenaments (1r 50 epochs, 2n 100)
            - Accuracy darrer epoch 2n entrenament: 0.78 
            - Accuracy matriu de confusió "effusion": 42/50 (0.84)
            - Accuracy matriu de confusió "normal": 36/50 (0.72)

<br/>

* Classificador 2: dataset d'entrada conté imatges de pulmons 
    - [VERSIÓ VIGENT](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_troballes/Troballes_principals/Pleural_Effusion/Models_classificacio_effusion/Dataset_entrada_ROIS_pulmons/Classificador/v3/Model_a_entrenar/effusion_classification_ROIS_lungs_v3.ipynb) 
        - RESULTATS MODEL ENTRENAT:
            - 2 entrenaments (1r 100 epochs, 2n 200)
            - Accuracy darrer epoch 2n entrenament: 0.75
            - Accuracy matriu de confusió "effusion": 39/50 (0.78)
            - Accuracy matriu de confusió "normal": 36/50 (0.72)

<br/>

* Classificador 3: dataset d'entrada conté quadrícules on pot ubicar-se l'anomalia "pleural effusion"
    - [VERSIÓ VIGENT](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_troballes/Troballes_principals/Pleural_Effusion/Models_classificacio_effusion/Dataset_entrada_grids_effusion/Classificador/v2_gridsCombined/Model_v1/Model_a_entrenar/effusion_classif_grids_combined_v1.ipynb) 
        - RESULTATS MODEL ENTRENAT:
            - 2 entrenaments (1r 100 epochs, 2n 200)
            - Accuracy darrer epoch 2n entrenament: 0.85
            - Accuracy matriu de confusió "effusion": 47/50 (0.94)
            - Accuracy matriu de confusió "normal": 38/50 (0.76)


<br/><br/>


### Detecció de pulmons

* V1: 1 ROI (1 bounding box pels 2 pulmons)
    - [Versió original Robert](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_pulmons/v_1ROI/Implementacions/Original_Robert/roi_detection_Robert_fastaiv2.ipynb) 

    - Versió adaptada per mi: de moment no tinc cap versió funcional


* V2: 2 ROIs (1 bounding box per pulmó)  
    - [Versió vigent](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_pulmons/v_2ROIs/Implementacions/Implementacions_funcionals/11_5_22/model_lungs_detection_v3_discrimLr_local.ipynb)

<br/><br/>

### Segmentació de pulmons

- [VERSIÓ VIGENT](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Segmentacio_pulmons/v3/grid_image.ipynb) 








