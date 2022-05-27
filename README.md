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

<br/>



### Models CNN de classificació de "pleural effusion"

#### CONFIGURACIONS DE TOTS ELS CLASSIFICADORS: 

Train: 800 imatges (400 normal / 400 effusion)

Validació: 100 imatges (50 normal / 50 effusion)

Test: 100 imatges (50 normal / 50 effusion)

#epochs 1r entrenament (després de congelar totes les capes excepte la última): 100

#epochs 2n entrenament (després de descongelar totes les capes): 200

Arquitectura: ResNet50

<br/>

PENDENT POSAR TAULA RESULTATS

* Classificador 1: dataset d'entrada conté imatges de radiografia de tòrax
    - [Versió entrenament](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_COVID/Detec_preliminars/Pleural_Effusion/Classificadors/v_imatgesTorax/Classificador/Meva_adaptacio_fastai/resnet50/Entrenar_model/train_effusion_classif_cossos_resnet50.ipynb)
    - [Versió test](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_COVID/Detec_preliminars/Pleural_Effusion/Classificadors/v_imatgesTorax/Classificador/Meva_adaptacio_fastai/resnet50/Testejar_model/test_eff_classif_cossos_resnet50.ipynb) 
       

<br/>

* Classificador 2: dataset d'entrada conté imatges de pulmons 
    - [Versió entrenament](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_COVID/Detec_preliminars/Pleural_Effusion/Classificadors/v_imatgesPulmons/Classificador/v_2ROIs/Entrenar_model/train_effusion_classification_ROIS_lungs_v3.ipynb) 
    - [Versió test](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_COVID/Detec_preliminars/Pleural_Effusion/Classificadors/v_imatgesPulmons/Classificador/v_2ROIs/Testejar_model/test_eff_classif_ROIS_lungs_v3.ipynb)  

<br/>

* Classificador 3: dataset d'entrada conté quadrícules on pot ubicar-se l'anomalia "pleural effusion" (v1ROI)
    - [Versió entrenament](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_COVID/Detec_preliminars/Pleural_Effusion/Classificadors/v_imatgesGridsEffusion/Classificador/v1ROI/Entrenar_model/train_effusion_classif_grids_combined_v3.ipynb) 
    - [Versió test](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_COVID/Detec_preliminars/Pleural_Effusion/Classificadors/v_imatgesGridsEffusion/Classificador/v1ROI/Testejar_model/test_effusion_classif_grids_combined_v3.ipynb) 
      

<br/><br/>


### Detecció de pulmons

* V1: 1 ROI (1 bounding box pels 2 pulmons)
    - [Versió original Robert](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_pulmons/v_1ROI/Implementacions/Original_Robert/roi_detection_Robert_fastaiv2.ipynb) 

    - De les versions adaptades per mi, no tinc cap de funcional totalment. 


* V2: 2 ROIs (1 bounding box per pulmó)  
    - [Versió entrenament](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_pulmons/v_2ROIs/Implementacions/Entrenar_model/train_model_lungs_detection_v3_discrimLr_local.ipynb)
    - [Versió test](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_pulmons/v_2ROIs/Implementacions/Testejar_model/kaggleCOVID/test_lungs_detector_kaggleCOVID.ipynb)

<br/><br/>

### Segmentació de pulmons

- [VERSIÓ VIGENT](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Segmentacio_pulmons/v3/grid_image.ipynb) 








