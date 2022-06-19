# PROJECTE FI DE GRAU


## DESCRIPCIÓ

<p> Anàlisi i processament d'imatges de radiografia de tòrax per ajudar en la detecció del COVID-19 i de malaties pulmonars.</p>


![Imatge del projecte](/Imatges/imatgeProjecte.jpeg "Imatge del projecte")

<p> S'han realitzat algorismes d'intel·ligència artificial basats en xarxes neuronals convolucionals, anomenades en anglès Convolutional Neural Networks o CNNs. </p>


<br/>

<p> Els algorismes implementats són:

1. Models CNN de detecció de pulmons. 

2. Algorismes de divisió dels pulmons en quadrícules. 

3.  Models CNN de classificació d'imatges de radiografia segons si contenen o no embassament pleural* / "pleural effusion". </p>

<br/>

<p> * L'embassament pleural és una acumulació excessiva de líquid que es situa en l'espai pleural, que és una regió entre el pulmó i la membrana externa que el cobreix. </p>

![Ubicació embassament pleural](/Imatges/DefinicioEmbassament/embassamentUbicacio.jpg "Ubicació embassament pleural")

<p> El mètode més utilitzat per detectar l'embassament pleural és realitzar una radiografia de tòrax. En una radiografia l'embassament es representa com a regions de color blanc.</p>  

![Detecció embassament pleural radiografia de tòrax](/Imatges/DefinicioEmbassament/embassamentDeteccioRx.JPG "Detecció embassament pleural radiografia de tòrax")

<p> Si es detecta embassament pleural es pot confirmar que un pacient no tindrà COVID-19.</p>

<<<<<<< HEAD

=======
>>>>>>> 363d859b1edd0fbce097bef020865e74681d968b
<br/>


## IMPLEMENTACIONS 

<br/>

<<<<<<< HEAD
### 1. DETECCIÓ DE PULMONS

<br/>

![Detecció embassament pleural radiografia de tòrax](/Imatges/Implementacions/DeteccioPulmons/versionsROIs.JPG "Detecció embassament pleural radiografia de tòrax")

* VERSIÓ 1: 1 requadre / bounding box pels 2 pulmons
    - [Versió original](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_pulmons/v_1ROI/Implementacions/Original_Robert/roi_detection_Robert_fastaiv2.ipynb) 

    - [Versió adaptada](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_pulmons/v_1ROI/Implementacions/Meva_adaptacio/v1/model_lungs_detection_1ROI_ambClasses.ipynb) 


* VERSIÓ 2: 1 requadre / bounding box per pulmó
    - [Versió vigent](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_pulmons/v_2ROIs/Implementacions/Entrenar_model/train_model_2ROIs_lungs_detection_v3_discrimLr_local.ipynb)

![Precisió detecció de pulmons](/Imatges/Implementacions/DeteccioPulmons/precisio.png "Precisió detecció de pulmons")


<br/>

### 2. DIVISIÓ DELS PULMONS

<br/>

![Proves de divisó dels pulmons](/Imatges/Implementacions/DivisioPulmons/resultatsDivisio.png "Proves de divisió dels pulmons")

- [VERSIÓ VIGENT](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Divisio_pulmons/v2/grid_bboxes.ipynb) 
=======
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
      
>>>>>>> 363d859b1edd0fbce097bef020865e74681d968b

<br/><br/>


### 3. CLASSIFICACIÓ D'IMATGES DE RADIOGRAFIA PER EMBASSAMENT PLEURAL

<br/>

![CNN classificació d'imatges](/Imatges/Implementacions/Classificacio/esquemaCNNclassif.png "CNN classificació d'imatges")

<<<<<<< HEAD
![Versions classificació](/Imatges/Implementacions/Classificacio/esquemes_classif_effusion.JPG "Versions classificació")


=======
    - De les versions adaptades per mi, no tinc cap de funcional totalment. 


* V2: 2 ROIs (1 bounding box per pulmó)  
    - [Versió entrenament](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_pulmons/v_2ROIs/Implementacions/Entrenar_model/train_model_lungs_detection_v3_discrimLr_local.ipynb)
    - [Versió test](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_pulmons/v_2ROIs/Implementacions/Testejar_model/kaggleCOVID/test_lungs_detector_kaggleCOVID.ipynb)
>>>>>>> 363d859b1edd0fbce097bef020865e74681d968b

* Classificador 1: dataset d'entrada conté imatges de radiografia de tòrax
    - [Versió entrenament](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_COVID/Detec_preliminars/Pleural_Effusion/Classificadors/v_imatgesTorax/Classificador/Meva_adaptacio_fastai/resnet50/Entrenar_model/train_effusion_classif_torax_resnet50_ds_git.ipynb)
    - [Versió test](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_COVID/Detec_preliminars/Pleural_Effusion/Classificadors/v_imatgesTorax/Classificador/Meva_adaptacio_fastai/resnet50/Testejar_model/test_eff_classif_torax_resnet50.ipynb) 
       

<br/>

* Classificador 2: dataset d'entrada conté imatges de pulmons 
    - [Versió entrenament](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_COVID/Detec_preliminars/Pleural_Effusion/Classificadors/v_imatgesPulmons/Classificador/v_1ROI/Entrenar_model/train_effusion_classification_ROIS_lungs_v3.ipynb) 
    - [Versió test](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_COVID/Detec_preliminars/Pleural_Effusion/Classificadors/v_imatgesPulmons/Classificador/v_1ROI/Testejar_model/test_eff_classif_ROIS_lungs_v3.ipynb)  

<br/>

* Classificador 3: dataset d'entrada conté imatges de les regions dels pulmons on pot haver embassament pleural
    - [Versió entrenament](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_COVID/Detec_preliminars/Pleural_Effusion/Classificadors/v_imatgesGridsEffusion/Classificador/v2ROIs/Entrenar_model/train_effusion_classif_grids_combined_2ROIs.ipynb) 
    - [Versió test](https://github.com/Marctrix14/TFG_Covid/blob/main/Codi_font/Deteccio_COVID/Detec_preliminars/Pleural_Effusion/Classificadors/v_imatgesGridsEffusion/Classificador/v2ROIs/Testejar_model/test_effusion_classif_grids_combined_v3.ipynb) 
  
![Precisió classificació d'imatges](/Imatges/Implementacions/Classificacio/precisio.png "Precisió classificació d'imatges")








