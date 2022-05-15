V1: Classificador que es caracteritza per tenir com a learning rate màxim del segon entrenament el valor: slice(1e-6, 1e-4)

	ACCURACY ENTRENAMENTS
		1r entrenament (100 epochs): acc 0.77
		2n entrenament (200 epochs): acc 0.79
	
	RESULTATS MATRIU DE CONFUSIÓ
		Nombre d’imatges de validació total: 100
		Total d’imatges effusion: 50
		Total d’imatges normal: 50
		Total prediccions effusion correctes: 42 (42/50 = accuracy effusion 0.84)
		Total prediccions normal correctes: 37 (37/50 = accuracy normal 0.74)
	

V2: Classificador que es caracteritza per tenir com a learning rate màxim del segon entrenament el valor: slice(lr_valley, lr_valley*10)

	ACCURACY ENTRENAMENTS
		1r entrenament (100 epochs): acc 0.76
	
	AQUESTA VERSIÓ HA QUEDAT DESCARTADA, PRINCIPALMENT, PERQUÈ M'HE ADONAT QUE UTILITZAR EL LEARNING RATE (lr_valley*10) PER A LES ÚLTIMES CAPES DE LA CNN  
	NO ÉS ADEQUAT PERQUÈ EL lr_find() obtingut mostra que a partir aproximadament del lr_valley, el loss va en ascens.
	
	
V3: Classificador que es caracteritza per tenir com a learning rate màxim del segon entrenament el valor: lr_valley

	ACCURACY ENTRENAMENTS
		1r entrenament (100 epochs): acc 0.76 (com de la v2 a la 3 només canvia el learning rate del segon entrenament, el 1r entrenament he fet que sigui igual en les 2 versions i per això he obtingut el mateix accuracy)
		2n entrenament (200 epochs): acc 0.75
	
	RESULTATS MATRIU DE CONFUSIÓ:
		Nombre d’imatges de validació total: 100
		Total d’imatges effusion: 50
		Total d’imatges normal: 50
		Total prediccions effusion correctes:  (39/50 = 0.78 accuracy effusion)
		Total prediccions normal correctes:  (36/50 = 0.72 accuracy normal)