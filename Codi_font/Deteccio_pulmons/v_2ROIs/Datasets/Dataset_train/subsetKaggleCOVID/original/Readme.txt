Aquest dataset ha estat generat a través d'un script que vaig crear amb el qual he realitzat la següent distribució:

TRAIN

- 300 imatges 
- 300 anotacions (*.txt)

VALID

- 30 imatges
- 30 anotacions

TEST

- 30 imatges
No hi ha anotacions en el test set, ja que no són imatges per testejar el model, per analitzar com prediu de bé
les coordenades dels bboxes, sense disposar de les coordenades reals/targets. 


NOTA: A través d'implementar el Python Notebook "compare_sets_content" he comprovat que les imatges d'un set, només pertanyen a aquell set. 
No hi ha imatges duplicades en sets diferents. 