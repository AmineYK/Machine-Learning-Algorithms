
# Import de packages externes
import numpy as np
from numpy import linalg as la

import AbstractClassifier as AC


#  Classifier Random Lineaire
# Get random values for the decision boundary " W " 
#---------------------------------------------------------------------------------------------------
class ClassifierLineaireRandom(AC.Classifier):

    def __init__(self, input_dimension):
        w = np.random.uniform(-1 , 1 , input_dimension)
        self.w = w / la.norm(w)

		
    def train(self, desc_set, label_set):
        print("No training !")
    
    def score(self,x):
        return np.dot(x,self.w)

    
    def predict(self, x):
        if(self.score(x) >= 0):
            return 1
        else:
            return -1
#---------------------------------------------------------------------------------------