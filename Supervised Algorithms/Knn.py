

# Import de packages externes
import numpy as np
import pandas as pd
import math as mt
import random as rd
from numpy import linalg as la

import AbstractClassifier as AC

#  Classifier KNN
# K-Neirest Neighbors
# the class of point X is the same than the majority class of K-neirest points to X . 
#---------------------------------------------------------------------------------------------------
class ClassifierKNN(AC.Classifier):
    
    def __init__(self, input_dimension, k):
		# Initialise the input dimension and the number of neighbors
        self.input_dimension = input_dimension
        self.nombre_voisin = k
        
    def score(self,x):
        liste_distance = []
		# get distances between x and all other points 
        for elem in self.desc_set:
            liste_distance.append(np.sqrt(np.sum((x - elem)**2)))
        
		# get only the K neirest
        liste_k_proche = np.argsort(liste_distance)[0:self.nombre_voisin]   

		# determine the class of K-neirest points from x     
        cpt = 0
        for elem in liste_k_proche: 
            if self.label_set[elem] == 1:
                cpt += 1
          
        return 2* (cpt / self.nombre_voisin - 0.5) 
                
    
    def predict(self, x):
        if(self.score(x) >= 0):
            return 1 
        return -1

    def train(self, desc_set, label_set):  
		# No train in KNN    
        self.desc_set = desc_set
        self.label_set = label_set
#---------------------------------------------------------------------------------------------------
