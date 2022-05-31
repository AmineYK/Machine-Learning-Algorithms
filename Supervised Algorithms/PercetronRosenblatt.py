

# Import de packages externes
import numpy as np
import random as rd

import AbstractClassifier as AC

# Classifier Perceptron Rosenbalt
# Describe the Percetron of Rosenblatt
#---------------------------------------------------------------------------------------
class ClassifierPerceptron(AC.Classifier):

    def __init__(self, input_dimension, learning_rate, init=0):
		#initialise the input dimension and the learning rate
        self.input_dimension = input_dimension
        self.rate = learning_rate

		# intialise the weights of " W " to zeros if init = 0 
        w = np.zeros(self.input_dimension)

		# if init = 1 initalise the weights of " W " to random values between 0,1
        if init == 1:
            w = [2*(rd.randrange(0,1)-1)*0.001 for i in range(input_dimension)]

        self.w = w 
        
    # just a training step    
    def train_step(self, desc_set, label_set):

		# get a random order of indexes
        index = [i for i in range(len(label_set))]
        np.random.shuffle(index)

		# for any index 
        for i in index:
			# if the prediction is false
            if(self.score(desc_set[i]) * label_set[i] <= 0 ):
				# modify the " W "
                self.w = self.w + self.rate*label_set[i]*desc_set[i]
     
    def train(self, desc_set, label_set, niter_max=100, seuil=0.001):   
		# initialisations   
        b = True
        cpt = 0
        liste_norme = []
		# at least it finish with niter_max in order to avoid the infinity boucle
        while (cpt < niter_max) & b:
            w_before = self.w.copy()
            self.train_step(desc_set,label_set)
            cpt += 1
            norme_i = np.linalg.norm(self.w - w_before)
            liste_norme.append(norme_i)
			# try to get the convergence limit
            if norme_i < seuil:
                b = False
        
        return liste_norme   
            
    def score(self,x):
        return np.dot(x,self.w)
    
    def predict(self, x):
        if(self.score(x) < 0):
            return -1
        return 1
#---------------------------------------------------------------------------------------