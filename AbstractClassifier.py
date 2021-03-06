# -*- coding: utf-8 -*-


# Import de packages externes
import numpy as np
import pandas as pd
import math as mt
import random as rd
from numpy import linalg as la

# Class mere de tous les classifiers
#----------------------------------------------------------------------------------------------
class Classifier:
    def __init__(self, input_dimension):
        raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        cpt = 0
        for x in range(len(desc_set)):
            if(self.predict(desc_set[x])) == label_set[x]:
                cpt = cpt + 1
        
        return cpt / len(label_set)
#---------------------------------------------------------------------------------------------------       


# Classifier Perceptron Kernel
#---------------------------------------------------------------------------------------
class ClassifierPerceptronKernel(Classifier):

    def __init__(self, input_dimension, learning_rate, noyau, init=0):
        self.input_dimension = input_dimension
        self.rate = learning_rate
        w = np.zeros(noyau.output_dim)
        if init == 1:
            w = [2*(rd.randrange(0,1)-1)*0.001 for i in range(noyau.output_dim)]
        self.w = w 
        self.noyau = noyau
        
    def train_step(self, desc_set, label_set): 
        data_after = self.noyau.transform(desc_set)
        index = [i for i in range(len(label_set))]
        np.random.shuffle(index)
        for i in index:
            if(self.score(data_after[i]) * label_set[i] <= 0 ):
                self.w = self.w + self.rate*label_set[i]*data_after[i]
     
    def train(self, desc_set, label_set, niter_max=100, seuil=0.001):
        b = True
        cpt = 0
        liste_norme = []
        while (cpt < niter_max) & b:
            w_before = self.w.copy()
            self.train_step(desc_set,label_set)
            cpt += 1
            norme_i = np.linalg.norm(self.w - w_before)
            liste_norme.append(norme_i)
            if norme_i < seuil:
                b = False
        
        return liste_norme   
    
    def score(self,x):
        if len(x) == self.input_dimension:
            x = np.asarray([x])
            x = self.noyau.transform(x)[0]
            
        return np.dot(self.w, x)
    
    def predict(self, x):
        if(self.score(x) < 0):
            return -1
        return 1
#---------------------------------------------------------------------------------------

# Classifier Perceptron  Biais
#---------------------------------------------------------------------------------------

class ClassifierPerceptronBiais(Classifier):

    def __init__(self, input_dimension, learning_rate, init=0):

        self.input_dimension = input_dimension
        self.rate = learning_rate
        w = np.zeros(self.input_dimension)
        if init == 1:
            w = [2*(rd.randrange(0,1)-1)*0.001 for i in range(input_dimension)]
        self.w = w 
        self.allw = [w,]
        

    def get_allw(self):
        return self.allw

     
    def train_step(self, desc_set, label_set):
        index = [i for i in range(len(label_set))]
        np.random.shuffle(index)
        for i in index:
            if(self.score(desc_set[i]) * label_set[i] < 1 ):
                #self.w = self.w + self.rate*label_set[i]*desc_set[i]
                self.w = self.w + self.rate*(label_set[i] - self.score(desc_set[i])) * desc_set[i]
                self.allw.append(self.w)
    def train(self, desc_set, label_set, niter_max=100, seuil=0.001):      
        b = True
        cpt = 0
        liste_norme = []
        while (cpt < niter_max) & b:
            w_before = self.w.copy()
            self.train_step(desc_set,label_set)
            cpt += 1
            norme_i = np.linalg.norm(self.w - w_before)
            liste_norme.append(norme_i)
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

# CLASSIFIER MULTI OOA
#---------------------------------------------------------------------------------------

import copy
class ClassifierMultiOAA(Classifier):

    def __init__(self, classifier):
        self.classifiers = []
        self.classifier = classifier
    
    def train(self, desc_set, label_set):
        
        classes = np.unique(label_set)
        
        for i in range(len(classes)):
            label_mdf = [1 for i in range(len(label_set))]
            for j in range(len(label_set)):
                if(label_set[j] != classes[i]):
                    label_mdf[j] = -1  
            
            percep = copy.deepcopy(self.classifier)
            percep.train(desc_set,label_mdf)
            self.classifiers.append(percep)
    
    def accuracy(self, desc_set, label_set):
        yhat = np.array([self.predict(x) for x in desc_set])
        return np.where(label_set == yhat, 1., 0.).mean()
    
    
    def score(self,x):
        return [cla.score(x) for cla in self.classifiers]
    
    def predict(self, x):
        return np.argmax(self.score(x))




#---------------------------------------------------------------------------------------
# Classifier Adaline
# code de la classe pour le classifieur ADALINE


#??ATTENTION: contrairement ?? la classe ClassifierPerceptron, on n'utilise pas de m??thode train_step()
#??dans ce classifier, tout se fera dans train()


#TODO: Classe ?? Compl??ter
class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """
    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypoth??se : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.history = history
        self.niter_max = niter_max
        self.w = np.zeros(self.input_dimension)
        if(self.history == True):
            self.allw = [self.w,]
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donn??
            r??alise une it??ration sur l'ensemble des donn??es prises al??atoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypoth??se: desc_set et label_set ont le m??me nombre de lignes
        """        
        self.desc_set = desc_set
        self.label_set = label_set
        i = 0
        while (i<self.niter_max):
            #w_before = self.w.copy()
            grad = desc_set[i].T * (desc_set[i]*self.w - self.label_set[i])
            self.w = self.w - self.learning_rate*grad
            if(self.history == True):
                self.allw.append(self.w)
            #norme_i = np.linalg.norm(self.w - w_before)
              #if norme_i < 0.001:
                #b = False
            i = i + 1
            
        
        if(self.history == True):
            return self.allw
        
            
    
    def score(self,x):
        """ rend le score de pr??diction sur x (valeur r??elle)
            x: une description
        """
        return np.dot(x,self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if(self.score(x) < 0):
            return -1
        return 1
    
    def get_allw(self):
        return self.allw

#-------------------------------------------------------------------------------------------
# Classifier Adaline2

class ClassifierADALINE2(Classifier):
    """ Perceptron de ADALINE
    """
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypoth??se : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.w = np.zeros(self.input_dimension)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donn??
            r??alise une it??ration sur l'ensemble des donn??es prises al??atoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypoth??se: desc_set et label_set ont le m??me nombre de lignes
        """        
        self.desc_set = desc_set
        self.label_set = label_set
        self.w = np.linalg.solve(desc_set.T@desc_set,desc_set.T@label_set)
        
    
    def score(self,x):
        """ rend le score de pr??diction sur x (valeur r??elle)
            x: une description
        """
        return np.dot(x,self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if(self.score(x) < 0):
            return -1
        return 1
    
    def get_allw(self):
        return self.allw
    
    
#----------------------------------------------------------------------------------------------
# Classe Majoritaire
def classe_majoritaire(Y):
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    return valeurs[np.argmax(nb_fois)]
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Shannon
import math as mt
def shannon(P):
    """ list[Number] -> float
        Hypoth??se: la somme des nombres de P vaut 1
        P correspond ?? une distribution de probabilit??
        rend la valeur de l'entropie de Shannon correspondante
    """
    if(len(P) > 1):
        p2 = [elem*mt.log(elem) for elem in P if elem != 0]
        return -1 * np.sum(p2)
    return 0
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Entropie
def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    list_shanon = [nb_fois[elem]/sum(nb_fois) for elem in range(len(valeurs))]
    return (shannon(list_shanon))
#----------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------
# Construire_Ad
import sys 

def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le crit??re d'arr??t 
        LNoms : liste des noms de features (colonnes) de description 
    """
    # dimensions de X:
    (nb_lig, nb_col) = X.shape
    
    entropie_classe = entropie(Y)
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on cr??e une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = sys.float_info.min  #??meilleur gain trouv?? (initalis?? ?? -infinie)
        i_best = -1         #??num??ro du meilleur attribut
        Xbest_valeurs = None
        
        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le num??ro de l'attribut qui maximise le gain d'information.  En cas d'??galit??,
        #          le premier rencontr?? est choisi.
        # gain_max : la plus grande valeur de gain d'information trouv??e.
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc n??cessaire ici de parcourir tous les attributs et de calculer
        # la valeur du gain d'information pour chaque attribut.
        
        ##################
        lessc = []
        
        for ind_attribut in range(X.shape[1]):
            sc = 0
            attribut_valeur, attribut_nb_occu = np.unique(X[:, ind_attribut], return_counts = True)
            for elem in attribut_valeur :
                sc = sc + entropie(Y[X[:, ind_attribut] == elem]) *(len(attribut_valeur)/len(Y))
            lessc.append(sc)
        
       
        
        i_best = np.argmin(lessc)
        gain_max = entropie(Y) - lessc[i_best]
        Xbest_valeurs = np.unique(X[:, i_best])
        
        ##################
        
        #############
        
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Noeud Categoriel
# La librairie suivante est n??cessaire pour l'affichage graphique de l'arbre:
import graphviz as gv

# Pour plus de d??tails : https://graphviz.readthedocs.io/en/stable/manual.html

# Eventuellement, il peut ??tre n??cessaire d'installer graphviz sur votre compte:
# pip install --user --install-option="--prefix=" -U graphviz

class NoeudCategoriel:
    """ Classe pour repr??senter des noeuds d'un arbre de d??cision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le num??ro de l'attribut auquel il se rapporte: de 0 ?? ...
              si le noeud se rapporte ?? la classe, le num??ro est -1, on n'a pas besoin
              de le pr??ciser
            - nom (str) : une cha??ne de caract??res donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donn?? de fa??on 
              g??n??rique: "att_Num??ro")
        """
        self.attribut = num_att    # num??ro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.Les_fils = None       # aucun fils ?? la cr??ation, ils seront ajout??s
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit ??tre associ??e ?? Fils
                     le type de cette valeur d??pend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stock??s sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contr??le, la nouvelle association peut
        # ??craser une association existante.
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour ??tre s??r
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en g??n??ral)
            on rend la valeur 0 si l'exemple ne peut pas ??tre class?? (cf. les questions
            pos??es en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente r??cursive dans le noeud associ?? ?? la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de r??soudre ce myst??re...            
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return 0
    
    def to_graph(self, g, prefixe='A'):
        """ construit une repr??sentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous int??ressera pas plus que ??a, elle ne sera donc pas expliqu??e            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g

#----------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------
# Classifier Arbre Decision
class ClassifierArbreDecision(Classifier):
    """ Classe pour repr??senter un classifieur par arbre de d??cision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : param??tre de l'algorithme (cf. explications pr??c??dentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypoth??se : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipul?? par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses param??tres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donn??
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypoth??se: desc_set et label_set ont le m??me nombre de lignes
        """        
        ##################
        self.racine = construit_AD(desc_set,label_set,self.epsilon,self.LNoms)
        ##################
    
    def score(self,x):
        """ rend le score de pr??diction sur x (valeur r??elle)
            x: une description
        """
        # cette m??thode ne fait rien dans notre impl??mentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        ##################
        return self.racine.classifie(x)
            
        ##################

    def accuracy(self, desc_set, label_set):  # Version propre ?? aux arbres
        """ Permet de calculer la qualit?? du syst??me sur un dataset donn??
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypoth??se: desc_set et label_set ont le m??me nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)
        
#--------------------------------------------------------------------------------------------------




