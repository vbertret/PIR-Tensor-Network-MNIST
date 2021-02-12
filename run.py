from src.data.make_dataset import make_dataset_easy , make_dataset_random
from src.models.tensor_network.modelMPS import ModelMPS
from src.models.tensor_network.tensor.tensor import contractMPS

import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":

    #Creation dataset
    N=30 ; nbExample=100 ; nbClass=10
    data,y = make_dataset_random(N,nbExample,nbClass)

    #Création du modèle
    A = ModelMPS(N,nbClass)
    A.algo("DMRG")
    A.normalInitialisation(5,0.33)

    #Entrainement du modèle
    err=[]
    for epoch in range(20):
        if(epoch==0):
            alpha=0
        else:
            alpha=9*10**(0)
            err += A.train(data,y,alpha)
            print(err[-1])

    #Evaluation du modèle
    acc = A.accuracy(data,y)
    print("Accuracy : ", acc*100 , "%") 

    plt.plot(err)
    plt.show()