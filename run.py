from src.data.make_dataset import make_dataset_easy , make_dataset_random
from src.tensor_network.modelMPS import ModelMPS
from src.tensor.tensor import contractMPS

import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":

    np.random.seed(0)
    #Creation dataset
    N=10 ; nbExample=10 ; nbClass=2
    data,y = make_dataset_random(N,nbExample,nbClass)

    #Création du modèle
    A = ModelMPS(N,nbClass)
    A.choose_algo("DMRG")
    A.choose_optimizer("fixed")
    A.choose_loss_function("log-quadratic")
    A.onesInitialisation(5,0.20)

    #Entrainement du modèle
    errA=[]
    for epoch in range(9):
        errA += A.train(data,y,alpha=10**(-3),nmethod=2,Npass=2)
        print("Erreur : ", errA[-1])

    #Evaluation du modèle
    acc = A.accuracy(data,y)
    print("Accuracy : ", acc*100 , "%") 

    plt.plot(errA,label="Fonction de base")
    plt.legend()
    plt.show()