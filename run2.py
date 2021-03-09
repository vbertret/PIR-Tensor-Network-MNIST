from src.data.make_dataset import load_MNIST_dataset_small , convert_one_hot_encoding
from src.models.tensor_network.modelMPS import ModelMPS
from src.models.tensor_network.tensor.tensor import contractMPS
from src.models.tensor_network.mapping.phi import phi

import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":

    #Creation dataset
    train_data , test_data , train_labels , test_labels = load_MNIST_dataset_small()
    train_labels = convert_one_hot_encoding(train_labels)
    test_labels = convert_one_hot_encoding(test_labels)
    N=train_data.shape[1]*train_data.shape[2]

    #Plus petit Ensemble
    train_data = train_data[0:100]
    train_labels = train_labels[0:100]

    test_data = test_data[0:100]
    test_labels = test_labels[0:100]
    #Création du modèle
    A = ModelMPS(N,10)
    A.algo("DMRG")
    A.normalInitialisation(5,0.33)

    #Calcul première valeur
    """img=train_data[0].reshape(-1,)
    Phi=phi(img)
    z=contractMPS(A.W,Phi)
    print(z,end=";")"""
    
    #Entrainement du modèle
    err=[]
    for epoch in range(8):
        err +=  A.trainDMRG_test(train_data,train_labels,0.3)  # A.train(train_data,train_labels,alpha)
        print(f"Erreur apprentissage : {err[-1]}.")

        #Evaluation du modèle
        acc = A.accuracy(test_data,test_labels)
        print("Taux de bonne classification ( test ) : ", acc*100 , "%") 



    plt.plot(err)
    plt.show()

