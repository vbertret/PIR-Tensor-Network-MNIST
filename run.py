from src.data.make_dataset import make_dataset_easy , make_dataset_random , load_subpart_MNIST_dataset_small , convert_one_hot_encoding
from src.tensor_network.modelMPS import ModelMPS
from src.tensor.tensor import contractMPS

import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":

    #Creation dataset
    N=196 ; nbExample=10 ; nbClass=2
    digits=[0,1]
    path="data"
    train_data , test_data , train_labels , test_labels = load_subpart_MNIST_dataset_small(path,digits)

    #Plus petit Ensemble
    nbExample=10
    train_data = train_data[0:nbExample]
    train_labels = train_labels[0:nbExample]

    test_data = test_data[0:nbExample]
    test_labels = test_labels[0:nbExample]

    train_labels = convert_one_hot_encoding(train_labels)
    test_labels = convert_one_hot_encoding(test_labels)

    #Création du modèle
    A = ModelMPS(N,nbClass)
    A.choose_algo("DMRG")
    A.choose_optimizer("CG")
    A.choose_loss_function("quadratic")
    A.onesInitialisation(5,data=train_data,goal=1)

    #Entrainement du modèle
    err=[]
    for epoch in range(1):
        err += A.train(train_data,train_labels,nm)
        print(err[-1])

    #Evaluation du modèle
    acc = A.accuracy(train_data,train_labels)

    print("Accuracy : ", acc*100 , "%") 

    plt.plot(np.log(err))
    plt.show()