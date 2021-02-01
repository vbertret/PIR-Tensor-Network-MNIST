from src.data.make_dataset import make_dataset_easy , make_dataset_random
from src.models.tensor_network.modelMPS import ModelMPS

import numpy as np
import matplotlib.pyplot as plt

#Creation dataset
N=50 ; nbExample=30
data,y = make_dataset_random(N,nbExample)

#Création du modèle
A = ModelMPS(N,2)
A.algo("DMRG")
A.normalInitialisation(5,0.5)

#Entrainement du modèle
err=[]
for epoch in range(20):
    if(epoch==0):
        alpha=0
    else:
        alpha=5*10**(-1)
    err += A.train(data,y,alpha)
    print(err[-1])

#Evaluation du modèle
prediction=A.predict(data)
cpt=0
for example in range(len(prediction)):
    #print("Prediction : ", np.around(prediction[example],decimals=2), "True Label : ", y[example])
    if(np.argmax(prediction[example])==np.argmax(y[example])):
        cpt+=1
print("Accuracy : ", cpt/nbExample*100 , "%")

plt.plot(err)
plt.show()