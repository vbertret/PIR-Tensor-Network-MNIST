import numpy as np
import math
def make_dataset_easy():

    Nt=8 #Nombre de training example
    data=np.zeros((Nt,3,3))

    #Carré blanc à gauche
    data[0]=np.array([[1,0,0],[1,0,0],[1,0,0]])
    data[1]=np.array([[1,0,0],[0,0,0],[0,0,0]])
    data[2]=np.array([[0,0,0],[1,0,0],[0,0,0]])
    data[3]=np.array([[0,0,0],[0,0,0],[1,0,0]])

    #Carré blanc à droite
    data[4]=np.array([[0,0,1],[0,0,1],[0,0,1]])
    data[5]=np.array([[0,0,1],[0,0,0],[0,0,0]])
    data[6]=np.array([[0,0,0],[0,0,1],[0,0,0]])
    data[7]=np.array([[0,0,0],[0,0,0],[0,0,1]])

    #Creation des labels
    y=np.array([[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1]])

    return (data,y)

def make_dataset_random(N,nbExample):
    ##### Creation Dataset #####

    middle=math.floor(nbExample/2)
    
    #Creation des images
    data=np.random.random_sample((nbExample,N))
    #Creation des labels
    y=np.zeros((nbExample,2)) ; y[0:middle,0] = 1 ; y[middle:nbExample,1]=1

    return (data,y)
