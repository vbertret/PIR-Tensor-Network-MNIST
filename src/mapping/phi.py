import numpy as np

#Fonction de plongement
def phi(x):
	return np.array([np.cos((np.pi*x)/2),np.sin((np.pi*x)/2)])

#Fonction de plongement sur toute une image
def Phi(img):
    return phi(img.reshape(-1,))
    
    

