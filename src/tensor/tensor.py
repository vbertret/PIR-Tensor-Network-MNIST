import numpy as np
import tensorly as tl

def contractMPS(W,Phi): 
    """
    Contraction of the MPS form with a picture mapped with the mapping function Phi

    Making all the contractions needed to compute the decision function.

    Parameters
    ---------
    W : list of numpy array
        a list containing all the tensors of the MPS form
    Phi : list of numpy array of shape (2)
        a list containing all the value of the mapping function for all the pixels of a picture

    Return
    -------
    res : numpy array of shape(diml)
        the value of the decision function
    """

    #Initialization of the final value with the contraction between the first tensor 
    #and the value of the mapping function from the first pixel
    res=tl.tenalg.mode_dot(W[0],Phi[:,0],0)

    #Be careful with different scenarios, that is for example is we have all the tensor
    #of the MPS form or just a subpart
    if(len(W)>1):
        if(len(W[0].shape)==2):
            res=tl.tenalg.contract(W[1],1,res,0)
        else:
            res=tl.tenalg.contract(W[1],1,res,1)
        res=tl.tenalg.mode_dot(res,Phi[:,1],0)

    #Do all the contractions that are left
    for i in range(2,len(W)):
        res=tl.tenalg.contract(W[i],1,res,0)  
        res=tl.tenalg.mode_dot(res,Phi[:,i],0)

    return(res)

if __name__ == "__main__":
    #test de la fonction contractMPS
    t1=np.array([[1,0],[0,4]])
    t2=np.array([[[1,0],[0,1]],[[2,0],[0,0]]])
    t3=np.array([[0,1],[2,3]])
    p1=np.array([1,0]) ; p2=np.array([0,2]) ; p3=np.array([1,1])
    W=[t1,t2,t3]
    phi=np.array([p1,p2,p3]).T
    print(contractMPS(W,phi))


