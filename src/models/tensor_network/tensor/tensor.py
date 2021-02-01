import numpy as np
import tensorly as tl

#Contraction totale d'un tenseur W sous forme MPS avec une fonction de plongement Phi
def contractMPS(W,Phi): 
    res=tl.tenalg.mode_dot(W[0],Phi[:,0],0)
    if(len(W)>1):
        #on gère le cas du premier tenseur de la décomposition qui peut être d'ordre 2 ou 3 
        #( jamais l sur le premier tenseur )
        if(len(W[0].shape)==2):
            res=tl.tenalg.contract(W[1],1,res,0)
        else:
            res=tl.tenalg.contract(W[1],1,res,1)
        res=tl.tenalg.mode_dot(res,Phi[:,1],0)

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


