import numpy as np
import tensorly as tl

def GD_creation_phi_tilde(A_tilde,Phi,sel,n,N,nbTraining):
    Phi_tilde1 = 0 ; Phi_tilde2 = 0 

    if(n==0 and sel == 0):
        GD_creation_phi_tilde.Phi_tilde1_table=[0]*nbTraining
        GD_creation_phi_tilde.Phi_tilde2_table=[[0 for i in range(N-1)] for j in range(nbTraining)]

    #Construction de phi_tilde1
    if(sel==1):
        GD_creation_phi_tilde.Phi_tilde1_table[n]=tl.tenalg.mode_dot(A_tilde[0],Phi[:,0],0) #contractMPS(A_tilde[0],Phi[:,0])
        Phi_tilde1=GD_creation_phi_tilde.Phi_tilde1_table[n]

    if(sel!=0 and sel!=1):
        GD_creation_phi_tilde.Phi_tilde1_table[n]=tl.tenalg.mode_dot(A_tilde[sel-1],GD_creation_phi_tilde.Phi_tilde1_table[n],1)
        GD_creation_phi_tilde.Phi_tilde1_table[n]=tl.tenalg.mode_dot(GD_creation_phi_tilde.Phi_tilde1_table[n],Phi[:,sel-1],0)
        Phi_tilde1=GD_creation_phi_tilde.Phi_tilde1_table[n]

        #Construction de phi_tilde2
    if(sel==0):
        GD_creation_phi_tilde.Phi_tilde2_table[n][N-2]=tl.tenalg.mode_dot(A_tilde[N-2],Phi[:,N-2],0)
        for i in range(N-3,-1,-1) :
            GD_creation_phi_tilde.Phi_tilde2_table[n][i]=tl.tenalg.contract(A_tilde[i],2,GD_creation_phi_tilde.Phi_tilde2_table[n][i+1],0)
            GD_creation_phi_tilde.Phi_tilde2_table[n][i]=tl.tenalg.mode_dot(GD_creation_phi_tilde.Phi_tilde2_table[n][i],Phi[:,i],0)
        Phi_tilde2=GD_creation_phi_tilde.Phi_tilde2_table[n][0]
        
    if(sel!=0  and sel!=(N-1)):
        Phi_tilde2=GD_creation_phi_tilde.Phi_tilde2_table[n][sel]

    return (Phi_tilde1, Phi_tilde2 )

def GD_calcul_cout_gradient(A,Phi_tilde1,Phi_tilde2,si,y,sel,N):
    ### Calcul Gradient et Cout ###
    if(sel==0):
        Phi_tilde=np.multiply.outer(si,Phi_tilde2) #Phi_tilde2(alpha1,l) => Phi_tilde(s1,alpha1,l)
        fl=tl.tenalg.contract(A,(0,1),Phi_tilde,(0,1))-y # => vecteur de taille l
        gradW=tl.tenalg.mode_dot(Phi_tilde,fl,2) # => gradW(s1,alpha1)
    elif(sel==N-1):
        Phi_tilde=np.multiply.outer(si,Phi_tilde1) #Phi_tilde1(alpha(N-1)) => Phi_tilde(sN,alpha(N-1))
        fl=tl.tenalg.contract(A,(0,1),Phi_tilde,(0,1))-y #=> vecteur de taille l
        gradW=np.multiply.outer(Phi_tilde,fl) # => gradW(sN,alpha(N-1),l)
    else:
        Phi_tilde=np.multiply.outer(si,Phi_tilde1) #=> Phi_tilde1(alpha(i-1)) => Phi_tilde1(si,alpha(i-1))
        Phi_tilde=np.multiply.outer(Phi_tilde,Phi_tilde2) # Phi_tilde2(alpha(i),l) => Phi_tilde(si,alpha(i-1),alpha(i),l)
        fl=tl.tenalg.contract(A,(0,1,2),Phi_tilde,(0,1,2))-y # => vecteur de taille l
        gradW=tl.tenalg.mode_dot(Phi_tilde,fl,3) # => gradW(si,alpha(i-1),alpha(i))
    cost = sum([i**2 for i in fl])

    return ( cost , gradW )

