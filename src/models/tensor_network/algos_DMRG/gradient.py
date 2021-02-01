import numpy as np
import tensorly as tl

def DMRG_creation_B_Atilde(W,sel,pos0):
        
        #A_tilde va contenir les tenseurs de A qui ne vont être constant et B_vec les tenseurs qui vont être modifiés
        A_tilde=W.copy() ; B_vec=[] ; B_vec.append(A_tilde.pop(sel))
        if(sel<pos0) : B_vec.append(A_tilde.pop(pos0-1)) 
        else : B_vec.append(A_tilde.pop(pos0))
                       
        ###Construction de B ( contraction des 2 tenseurs )
        if(sel<pos0):
            if(sel==0):
                B=tl.tenalg.contract(B_vec[0],1,B_vec[1],1)
            else:
                B=tl.tenalg.contract(B_vec[0],2,B_vec[1],1)

        if(sel>pos0):
            if(sel==1):
                B=tl.tenalg.contract(B_vec[1],1,B_vec[0],1)
            else:
                B=tl.tenalg.contract(B_vec[1],2,B_vec[0],1)
        return (B,A_tilde)
    
def DMRG_creation_phi_tilde(A_tilde,Phi,sel,pos0,n,Min,N,nbTraining):
    Phi_tilde1=0 ; Phi_tilde2=0

    #Variable statique de la fonction
    if(n==0 and sel == 0 and pos0 == 1):
        DMRG_creation_phi_tilde.Phi_tilde1_table_to_right=[0]*nbTraining
        DMRG_creation_phi_tilde.Phi_tilde1_table_to_left=[[0 for i in range(N-2)] for j in range(nbTraining)]
        DMRG_creation_phi_tilde.Phi_tilde2_table_to_right=[[0 for i in range(N-2)] for j in range(nbTraining)]
        DMRG_creation_phi_tilde.Phi_tilde2_table_to_left=[0]*nbTraining
    
    ##Construction de Phi_tilde1 vers la droite
    if(sel==1 and sel<pos0):
        DMRG_creation_phi_tilde.Phi_tilde1_table_to_right[n]=tl.tenalg.mode_dot(A_tilde[0],Phi[:,0],0) #contractMPS(A_tilde[0:Min],Phi[:,:Min])
        Phi_tilde1=DMRG_creation_phi_tilde.Phi_tilde1_table_to_right[n]

    if(sel!=0 and sel!=1 and sel<pos0):
        DMRG_creation_phi_tilde.Phi_tilde1_table_to_right[n]=tl.tenalg.contract(A_tilde[Min-1],1,DMRG_creation_phi_tilde.Phi_tilde1_table_to_right[n],0)
        DMRG_creation_phi_tilde.Phi_tilde1_table_to_right[n]=tl.tenalg.mode_dot(DMRG_creation_phi_tilde.Phi_tilde1_table_to_right[n],Phi[:,Min-1],0)
        Phi_tilde1=DMRG_creation_phi_tilde.Phi_tilde1_table_to_right[n]

    ##Construction de Phi_tilde1 vers la gauche
    if(sel==(N-1)):
        DMRG_creation_phi_tilde.Phi_tilde1_table_to_left[n][0]=tl.tenalg.mode_dot(A_tilde[0],Phi[:,0],0)
        for i in range(1,N-2) :
            DMRG_creation_phi_tilde.Phi_tilde1_table_to_left[n][i]=tl.tenalg.contract(A_tilde[i],1,DMRG_creation_phi_tilde.Phi_tilde1_table_to_left[n][i-1],0)
            DMRG_creation_phi_tilde.Phi_tilde1_table_to_left[n][i]=tl.tenalg.mode_dot(DMRG_creation_phi_tilde.Phi_tilde1_table_to_left[n][i],Phi[:,i],0)
        Phi_tilde1=DMRG_creation_phi_tilde.Phi_tilde1_table_to_left[n][N-3]

    if(sel>pos0 and  not(sel==1 and sel>pos0)):
        Phi_tilde1=DMRG_creation_phi_tilde.Phi_tilde1_table_to_left[n][sel-2]

    ##Construction de Phi_tilde2 vers la droite
    if(sel==0):
        DMRG_creation_phi_tilde.Phi_tilde2_table_to_right[n][N-3]=tl.tenalg.mode_dot(A_tilde[N-3],Phi[:,N-3],0)
        for i in range(N-4,-1,-1) :
            DMRG_creation_phi_tilde.Phi_tilde2_table_to_right[n][i]=tl.tenalg.contract(A_tilde[i],2,DMRG_creation_phi_tilde.Phi_tilde2_table_to_right[n][i+1],0)
            DMRG_creation_phi_tilde.Phi_tilde2_table_to_right[n][i]=tl.tenalg.mode_dot(DMRG_creation_phi_tilde.Phi_tilde2_table_to_right[n][i],Phi[:,i],0)
        Phi_tilde2=DMRG_creation_phi_tilde.Phi_tilde2_table_to_right[n][0]

    if(sel!=0 and sel!=(N-2) and sel<pos0):
        Phi_tilde2=DMRG_creation_phi_tilde.Phi_tilde2_table_to_right[n][sel]

    ##Construction de Phi_tilde2 vers la gauche
    if(sel==N-2 and sel>pos0):
        DMRG_creation_phi_tilde.Phi_tilde2_table_to_left[n]=tl.tenalg.mode_dot(A_tilde[N-3],Phi[:,N-3],0)
        Phi_tilde2=DMRG_creation_phi_tilde.Phi_tilde2_table_to_left[n]

    if(sel != N-1 and sel!=N-2 and sel>pos0):
        DMRG_creation_phi_tilde.Phi_tilde2_table_to_left[n]=tl.tenalg.contract(A_tilde[sel-1],2,DMRG_creation_phi_tilde.Phi_tilde2_table_to_left[n],0)
        DMRG_creation_phi_tilde.Phi_tilde2_table_to_left[n]=tl.tenalg.mode_dot(DMRG_creation_phi_tilde.Phi_tilde2_table_to_left[n],Phi[:,sel-1],0)
        Phi_tilde2=DMRG_creation_phi_tilde.Phi_tilde2_table_to_left[n]
        
    return (Phi_tilde1,Phi_tilde2)

#Function to compute the gradient
def DMRG_calcul_cout_gradient(B,Phi_tilde1,Phi_tilde2,si,y,sel,pos0,posL,N):
    ##Computation of the cost
    if(sel==0 or (sel==1 and sel>pos0) ):
        Phi_tilde=np.multiply.outer(si[:,1],Phi_tilde2) #Phi_tilde2(alpha2,l) => Phi_tilde(s2,alpha2,l)
        Phi_tilde=np.multiply.outer(si[:,0],Phi_tilde) #=> Phi_tilde(s1,s2,alpha2,l)
        fl=tl.tenalg.contract(B,(0,1,2),Phi_tilde,(0,1,2))-y
        gradB=tl.tenalg.mode_dot(Phi_tilde,fl,3) #=> gradB(s1,s2,alpha2)
    elif( (sel==posL-1 and sel< pos0) or (sel==posL and sel> pos0) ):
        Phi_tilde=np.multiply.outer(si[:,0],Phi_tilde1) #=> Phi_tilde1(alpha(i-1)) => Phi_tilde1(si,alpha(i-1))
        Phi_tilde=np.multiply.outer(Phi_tilde,si[:,1]) #=> Phi_tilde(si,alpha(i-1),s(i+1))
        Phi_tilde=np.multiply.outer(Phi_tilde,Phi_tilde2) #=>Phi_tilde(si,alpha(i-1),s(i+1),alpha(i+1))
        fl=tl.tenalg.contract(B,(0,1,2,3),Phi_tilde,(0,1,2,3))-y
        gradB=np.multiply.outer(Phi_tilde,fl) #gradB(si,alpha(i-1),s(i+1),alpha(i+1),l)
    elif( (sel==posL and sel< pos0) or (sel==posL+1 and sel > pos0)  ):
        Phi_tilde=np.multiply.outer(si[:,0],Phi_tilde1) #=> Phi_tilde1(alpha(i-1)) => Phi_tilde1(si,alpha(i-1))
        Phi_tilde=np.multiply.outer(Phi_tilde,si[:,1]) #=> Phi_tilde(si,alpha(i-1),s(i+1))
        Phi_tilde=np.multiply.outer(Phi_tilde,Phi_tilde2) #=>Phi_tilde(si,alpha(i-1),s(i+1),alpha(i+1))
        fl=tl.tenalg.contract(B,(0,1,3,4),Phi_tilde,(0,1,2,3))-y
        Phi_tilde=np.multiply.outer(Phi_tilde,fl) #=>Phi_tilde(si,alpha(i-1),s(i+1),alpha(i+1),l)
        Phi_tilde=Phi_tilde.transpose((0,1,4,2,3)) #=>Phi_tilde(si,alpha(i-1),l,s(i+1),alpha(i+1))
        gradB=Phi_tilde
    elif( (sel==N-2 and sel<pos0 ) or sel==N-1):
        Phi_tilde=np.multiply.outer(si[:,0],Phi_tilde1) #Phi_tilde1(alpha(N-2),l) =>Phi_tilde1(s(N-1),alpha(N-2),l)
        Phi_tilde=np.multiply.outer(Phi_tilde,si[:,1]) #=> Phi_tilde(s(N-1),alpha(N-2),l,sN)
        fl=tl.tenalg.contract(B,(0,1,2),Phi_tilde,(0,1,3))-y
        gradB=tl.tenalg.mode_dot(Phi_tilde,fl,2) #=> gradB(s(N-1),alpha(N-2),sN)
    else:
        Phi_tilde=np.multiply.outer(si[:,0],Phi_tilde1) #=> Phi_tilde1(alpha(i-1)) => Phi_tilde1(si,alpha(i-1))
        Phi_tilde=np.multiply.outer(Phi_tilde,si[:,1]) #=> Phi_tilde(si,alpha(i-1),s(i+1))
        Phi_tilde=np.multiply.outer(Phi_tilde,Phi_tilde2) #=>Phi_tilde(si,alpha(i-1),s(i+1),alpha(i+1),l)
        if(sel>posL):
            fl=tl.tenalg.contract(B,(0,1,2,3),Phi_tilde,(0,1,3,4))-y
            gradB=tl.tenalg.mode_dot(Phi_tilde,fl,2) #gradB(si,alpha(i-1),s(i+1),alpha(i+1))
        else:
            fl=tl.tenalg.contract(B,(0,1,2,3),Phi_tilde,(0,1,2,3))-y
            gradB=tl.tenalg.mode_dot(Phi_tilde,fl,4) #gradB(si,alpha(i-1),s(i+1),alpha(i+1))
    cost=sum([i**2 for i in fl])

    return (cost,gradB)



if __name__ == "__main__":

    #test de la fonction creation_B_Atilde
    t1=np.array([[1,0],[0,4]])
    t2=np.array([[[1,0],[0,1]],[[2,0],[0,0]]])
    t3=np.array([[0,1],[2,3]])
    W=[t1,t2,t3]
    sel=1 ; pos0=2
    B , A_tilde = DMRG_creation_B_Atilde(W,sel,pos0)

    #test de la fonction DMRG_creation_phi_tilde
    n=0 ; Min=min(sel,pos0) ; N=9 ; nbTraining=30 ; posL=2
    p1=np.array([1,0]) ; p2=np.array([0,2]) ; p3=np.array([1,1])
    Phi=np.array([p1,p2,p3]).T

    (Phi_tilde1,Phi_tilde2) = DMRG_creation_phi_tilde(A_tilde,Phi,sel,pos0,n,Min,N,nbTraining)

    print(Phi_tilde1)
