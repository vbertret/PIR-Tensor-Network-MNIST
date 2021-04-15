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

#Function to compute the gradient
def DMRG_calcul_cout_gradient_test(B,Phi_tilde1,Phi_tilde2,si,y,sel,pos0,posL,N):
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
        gradB=np.multiply.outer(Phi_tilde,fl) #=>gradB(si,alpha(i-1),s(i+1),alpha(i+1),l)
        gradB=gradB.transpose((0,1,4,2,3)) #=>gradB(si,alpha(i-1),l,s(i+1),alpha(i+1))
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

    return (cost,gradB,Phi_tilde)

def ConjugateGradient(Npass,B,sel,pos0,posL,N,Stockage,label,gradB,nbTraining):
    p=[]
    pAp=[]
    pA=[]
    r_tab=[]
    A=0
    b=np.zeros(B.shape)
    for i in range(Npass):
        
        ###Creation de A et b
        if(i==0):
            for n in range(nbTraining):

                (si,Phi_tilde1,Phi_tilde2,Phi_tilde)=Stockage[n]
                temp1=si[:,0].reshape(2,1)
                An=np.dot(temp1,temp1.T)
                temp1=si[:,1].reshape(2,1)
                An=np.multiply.outer(An,np.dot(temp1,temp1.T))
                
                if(sel==0 or (sel==1 and sel>pos0) ):
                    temp2=np.dot(Phi_tilde2,Phi_tilde2.T)
                    An=np.multiply.outer(An,temp2) #An -> (s1,s1,s2,s2,alpha2,alpha2)
                    b+=tl.tenalg.mode_dot(Phi_tilde,label[n,:],3)
                elif( (sel==posL-1 and sel< pos0) or (sel==posL and sel> pos0) ):
                    Phi_tilde1=Phi_tilde1.reshape(Phi_tilde1.shape[0],1) ; Phi_tilde2=Phi_tilde2.reshape(Phi_tilde2.shape[0],1)
                    temp2=np.dot(Phi_tilde1,Phi_tilde1.T)
                    An=np.multiply.outer(An,temp2)
                    temp2=np.dot(Phi_tilde2,Phi_tilde2.T)
                    An=np.multiply.outer(An,temp2) #An -> (sj,sj,sj+1,sj+1,alphaj-1,alphaj-1,alphaj+1,alphaj+1)
                    An=An.transpose((0,1,4,5,2,3,6,7)) #An -> (sj,sj,alphaj-1,alphaj-1,sj+1,sj+1,alphaj+1,alphaj+1)
                    b+=np.multiply.outer(Phi_tilde,label[n,:])
                elif( (sel==posL and sel< pos0) or (sel==posL+1 and sel > pos0)  ):
                    Phi_tilde1=Phi_tilde1.reshape(Phi_tilde1.shape[0],1) ; Phi_tilde2=Phi_tilde2.reshape(Phi_tilde2.shape[0],1)
                    temp2=np.dot(Phi_tilde1,Phi_tilde1.T)
                    An=np.multiply.outer(An,temp2)
                    temp2=np.dot(Phi_tilde2,Phi_tilde2.T)
                    An=np.multiply.outer(An,temp2) #An -> (sj,sj,sj+1,sj+1,alphaj-1,alphaj-1,alphaj+1,alphaj+1)
                    An=An.transpose((0,1,4,5,2,3,6,7)) #An -> (sj,sj,alphaj-1,alphaj-1,sj+1,sj+1,alphaj+1,alphaj+1)
                    bn=np.multiply.outer(Phi_tilde,label[n,:])
                    b+=bn.transpose((0,1,4,2,3))
                elif( (sel==N-2 and sel<pos0 ) or sel==N-1):
                    temp2=np.dot(Phi_tilde1,Phi_tilde1.T)
                    An=np.multiply.outer(An,temp2) #An -> (sN-1,sN-1,sN,sN,alphaN-2,alphaN-2)
                    An=An.transpose((0,1,4,5,2,3)) #An -> (sN-1,sN-1,alphaN-2,alphaN-2,sN,sN)
                    b+=tl.tenalg.mode_dot(Phi_tilde,label[n,:],2)
                else:
                    if(sel<posL):
                        Phi_tilde1=Phi_tilde1.reshape(Phi_tilde1.shape[0],1)
                    else:
                        Phi_tilde2=Phi_tilde2.reshape(Phi_tilde2.shape[0],1)
                    temp2=np.dot(Phi_tilde1,Phi_tilde1.T)
                    An=np.multiply.outer(An,temp2)
                    temp2=np.dot(Phi_tilde2,Phi_tilde2.T)
                    An=np.multiply.outer(An,temp2) #An -> (s2,s2,s3,s3,alpha1,alpha1,alpha3,alpha3)
                    An=An.transpose((0,1,4,5,2,3,6,7)) #An -> (s2,s2,alpha1,alpha1,s3,s3,alpha3,alpha3)
                    if(sel<posL):
                        b+=tl.tenalg.mode_dot(Phi_tilde,label[n,:],4)
                    else:
                        b+=tl.tenalg.mode_dot(Phi_tilde,label[n,:],2)
                if(n==0):
                    A=np.zeros(An.shape)
                A+=An
        else:
            if(sel==0 or (sel==1 and sel>pos0) ):
                gradB=tl.tenalg.contract(A,(0,2,4),B,(0,1,2))-b
            elif( (sel==posL-1 and sel< pos0) or (sel==posL and sel> pos0) ):
                gradB=tl.tenalg.contract(A,(0,2,4,6),B,(0,1,2,3))-b 
            elif( (sel==posL and sel< pos0) or (sel==posL+1 and sel > pos0)  ):
                gradB=tl.tenalg.contract(A,(0,2,4,6),B,(0,1,3,4)).transpose((0,1,4,2,3))-b 
            elif( (sel==N-2 and sel<pos0 ) or sel==N-1):
                gradB=tl.tenalg.contract(A,(0,2,4),B,(0,1,2))-b
            else:
                gradB=tl.tenalg.contract(A,(0,2,4,6),B,(0,1,2,3))-b 
        #Gradient conjugué
        if(sel==0 or (sel==1 and sel>pos0) ):
            if i==0:
                r=-gradB
                ptemp=r
                p.append(ptemp)
        elif( (sel==posL-1 and sel< pos0) or (sel==posL and sel> pos0) ):
            if i==0:
                r=-gradB
                ptemp=r
                p.append(ptemp)
        elif( (sel==posL and sel< pos0) or (sel==posL+1 and sel > pos0)  ):
            if i==0:
                r=-gradB
                ptemp=r
                p.append(ptemp)
        elif( (sel==N-2 and sel<pos0 ) or sel==N-1):
            if i==0:
                r=-gradB
                ptemp=r
                p.append(ptemp)
        else:
            if i==0:
                r=-gradB
                ptemp=r
                p.append(ptemp)
        

        if(sel==0 or (sel==1 and sel>pos0) ):
            pA.append(tl.tenalg.contract(p[i],(0,1,2),A,(0,2,4)))
            pAp.append(float(tl.tenalg.contract(p[i],(0,1,2),pA[i],(0,1,2)))) 
            r_tab.append(tl.tenalg.contract(r,(0,1,2),r,(0,1,2)))
            alpha=r_tab[i]/pAp[i]
            new_r=r-alpha*pA[i]
            r_tab.append(tl.tenalg.contract(new_r,(0,1,2),new_r,(0,1,2)))
            beta=r_tab[i+1]/r_tab[i]
            p.append(new_r+beta*p[i])
            r=new_r
            """for len_p in range(len(p)):
                ptemp-= float(tl.tenalg.contract(pA[len_p],(0,1,2),r,(0,1,2))/pAp[len_p])*p[len_p]
            p.append(ptemp)

            pA.append(tl.tenalg.contract(p[i],(0,1,2),A,(0,2,4)))
            pAp.append(float(tl.tenalg.contract(p[i],(0,1,2),pA[i],(0,1,2)))) 
            alpha=tl.tenalg.contract(pA[i],(0,1,2),r,(0,1,2))/pAp[i]"""
        elif( (sel==posL-1 and sel< pos0) or (sel==posL and sel> pos0) ):
            pA.append(tl.tenalg.contract(A,(0,2,4,6),p[i],(0,1,2,3)))
            pAp.append(float(tl.tenalg.contract(p[i],(0,1,2,3,4),pA[i],(0,1,2,3,4)))) 
            r_tab.append(tl.tenalg.contract(r,(0,1,2,3,4),r,(0,1,2,3,4)))
            alpha=r_tab[i]/pAp[i]
            new_r=r-alpha*pA[i]
            r_tab.append(tl.tenalg.contract(new_r,(0,1,2,3,4),new_r,(0,1,2,3,4)))
            beta=r_tab[i+1]/r_tab[i]
            p.append(new_r+beta*p[i])
            r=new_r
            """for len_p in range(len(p)):
                ptemp-= float(tl.tenalg.contract(pA[len_p],(0,1,2,3,4),r,(0,1,2,3,4))/pAp[len_p])*p[len_p]
            p.append(ptemp)

            pA.append(tl.tenalg.contract(A,(0,2,4,6),p[i],(0,1,2,3)))
            pAp.append(float(tl.tenalg.contract(p[i],(0,1,2,3,4),pA[i],(0,1,2,3,4)))) 
            alpha=tl.tenalg.contract(pA[i],(0,1,2,3,4),r,(0,1,2,3,4))/pAp[i]"""
        elif( (sel==posL and sel< pos0) or (sel==posL+1 and sel > pos0)  ):
            pA.append(tl.tenalg.contract(A,(0,2,4,6),p[i],(0,1,3,4)).transpose((0,1,4,2,3)))
            pAp.append(float(tl.tenalg.contract(p[i],(0,1,2,3,4),pA[i],(0,1,2,3,4)))) 
            r_tab.append(tl.tenalg.contract(r,(0,1,2,3,4),r,(0,1,2,3,4)))
            alpha=r_tab[i]/pAp[i]
            new_r=r-alpha*pA[i]
            r_tab.append(tl.tenalg.contract(new_r,(0,1,2,3,4),new_r,(0,1,2,3,4)))
            beta=r_tab[i+1]/r_tab[i]
            p.append(new_r+beta*p[i])
            r=new_r
            """for len_p in range(len(p)):
                ptemp-= float(tl.tenalg.contract(pA[len_p],(0,1,4,2,3),r,(0,1,2,3,4))/pAp[len_p])*p[len_p]
            p.append(ptemp)

            pA.append(tl.tenalg.contract(A,(0,2,4,6),p[i],(0,1,3,4)))
            pAp.append(float(tl.tenalg.contract(p[i],(0,1,2,3,4),pA[i],(0,1,4,2,3)))) 
            alpha=tl.tenalg.contract(pA[i],(0,1,4,2,3),r,(0,1,2,3,4))/pAp[i]"""
        elif( (sel==N-2 and sel<pos0 ) or sel==N-1):
            pA.append(tl.tenalg.contract(p[i],(0,1,2),A,(0,2,4)))
            pAp.append(float(tl.tenalg.contract(p[i],(0,1,2),pA[i],(0,1,2)))) 
            r_tab.append(tl.tenalg.contract(r,(0,1,2),r,(0,1,2)))
            alpha=r_tab[i]/pAp[i]
            new_r=r-alpha*pA[i]
            r_tab.append(tl.tenalg.contract(new_r,(0,1,2),new_r,(0,1,2)))
            beta=r_tab[i+1]/r_tab[i]
            p.append(new_r+beta*p[i])
            r=new_r
            
            """for len_p in range(len(p)):
                ptemp-= float(tl.tenalg.contract(pA[len_p],(0,1,2),r,(0,1,2))/pAp[len_p])*p[len_p]
            p.append(ptemp)

            pA.append(tl.tenalg.contract(p[i],(0,1,2),A,(0,2,4)))
            pAp.append(float(tl.tenalg.contract(p[i],(0,1,2),pA[i],(0,1,2)))) 
            alpha=tl.tenalg.contract(pA[i],(0,1,2),r,(0,1,2))/pAp[i]"""
        else:
            pA.append(tl.tenalg.contract(p[i],(0,1,2,3),A,(0,2,4,6)))
            pAp.append(float(tl.tenalg.contract(p[i],(0,1,2,3),pA[i],(0,1,2,3)))) 
            r_tab.append(tl.tenalg.contract(r,(0,1,2,3),r,(0,1,2,3)))
            alpha=r_tab[i]/pAp[i]
            new_r=r-alpha*pA[i]
            r_tab.append(tl.tenalg.contract(new_r,(0,1,2,3),new_r,(0,1,2,3)))
            beta=r_tab[i+1]/r_tab[i]
            p.append(new_r+beta*p[i])
            r=new_r
            """for len_p in range(len(p)):
                ptemp-= float(tl.tenalg.contract(pA[len_p],(0,1,2,3),r,(0,1,2,3))/pAp[len_p])*p[len_p]
            p.append(ptemp)

            pA.append(tl.tenalg.contract(p[i],(0,1,2,3),A,(0,2,4,6)))
            pAp.append(float(tl.tenalg.contract(p[i],(0,1,2,3),pA[i],(0,1,2,3)))) 
            alpha=tl.tenalg.contract(pA[i],(0,1,2,3),r,(0,1,2,3))/pAp[i]"""



        B=B+alpha*p[i]
    return B
        

#sortir d'ici mettre dans un nouveau dossier avec tous les algos d'optimisation de gradient
def gradient_descent_fixed_stepsize(B,alpha,gradB,nbTraining):
        return B-alpha*gradB/nbTraining



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