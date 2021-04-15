import math
import numpy as np
import matplotlib.pyplot as plt

from src.tensor_network.algos_DMRG.gradient import DMRG_creation_B_Atilde, DMRG_creation_phi_tilde, DMRG_calcul_cout_gradient , DMRG_calcul_cout_gradient_test , gradient_descent_fixed_stepsize , ConjugateGradient , ConjugateGradient2 , compute_stuff_gradient , compute_cost , DMRG_creation_phi_tilde_test
from src.tensor_network.algos_DMRG.SVD import SVD_B
from src.mapping.phi import phi
from src.tensor_network.algo_GD.gradient import GD_creation_phi_tilde, GD_calcul_cout_gradient 
from src.tensor.tensor import contractMPS


class ModelMPS :
    
    def __init__(self,N,diml,W=[]):
        self.N=N #Number of tensor in the MPS form ( number of pixel )
        self.diml=diml #Number of classes
        self.W=W.copy() #Forme MPS de W
        self.nbSweep=0

    #Constant initialisation of the MPS Form
    def onesInitialisation(self,dimalpha,posL=-1,M=-1,goal=1,data=[]):
        self.W=[]
        self.posL = posL

        if(M==-1):
            if(len(data)==0):
                M=((np.sqrt(3)+0.80)/2)**(self.N)
            else:
                M=0
                for example in data:
                    M_ite = np.prod([np.cos(np.pi/2*elem) + np.sin(np.pi/2*elem) for elem in example])
                    if M_ite>M:
                        M=M_ite

        mfact=(goal)**(1/self.N)/(dimalpha**(1-(1/self.N))*M**(1/self.N))

        if(posL==-1):
            self.posL=math.floor(self.N/2) #le tenseur qui portera l
        
        if(self.algo == "GD"):
            self.posL=self.N-1

        self.W.append(np.ones((2,dimalpha))*mfact) #dim1 s(1), dim2 alpha(1) 
        for i in range(1,self.N-1):
            if(i==self.posL):
                self.W.append(np.ones((2,dimalpha,dimalpha,self.diml))*mfact) # dim1 : s(i) , dim2 alpha(i) ,dim3 alpha(i+1) ,dim4 l
            else:
                self.W.append(np.ones((2,dimalpha,dimalpha))*mfact) # dim1 : s(i) , dim2 alpha(i) ,dim3 alpha(i+1)
        if(self.algo=="DMRG"):
            self.W.append(np.ones((2,dimalpha))*mfact)  # dim1 : s(N) , dim2 alpha(N) 
        else:
            self.W.append(np.ones((2,dimalpha,self.diml))*mfact)  # dim1 : s(N) , dim2 alpha(N) , dim3 l

    #Normal initialisation of the MPS Form
    def normalInitialisation(self,dimalpha,mfact=0.95,posL=-1):
        self.posL = posL

        if(posL==-1):
            self.posL=math.floor(self.N/2) #le tenseur qui portera l
        
        if(self.algo == "GD"):
            self.posL=self.N-1

        self.W.append(np.random.randn(2,dimalpha)*mfact) #dim1 s(1), dim2 alpha(1) 
        for i in range(1,self.N-1):
            if(i==self.posL and self.algo=="DMRG"):
                self.W.append(np.random.randn(2,dimalpha,dimalpha,self.diml)*mfact) # dim1 : s(i) , dim2 alpha(i) ,dim3 alpha(i+1) ,dim4 l
            else:
                self.W.append(np.random.randn(2,dimalpha,dimalpha)*mfact) # dim1 : s(i) , dim2 alpha(i) ,dim3 alpha(i+1)
        if(self.algo=="DMRG"):
            self.W.append(np.random.randn(2,dimalpha)*mfact)  # dim1 : s(N) , dim2 alpha(N)
        else:
            self.W.append(np.random.randn(2,dimalpha,self.diml)*mfact)

    #Choice of the loss function
    def choose_loss_function(self,loss_function):
        self.loss_function = loss_function

    #Choice of the algo ( DMRG or GD )
    def choose_algo(self,algo_name):
        self.algo=algo_name
    
    #Choice of the optimizer
    def choose_optimizer(self,optimizer_name):
        self.optimizer = optimizer_name

    #Train the model with DMRG algo
    def trainDMRG_old(self,data_x,label,alpha=10**(-1),nmethod=1,maxalpha=10,cutoff=10**(-10)):
        err=[]
        nbTraining = data_x.shape[0]
        self.nbSweep+=2
        poss=[i for i in range(0,self.N)]+[i for i in range(0,self.N-1)][::-1]

        for _e in range(2*(self.N-1)):
            gradB=0
            cout=0
            sel=poss.pop(0)
            Max=max(sel,poss[0])
            Min=min(sel,poss[0])

            #Construction de B et A_tilde
            (B,A_tilde)=DMRG_creation_B_Atilde(self.W,sel,poss[0])
            
            if(alpha!=0):
                for n in range(nbTraining):
                    ### Création de phi_tilde ###
                    img=data_x[n].reshape(-1,)
                    si=phi(img[[Min,Max]]) ; Phi=phi(np.delete(img,(sel,poss[0])))

                    (Phi_tilde1,Phi_tilde2) = DMRG_creation_phi_tilde(A_tilde,Phi,sel,poss[0],n,Min,self.N,nbTraining)

                    ##Calcul Cout
                    (cout_ite,grad_ite)=DMRG_calcul_cout_gradient(B,Phi_tilde1,Phi_tilde2,si,label[n,:],sel,poss[0],self.posL,self.N)
                    cout+=cout_ite ; gradB+=grad_ite

                err.append( ((1/2)*cout)/nbTraining )
                B=B-alpha*gradB/nbTraining

            #SVD
            (self.W[Min],self.W[Max])=SVD_B(sel,poss[0],B,self.posL,self.N,maxalpha,cutoff,nmethod)
        return err

    #Train the model with DMRG algo
    def trainDMRG_old2(self,data_x,label,alpha=10**(-1),Npass=4,nmethod=1,maxalpha=10,cutoff=10**(-10)):
        err=[]
        nbTraining = data_x.shape[0]
        self.nbSweep+=2
        poss=[i for i in range(0,self.N)] +[i for i in range(0,self.N-1)][::-1]

        for _e in range(2*(self.N-1)):
            gradB=0
            cout=0
            sel=poss.pop(0)
            Max=max(sel,poss[0])
            Min=min(sel,poss[0])

            #Construction de B et A_tilde
            (B,A_tilde)=DMRG_creation_B_Atilde(self.W,sel,poss[0])
            
            Stockage=[]
            if(alpha!=0):
                for n in range(nbTraining):

                    ### Création de phi_tilde ###
                    img=data_x[n].reshape(-1,)
                    si=phi(img[[Min,Max]]) ; Phi=phi(np.delete(img,(sel,poss[0])))

                    (Phi_tilde1,Phi_tilde2) = DMRG_creation_phi_tilde(A_tilde,Phi,sel,poss[0],n,Min,self.N,nbTraining)

                    ##Calcul Cout
                    (cout_ite,grad_ite,Phi_tilde)=DMRG_calcul_cout_gradient_test(B,Phi_tilde1,Phi_tilde2,si,label[n,:],sel,poss[0],self.posL,self.N)
                    cout+=cout_ite ; gradB+=grad_ite 

                    Stockage.append((si,Phi_tilde1,Phi_tilde2,Phi_tilde))

                err.append( ((1/2)*cout)/nbTraining )
                #Algo minimisation gradient
                B = ConjugateGradient(Npass,B,sel,poss[0],self.posL,self.N,Stockage,label,gradB,nbTraining,cutoff)
                #B = gradient_descent_fixed_stepsize(B,alpha,gradB,nbTraining)
            #SVD
            (self.W[Min],self.W[Max])=SVD_B(sel,poss[0],B,self.posL,self.N,maxalpha,cutoff,nmethod)
        return err

    def trainDMRG(self,data_x,label,alpha=10**(-1),Npass=4,nmethod=1,maxalpha=10,cutoff=10**(-10)):
        err=[]
        nbTraining = data_x.shape[0]
        self.nbSweep+=2
        poss=[i for i in range(0,self.N)] +[i for i in range(0,self.N-1)][::-1]

        for _e in range(2*(self.N-1)):
            cout=0  #gradB = 0
            Phi_tilde_tab = []
            sel=poss.pop(0)
            Max=max(sel,poss[0])
            Min=min(sel,poss[0])

            #Construction de B et A_tilde
            (B,A_tilde)=DMRG_creation_B_Atilde(self.W,sel,poss[0])
            A=0 ; b=0
            for n in range(nbTraining):

                #Création de phi_tilde1 et phi_tilde2
                img=data_x[n].reshape(-1,)
                si=phi(img[[Min,Max]]) ; Phi=phi(np.delete(img,(sel,poss[0])))

                (Phi_tilde1,Phi_tilde2) = DMRG_creation_phi_tilde_test(A_tilde,Phi,sel,poss[0],n,Min,self.N,nbTraining)

                #Création de An, bn et Phi_tilde
                An , bn , Phi_tilde = compute_stuff_gradient(Phi_tilde1,Phi_tilde2,si,sel,poss[0],self.posL,self.N,label[n,:])
                
                if self.loss_function == "quadratic":
                    A+=An ; b+=bn 
                elif self.loss_function == "cross-entropy" or self.loss_function == "log-quadratic":
                    Phi_tilde_tab.append(Phi_tilde)

                #Calcul du cout
                cout += compute_cost(B,Phi_tilde,label[n,:],sel,poss[0],self.posL,self.N,self.loss_function)
                #(cost,gradite) = DMRG_calcul_cout_gradient(B,Phi_tilde1,Phi_tilde2,si,label[n,:],sel,poss[0],self.posL,self.N)
                #cout += cost
                #gradB += gradite

            #Ajout de l'erreur
            #err.append( ((1/2)*cout)/nbTraining )
            err.append( cout/nbTraining )

            #B=B-alpha*gradB/nbTraining
            if(self.optimizer=="fixed"):
                B = gradient_descent_fixed_stepsize(A,b,B,sel,poss[0],self.posL,self.N,alpha,nbTraining,cutoff,Npass,Phi_tilde_tab,self.loss_function,label)
            elif(self.optimizer=="CG"):
                B = ConjugateGradient2(A,b,Npass,B,sel,poss[0],self.posL,self.N,nbTraining,cutoff)
            elif(self.optimizer=="Adam"):
                pass
            
            #a decommenter si vous voulez calculer l'erreur suite a la descente de gradient
            """
            cout=0
            for n in range(nbTraining):
                ### Création de phi_tilde ###
                img=data_x[n].reshape(-1,)
                si=phi(img[[Min,Max]]) ; Phi=phi(np.delete(img,(sel,poss[0])))

                (Phi_tilde1,Phi_tilde2) = DMRG_creation_phi_tilde_test(A_tilde,Phi,sel,poss[0],n,Min,self.N,nbTraining,more=False)

                ##Calcul Cout
                An , bn , Phi_tilde = compute_stuff_gradient(Phi_tilde1,Phi_tilde2,si,sel,poss[0],self.posL,self.N,label[n,:])
                cout += compute_cost(B,Phi_tilde,label[n,:],sel,poss[0],self.posL,self.N)
            
            print(f"Erreur avant gradient conjuguée : {err[-1]} , après : {((1/2)*cout)/nbTraining}")
            if(err[-1]<((1/2)*cout)/nbTraining):
                print("#### WARNING ####")
                print(f"Erreur avant gradient conjuguée : {err[-1]} , après : {((1/2)*cout)/nbTraining}")
                print(f"sel = {sel} , pos0 = {poss[0]}")"""

            #SVD
            (self.W[Min],self.W[Max])=SVD_B(sel,poss[0],B,self.posL,self.N,maxalpha,cutoff,nmethod)
        return err

    def trainGD(self,data_x,label,alpha):
        err=[]
        nbTraining = data_x.shape[0]
        self.nbSweep+=1
        for e in range(self.N):
            gradW=0
            cout=0
            sel=e%self.N
            for n in range(nbTraining):
            ### Création de phi_tilde ###

                img=data_x[n].reshape(-1,)

                A_tilde=self.W.copy() ; A=A_tilde.pop(sel)
                si=phi(img[sel]) ; Phi=phi(np.delete(img,sel))

                (Phi_tilde1,Phi_tilde2) = GD_creation_phi_tilde(A_tilde,Phi,sel,n,self.N,nbTraining)

                (cout_ite,grad_ite)=GD_calcul_cout_gradient(A,Phi_tilde1,Phi_tilde2,si,label[n,:],sel,self.N)
                cout+=cout_ite ; gradW+=grad_ite

            self.W[sel]=self.W[sel]-alpha*gradW/nbTraining
            err.append( ((1/2)*cout)/nbTraining )
        return err

    def train(self,data_x,label,alpha=10**(-1),Npass=4,nmethod=1,maxalpha=10,cutoff=10**(-10)):
        if( self.algo == "GD"):
            return self.trainGD(data_x,label,alpha=alpha)
        elif self.algo == "DMRG":
            return self.trainDMRG(data_x,label,alpha=alpha,Npass=Npass,nmethod=nmethod,maxalpha=maxalpha,cutoff=cutoff)

    def predict(self,data_x):
        pred=[]
        nbExample = data_x.shape[0]
        for i in range(nbExample):
            img=data_x[i].reshape(-1,)
            Phi=phi(img)
            f = contractMPS(self.W,Phi)
            if(self.loss_function=="quadratic" or self.loss_function=="log-quadratic"):
                pred.append(f)
            elif self.loss_function=="cross-entropy":
                pred.append(np.exp(f)/sum(np.exp(f)))
        return np.array(pred)

    def accuracy(self,data_x,label):
        prediction=self.predict(data_x)
        nbExample=len(prediction)
        cpt=0
        for example in range(nbExample):
            if(np.argmax(prediction[example])==np.argmax(label[example])):
                cpt+=1
        return cpt/nbExample


if __name__ == "__main__":
    A = ModelMPS(9,2)
    A.choose_algo("DMRG")
    A.choose_optimizer("CG")
    A.normalInitialisation(5,0.4)
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

    err=[]
    for epoch in range(10):
        err += A.train(data,y,alpha=10**(-1))
        print(err[-1])
    print(A.predict(data))
    

    




