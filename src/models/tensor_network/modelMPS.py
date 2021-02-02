import math
import numpy as np
import matplotlib.pyplot as plt
try:
    from algos_DMRG.gradient import DMRG_creation_B_Atilde, DMRG_creation_phi_tilde, DMRG_calcul_cout_gradient
    from algos_DMRG.SVD import SVD_B
    from mapping.phi import phi
    from algo_GD.gradient import GD_creation_phi_tilde, GD_calcul_cout_gradient
    from tensor.tensor import contractMPS
except ImportError:
    from .algos_DMRG.gradient import DMRG_creation_B_Atilde, DMRG_creation_phi_tilde, DMRG_calcul_cout_gradient
    from .algos_DMRG.SVD import SVD_B
    from .mapping.phi import phi
    from .algo_GD.gradient import GD_creation_phi_tilde, GD_calcul_cout_gradient
    from .tensor.tensor import contractMPS

class ModelMPS :
    
    def __init__(self,N,diml,W=[]):
        self.N=N #Number of tensor in the MPS form ( number of pixel )
        self.diml=diml #Number of classes
        self.W=W.copy() #Forme MPS de W
        self.nbSweep=0

    #Constant initialisation of the MPS Form
    def onesInitialisation(self,dimalpha,mfact=0.95,posL=-1):
        self.posL = posL

        if(posL==-1):
            self.posL=math.floor(self.N/2) #le tenseur qui portera l
        
        if(self.algo_name == "GD"):
            self.posL=self.N-1

        self.W.append(np.ones((2,dimalpha))*mfact) #dim1 s(1), dim2 alpha(1) 
        for i in range(1,self.N-1):
            if(i==self.posL):
                self.W.append(np.ones((2,dimalpha,dimalpha,self.diml))*mfact) # dim1 : s(i) , dim2 alpha(i) ,dim3 alpha(i+1) ,dim4 l
            else:
                self.W.append(np.ones((2,dimalpha,dimalpha))*mfact) # dim1 : s(i) , dim2 alpha(i) ,dim3 alpha(i+1)
        self.W.append(np.ones((2,dimalpha))*mfact)  # dim1 : s(N) , dim2 alpha(N) 

    #Normal initialisation of the MPS Form
    def normalInitialisation(self,dimalpha,mfact=0.95,posL=-1):
        self.posL = posL

        if(posL==-1):
            self.posL=math.floor(self.N/2) #le tenseur qui portera l
        
        if(self.algo_name == "GD"):
            self.posL=self.N-1

        self.W.append(np.random.randn(2,dimalpha)*mfact) #dim1 s(1), dim2 alpha(1) 
        for i in range(1,self.N-1):
            if(i==self.posL and self.algo_name=="DMRG"):
                self.W.append(np.random.randn(2,dimalpha,dimalpha,self.diml)*mfact) # dim1 : s(i) , dim2 alpha(i) ,dim3 alpha(i+1) ,dim4 l
            else:
                self.W.append(np.random.randn(2,dimalpha,dimalpha)*mfact) # dim1 : s(i) , dim2 alpha(i) ,dim3 alpha(i+1)
        if(self.algo_name=="DMRG"):
            self.W.append(np.random.randn(2,dimalpha)*mfact)  # dim1 : s(N) , dim2 alpha(N)
        else:
            self.W.append(np.random.randn(2,dimalpha,self.diml)*mfact)

    #Choice of the algo to minimize the cost function
    def algo(self,algo_name):
        self.algo_name=algo_name

    #Train the model with DMRG algo
    def trainDMRG(self,data_x,label,alpha=10**(-1),nmethod=1,maxalpha=10,cutoff=10**(-10)):
        err=[]
        nbTraining = data_x.shape[0]
        self.nbSweep+=2
        poss=[i for i in range(0,self.N)]+[i for i in range(0,self.N-1)][::-1]

        for e in range(2*(self.N-1)):
            gradB=0
            cout=0
            sel=poss.pop(0)
            Max=max(sel,poss[0])
            Min=min(sel,poss[0])

            #Construction de B et A_tilde
            (B,A_tilde)=DMRG_creation_B_Atilde(self.W,sel,poss[0])

            for n in range(nbTraining):
                ### Création de phi_tilde ###
                img=data_x[n].reshape(-1,)
                si=phi(img[[Min,Max]]) ; Phi=phi(np.delete(img,(sel,poss[0])))

                (Phi_tilde1,Phi_tilde2) = DMRG_creation_phi_tilde(A_tilde,Phi,sel,poss[0],n,Min,self.N,nbTraining)

                ##Calcul Cout
                (cout_ite,grad_ite)=DMRG_calcul_cout_gradient(B,Phi_tilde1,Phi_tilde2,si,label[n,:],sel,poss[0],self.posL,self.N)
                cout+=cout_ite ; gradB+=grad_ite

            err.append( ((1/2)*cout)/nbTraining )
            B=B-alpha*gradB

            #SVD
            (self.W[Min],self.W[Max])=SVD_B(sel,poss[0],B,self.posL,self.N,maxalpha,cutoff,nmethod)
        return err

    def trainGD(self,data_x,label,alpha=10**(-1)):
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

            self.W[sel]=self.W[sel]-alpha*gradW
            err.append( ((1/2)*cout)/nbTraining )
        return err

    def train(self,data_x,label,alpha=10**(-1),nmethod=1,maxalpha=10,cutoff=10**(-10)):
        if( self.algo_name == "GD"):
            return self.trainGD(data_x,label,alpha)
        else:
            return self.trainDMRG(data_x,label,alpha,nmethod,maxalpha,cutoff)

    def predict(self,data_x):
        pred=[]
        nbExample = data_x.shape[0]
        for i in range(nbExample):
            img=data_x[i].reshape(-1,)
            Phi=phi(img)
            pred.append(contractMPS(self.W,Phi))
        return pred

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
    A.algo("DMRG")
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
    

    




