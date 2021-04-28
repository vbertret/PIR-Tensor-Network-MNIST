import math
import numpy as np
import matplotlib.pyplot as plt

from src.tensor_network.algos_DMRG.gradient import DMRG_creation_B_Atilde, DMRG_creation_phi_tilde, DMRG_calcul_cout_gradient , DMRG_calcul_cout_gradient_test , gradient_descent_fixed_stepsize , ConjugateGradient , ConjugateGradient2 , compute_stuff_gradient , compute_cost , DMRG_creation_phi_tilde_test , Adam 
from src.tensor_network.algos_DMRG.SVD import SVD_B
from src.mapping.phi import phi
from src.tensor_network.algo_GD.gradient import GD_creation_phi_tilde, GD_calcul_cout_gradient 
from src.tensor.tensor import contractMPS


class ModelMPS :
    """
    A class used to represent the Matrix Product State of a tensor

    This class is design in order to optimize a matrix product state in order to classify images or inputs in classes. 
    This process is simlar to what is described in the paper arxiv:1605.05775, but where the label index stays 
    fixed on the central tensor and does not move around during optimization.

    Attributes
    ----------
    N : int
        the number of tensor in the MPS form which is equal to the number of inputs
    diml : int
        the number of classes of the database
    W : list
        a list containing all the tensors of the MPS form
    nbSweep : int
        the number of sweeps made during optimization
    posL : int
        the position of the tensor with the label index
    loss_function : str
        the name of the loss function to use for the optimization
    algo : str
        the name of the algorithm to use for the optimization
    optimizer : str
        the name of the optimizer to use for the optimization

    Methods
    --------
    onesInitialization(dimalpha,posL=-1,M=-1,goal=1,data=[],sigma=0)
        Initializes all the terms of the tensors of the MPS form by the same constant
    normalInitialization(dimalpha,mfact=0.95,posL=-1)
        Initializes the terms of the MPS form using number randomly drawn from a normal distribution
    choose_loss_function(loss_function)
        Initializes the loss_function attribute
    choose_algo(algo_name)
        Initializes the algo attribute
    choose_optimizer(optimizer_name)
        Initializes the optimizer attribute
    trainDMRG(data_x,label,alpha=10**(-1),Npass=4,nmethod=1,maxalpha=10,cutoff=10**(-10))
        Trains the MPS form in order to be able to classify inputs with the Density Matrix Renormlization Group (DMRG) algorithm
    trainAGD(data_x,label,alpha)
        Trains the MPS form in order to be able to classify inputs with a basic Alternating Gradient Descent Algorithm
    train(data_x,label,alpha=10**(-1),Npass=4,nmethod=1,maxalpha=10,cutoff=10**(-10))
        Trains the MPS form in order to be able to classify inputs
    predict(data_x)
        Predicts labels based on inputs
    accuracy(self,data_x,label)
        Computes the accuracy score for some inputs and labels
    """
    
    def __init__(self,N,diml,W=[]):

        """
        Parameters
        ----------
        N : int 
            the number of tensor in the MPS form which is equal to the number of inputs
        diml : int 
            the number of classes of the database
        W : list, optional
            a list containing all the tensors of the MPS form (default is an empty list)
        """

        self.N=N 
        self.diml=diml 
        self.W=W.copy() 
        self.nbSweep=0

    def onesInitialization(self,dimalpha,posL=-1,M=-1,goal=1,data=[],sigma=0):

        """
        Initializes all the terms of the tensors of the MPS form by the same constant

        All the terms of the tensors of W are initialized with the same constant and with bond dimension alpha. 
        If the argument 'M' isn't passed in, either the argument 'data' is passed in and it is used in 
        order to compute 'M', or the method considers that the data is drawn from a uniform distribution on
        [0,1] i n order to compute 'M'.

        Parameters
        ----------
        dimalpha : int
            The bond dimension of all the tensors
        posL : int
            The index label (default is N/2 for DMRG and N-1 for GD)
        M : double
            A constant factor in order to have a good initial value
        goal : double
            The number corresponding to the maximum value of the decision function (default is 1)
        data : list
            The data used to compute M (default is an empty list)
        sigma : double
            The standard deviation that can be used to add some noises to the tensors (default is 0)
        """

        self.posL = posL

        #Computation of M
        if(M==-1):
            if(len(data)==0):
                M=((np.sqrt(3)+0.80)/2)**(self.N)
            else:
                M=0
                for example in data:
                    M_ite = np.prod([np.cos(np.pi/2*elem) + np.sin(np.pi/2*elem) for elem in example])
                    if M_ite>M:
                        M=M_ite

        #Computation of the constant value which is equal to all the terms of the tensors of the MPS form
        mfact=(goal)**(1/self.N)/(dimalpha**(1-(1/self.N))*M**(1/self.N))

        #Initialization of the label index
        if(posL==-1):
            self.posL=math.floor(self.N/2) #le tenseur qui portera l
        
        if(self.algo == "GD"):
            self.posL=self.N-1

        #Initialization of tensors
        self.W.append(np.ones((2,dimalpha))*mfact+sigma*np.random.randn(2,dimalpha)) #dim1 s(1), dim2 alpha(1) 
        for i in range(1,self.N-1):
            if(i==self.posL):
                self.W.append(np.ones((2,dimalpha,dimalpha,self.diml))*mfact+sigma*np.random.randn(2,dimalpha,dimalpha,self.diml)) # dim1 : s(i) , dim2 alpha(i) ,dim3 alpha(i+1) ,dim4 l
            else:
                self.W.append(np.ones((2,dimalpha,dimalpha))*mfact+sigma*np.random.randn(2,dimalpha,dimalpha)) # dim1 : s(i) , dim2 alpha(i) ,dim3 alpha(i+1)
        if(self.algo=="DMRG"):
            self.W.append(np.ones((2,dimalpha))*mfact+sigma*np.random.randn(2,dimalpha))  # dim1 : s(N) , dim2 alpha(N) 
        else:
            self.W.append(np.ones((2,dimalpha,self.diml))*mfact+sigma*np.random.randn(2,dimalpha,self.diml))  # dim1 : s(N) , dim2 alpha(N) , dim3 l

    def normalInitialization(self,dimalpha,mfact=0.95,posL=-1):

        """
        Initializes the terms of the MPS form using number randomly drawn from a normal distribution

        Parameters
        ----------
        dimalpha : int
            The bond dimension of all the tensors
        mfact : double
            A constant factor in order to have a good initial value
        posL : int
            The index label (default is N/2 for DMRG and N-1 for GD)
        """


        self.posL = posL

        #Initialization of the label index
        if(posL==-1):
            self.posL=math.floor(self.N/2) #le tenseur qui portera l
        
        if(self.algo == "GD"):
            self.posL=self.N-1

        #Initialization of tensors
        self.W.append(np.random.randn(2,dimalpha)*mfact) #dim1 s(1), dim2 alpha(1) 
        for i in range(1,self.N-1):
            if(i==self.posL and self.algo=="DMRG"):
                self.W.append(np.random.randn(2,dimalpha,dimalpha,self.diml)*mfact) # dim1 : s(i) , dim2 alpha(i) ,dim3 alpha(i+1) ,dim4 l
            else:
                self.W.append(np.random.randn(2,dimalpha,dimalpha)*mfact) # dim1 : s(i) , dim2 alpha(i) ,dim3 alpha(i+1)
        if(self.algo=="DMRG"):
            self.W.append(np.random.randn(2,dimalpha)*mfact)  # dim1 : s(N) , dim2 alpha(N)
        else:
            self.W.append(np.random.randn(2,dimalpha,self.diml)*mfact) # dim1 : s(N) , dim2 alpha(N) , dim3 l

    def choose_loss_function(self,loss_function):
        """
        Initializes the loss_function attribute

        Defines the loss_function to use to compute the loss.
        The possibilities are 'quadratic', 'log-quadratic' and 'cross-entropy'.

        Parameters
        ----------
        loss_function : str
            The name of the loss function to use for the optimization
        """

        self.loss_function = loss_function

    def choose_algo(self,algo_name):
        """
        Initializes the algo attribute

        Defines the algorithm to use for the optimization.
        The possibilities are 'DMRG' and 'AGD' for respectively Density Matrix Renormalization Group 
        and a simple algorithm of alternating gradient descent applied on each tensors.

        Parameters
        ----------
        algo_name : str
            The name of the algorithm to use for the optimization
        """
        self.algo=algo_name
    
    def choose_optimizer(self,optimizer_name):
        """
        Initializes the optimizer attribute

        Defines the optimizer to use in order to decrease the loss.
        The possibilities are 'CG' , 'GD' and 'Adam' for respectively 
        Conjugate Gradient, Gradient Descent and Adaptative Moment Estimation.
        The Conjugae Gradient can only be used with the 'quadratic' loss function.

        Parameters
        ----------
        optimizer_name : str
            The name of the optimizer to use for the optimization
        """
        self.optimizer = optimizer_name

    def trainDMRG(self,data_x,label,alpha=10**(-1),Npass=4,nmethod=1,maxalpha=10,cutoff=10**(-10)):
        """
        Trains the MPS form in order to be able to classify inputs with the Density Matrix Renormlization Group (DMRG) algorithm

        This is the central method of the class. It trains the MPS form using the DMRG algorithm doing 
        2 sweeps along the MPS form, one from the first tensor to the last and one from the last tensor to the first

        Parameters
        ----------
        data_x : numpy array
            The input data in a numpy array of shape (nbTraining,...)
        label : numpy array
            The labels corresponding to the input data in one encoding format. The shape needs to be (nbTraining,diml)
        alpha : double
            The learning rate for the optimizers Adam and GD (defaut is 10e-1)
        Npass : double
            The number of maximum pass to do at each step (default is 4)
        nmethod : int
            The number of the method to use for the Singular Value Decomposition (default is 1)
            More information in /algos_DMRG/SVD.py
        maxalpha : int
            The maximal bond dimension associated with each tensor (default is 10)
        cutoff : double
            The truncation error goal when optimizing the MPS (default is 10e-10)

        Returns
        -------
        list
            a list containing the error of each step
        """
        
        #Retrevial of the number of inputs and initialization of the error and all the steps of the algorithm
        err=[]
        nbTraining = data_x.shape[0]
        poss=[i for i in range(0,self.N)] +[i for i in range(0,self.N-1)][::-1]
        
        #Incrementation of the number of total sweeps
        self.nbSweep+=2
        
        #Loop over all the possibilities/steps
        for _e in range(2*(self.N-1)):
            #Initialization of the cost of the iteration and the tab which contains all the Phi_tilde
            cost=0  
            Phi_tilde_tab = []

            #Retrevial of impotant parameters of the step
            sel=poss.pop(0)
            Max=max(sel,poss[0])
            Min=min(sel,poss[0])

            #Construction of B and A_tilde and also initialization of A and b for the conjugate gradient
            (B,A_tilde)=DMRG_creation_B_Atilde(self.W,sel,poss[0])
            A=0 ; b=0

            #Loop over all the training data
            for n in range(nbTraining):

                #Flatten the data and compute Phi
                img=data_x[n].flatten()
                si=phi(img[[Min,Max]]) ; Phi=phi(np.delete(img,(sel,poss[0])))

                #Construction of phi_tilde1 and phi_tilde2
                (Phi_tilde1,Phi_tilde2) = DMRG_creation_phi_tilde_test(A_tilde,Phi,sel,poss[0],n,Min,self.N,nbTraining)

                #Construction of An, bn and Phi_tilde
                An , bn , Phi_tilde = compute_stuff_gradient(Phi_tilde1,Phi_tilde2,si,sel,poss[0],self.posL,self.N,label[n,:])
                
                #According to the loss function, either the method increments A and b, or it stocks Phi_tilde
                if self.loss_function == "quadratic":
                    A+=An ; b+=bn 
                elif self.loss_function == "cross-entropy" or self.loss_function == "log-quadratic":
                    Phi_tilde_tab.append(Phi_tilde)

                #Computation of the cost
                cost += compute_cost(B,Phi_tilde,label[n,:],sel,poss[0],self.posL,self.N,self.loss_function,cutoff)
                
            #Computation of the total cost of the step
            err.append( cost/nbTraining )

            #Optimization step
            if(self.optimizer=="fixed"):
                B = gradient_descent_fixed_stepsize(A,b,B,sel,poss[0],self.posL,self.N,alpha,nbTraining,cutoff,Npass,Phi_tilde_tab,self.loss_function,label)
            elif(self.optimizer=="CG"):
                B = ConjugateGradient2(A,b,Npass,B,sel,poss[0],self.posL,self.N,nbTraining,cutoff,Phi_tilde_tab,label,err[-1])
            elif(self.optimizer=="Adam"):
                B = Adam(A,b,B,sel,poss[0],self.posL,self.N,alpha,nbTraining,cutoff,Npass,Phi_tilde_tab,self.loss_function,label)
            
            #Computation of the SVD in order to find back the 2 tensors
            (self.W[Min],self.W[Max])=SVD_B(sel,poss[0],B,self.posL,self.N,maxalpha,cutoff,nmethod)

        return err

    def trainAGD(self,data_x,label,alpha):
        """
        Trains the MPS form in order to be able to classify inputs with a basic Alternating Gradient Descent Algorithm

        This method is not very important here. It was developped in order to compare with the DMRG algorithm. It can 
        only do basic gradient descent with a quadratic loss function.

        Parameters
        ----------
        data_x : numpy array
            The input data in a numpy array of shape (nbTraining,...)
        label : numpy array
            The labels corresponding to the input data in one encoding format. The shape needs to be (nbTraining,diml)
        alpha : double
            The learning rate for the optimizers Adam and GD (defaut is 10e-1)

        Returns
        -------
        list
            a list containing the error of each step
        """

        #Retrevial of the number of inputs and initialization of the error and all the steps of the algorithm
        err=[]
        nbTraining = data_x.shape[0]

        #Incrementation of the number of total sweeps
        self.nbSweep+=1

        #Loop over all the possibilities/steps
        for e in range(self.N):

            #Initialization of the cost and the gradient of the iteration 
            gradW=0
            cost=0

            #Retrevial of the tensor to be optimized
            sel=e%self.N

            #Construction of A and A_tilde
            A_tilde=self.W.copy() ; A=A_tilde.pop(sel)

            #Loop over all the training data
            for n in range(nbTraining):
                
                #Flatten the data and compute Phi
                img=data_x[n].flatten()
                si=phi(img[sel]) ; Phi=phi(np.delete(img,sel))

                #Construction of phi_tilde1 and phi_tilde2
                (Phi_tilde1,Phi_tilde2) = GD_creation_phi_tilde(A_tilde,Phi,sel,n,self.N,nbTraining)

                #Computation of the cost and gradient of the step
                (cost_ite,grad_ite)=GD_calcul_cout_gradient(A,Phi_tilde1,Phi_tilde2,si,label[n,:],sel,self.N)
                cost+=cost_ite ; gradW+=grad_ite

            #Optimization step : A gradient descent step with fixed stepsize
            self.W[sel]=self.W[sel]-alpha*gradW/nbTraining

            #Computation of the cost
            err.append( ((1/2)*cost)/nbTraining )
        return err

    def train(self,data_x,label,alpha=10**(-1),Npass=4,nmethod=1,maxalpha=10,cutoff=10**(-10)):
        """
        Trains the MPS form in order to be able to classify inputs

        According to the algo attribute, the method calls trainAGD or train DMRG

        Parameters
        ----------
        data_x : numpy array
            The input data in a numpy array of shape (nbTraining,...)
        label : numpy array
            The labels corresponding to the input data in one encoding format. The shape needs to be (nbTraining,diml)
        alpha : double
            The learning rate for the optimizers Adam and GD (defaut is 10e-1)
        Npass : double
            The number of maximum pass to do at each step (default is 4)
        nmethod : int
            The number of the method to use for the Singular Value Decomposition (default is 1)
            More information in /algos_DMRG/SVD.py
        maxalpha : int
            The maximal bond dimension associated with each tensor (default is 10)
        cutoff : double
            The truncation error goal when optimizing the MPS (default is 10e-10)
        
        Returns
        -------
        list from the method trainAGD or trainDMRG
            a list containing the error of each step
        """
        
        if( self.algo == "GD"):
            return self.trainAGD(data_x,label,alpha=alpha)
        elif self.algo == "DMRG":
            return self.trainDMRG(data_x,label,alpha=alpha,Npass=Npass,nmethod=nmethod,maxalpha=maxalpha,cutoff=cutoff)

    def predict(self,data_x):
        """
        Predicts labels based on inputs

        Compute the decision function for all the inputs.

        Parameters
        ----------
        data_x : numpy array
            The input data in a numpy array of shape (nbExample,...)

        Returns
        -------
        list of numpy array
            The value of the decision function for each input
        """

        #Initialization of the list which contains the value of the decision function
        pred=[]

        #Retrevial of the number of inputs
        nbExample = data_x.shape[0]

        #Loop over all the data
        for i in range(nbExample):

            #Flatten the data and compute Phi
            img=data_x[i].reshape(-1,)
            Phi=phi(img)

            #Computation of the value of the decision function
            f = contractMPS(self.W,Phi)
            if(self.loss_function=="quadratic" or self.loss_function=="log-quadratic"):
                pred.append(f)
            elif self.loss_function=="cross-entropy":
                pred.append(np.exp(f)/sum(np.exp(f)))

        return np.array(pred)

    def accuracy(self,data_x,label):
        """
        Computes the accuracy score for some inputs and labels

        Makes predictions for all the input data and after compares them with the labels
        to compute the accuracy score.

        Parameters
        ----------
        data_x : numpy array
            The input data in a numpy array of shape (nbExample,...)
        label : numpy array
            The labels corresponding to the input data in one encoding format. The shape needs to be (nbExamlple,diml)

        Returns
        ------
        int:
            The accuracy score
        """

        #Computation of the prediction
        prediction=self.predict(data_x)

        #Retrevial of the number of inputs
        nbExample=len(prediction)

        #Initialization of the counter of good classification
        cpt=0

        #Loop over all the examples
        for example in range(nbExample):

            #If the example is well classified, incrementation of the counter
            if(np.argmax(prediction[example])==np.argmax(label[example])):
                cpt+=1
                
        return cpt/nbExample


if __name__ == "__main__":
    A = ModelMPS(9,2)
    A.choose_algo("DMRG")
    A.choose_optimizer("CG")
    A.normalInitialization(5,0.4)
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
    

    




