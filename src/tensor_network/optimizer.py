import tensorly as tl
import numpy as np
from src.tensor_network.algos_DMRG.gradient import DMRG_compute_gradient

def ConjugateGradient_old(A,b,Npass,B,sel,pos0,posL,N,nbTraining,cutoff,Phi_tilde_tab,label,cost,eps=10**(-10)):
    """
    Conjugate gradient descent

    Optimizes the tensor B with conjugate gradient descent

    Parameters
    ----------
    A : numpy array
        the tensor A constructed for the conjugate gradient
    b : numpy array
        the tensor b constructed for the conjugate gradient
    Npass : int
        The number of maximum pass to do at each step 
    B : numpy array
        the tensor B to optimize
    sel : int
        the number of one of the tensor selected to be in B
    pos0 : int
        the number of the other tensor selected to be in B
    posL : int
        the position of the label index
    N : int
        the number of tensor in the MPS form which is equal to the number of input
    nbTraining : int
        the number of pictures used to train
    cutoff : double
        The truncation error goal when optimizing the MPS
    eps : double (optional)
        to avoid problems of division per zero (default is 10e-10)

    Return
    ------
    B : numpy array
        the new value of B after Conjugate Gradient descent
    """

    #Computation of the gradient
    gradB = DMRG_compute_gradient(A,b,B,sel,pos0,posL,N)

    #Initialization of r and p
    r=-gradB
    p=r

    #Computation of the norm of r
    if(sel==0 or (sel==1 and sel>pos0) ):
        r_prod = tl.tenalg.contract(r,(0,1,2),r,(0,1,2))
    elif( (sel==posL-1 and sel< pos0) or (sel==posL and sel> pos0) ):
        r_prod = tl.tenalg.contract(r,(0,1,2,3,4),r,(0,1,2,3,4))
    elif( (sel==posL and sel< pos0) or (sel==posL+1 and sel > pos0)  ):
        r_prod = tl.tenalg.contract(r,(0,1,2,3,4),r,(0,1,2,3,4))
    elif( (sel==N-2 and sel<pos0 ) or sel==N-1):
        r_prod = tl.tenalg.contract(r,(0,1,2),r,(0,1,2))
    else:
        r_prod = tl.tenalg.contract(r,(0,1,2,3),r,(0,1,2,3))

    #Loop over Npass
    for _i in range(Npass): 
        if _i==0 and tl.norm(r,2) < cutoff :
            break
        
        if(sel==0 or (sel==1 and sel>pos0) ):
            Ap = tl.tenalg.contract(A,(0,2,4),p,(0,1,2))
            alpha = r_prod/(tl.tenalg.contract(p,(0,1,2),Ap,(0,1,2))+eps)
            B = B + alpha*p
            r=r-alpha*Ap
            rnew_prod = tl.tenalg.contract(r,(0,1,2),r,(0,1,2))
            if tl.norm(r,2) < cutoff :
                break
            p= r + (rnew_prod/r_prod)*p
            r_prod = rnew_prod
        elif( (sel==posL-1 and sel< pos0) or (sel==posL and sel> pos0) ):
            Ap = tl.tenalg.contract(A,(0,2,4,6),p,(0,1,2,3))
            alpha = r_prod/(tl.tenalg.contract(p,(0,1,2,3,4),Ap,(0,1,2,3,4))+eps)
            B = B + alpha*p
            r=r-alpha*Ap
            rnew_prod = tl.tenalg.contract(r,(0,1,2,3,4),r,(0,1,2,3,4))
            if tl.norm(r,2) < cutoff :
                break
            p= r + (rnew_prod/r_prod)*p
            r_prod = rnew_prod
        elif( (sel==posL and sel< pos0) or (sel==posL+1 and sel > pos0)  ):
            Ap = tl.tenalg.contract(A,(0,2,4,6),p,(0,1,3,4)).transpose((0,1,4,2,3))
            alpha = r_prod/(tl.tenalg.contract(p,(0,1,2,3,4),Ap,(0,1,2,3,4))+eps)
            B = B + alpha*p
            r=r-alpha*Ap
            rnew_prod = tl.tenalg.contract(r,(0,1,2,3,4),r,(0,1,2,3,4))
            if tl.norm(r,2) < cutoff :
                break
            p= r + (rnew_prod/r_prod)*p
            r_prod = rnew_prod
        elif( (sel==N-2 and sel<pos0 ) or sel==N-1):
            Ap = tl.tenalg.contract(A,(0,2,4),p,(0,1,2))
            alpha = r_prod/(tl.tenalg.contract(p,(0,1,2),Ap,(0,1,2))+eps)
            B = B + alpha*p
            r=r-alpha*Ap
            rnew_prod = tl.tenalg.contract(r,(0,1,2),r,(0,1,2))
            if tl.norm(r,2) < cutoff :
                break
            p= r + (rnew_prod/r_prod)*p
            r_prod = rnew_prod
        else:
            Ap = tl.tenalg.contract(A,(0,2,4,6),p,(0,1,2,3))
            alpha = r_prod/(tl.tenalg.contract(p,(0,1,2,3),Ap,(0,1,2,3))+eps)
            B = B + alpha*p
            r=r-alpha*Ap
            rnew_prod = tl.tenalg.contract(r,(0,1,2,3),r,(0,1,2,3))
            if tl.norm(r,2) < cutoff :
                break
            p= r + (rnew_prod/r_prod)*p
            r_prod = rnew_prod

    #    new_cost=0
    #    for nb in range(len(Phi_tilde_tab)):
    #        Phi_tilde = Phi_tilde_tab[nb]
    #        new_cost+= compute_cost(B,Phi_tilde,label[nb,:],sel,pos0,posL,N,"quadratic",cutoff)
    #    
    #    print(f"cout : {new_cost/len(Phi_tilde_tab)} norme r : {tl.norm(r,1)}", end =" ,")
    #    if(new_cost/len(Phi_tilde_tab) > cost):
    #        erreur=True
    #    cost = new_cost/len(Phi_tilde_tab)

    #if erreur:
    #    print(f" sel : {sel} pos0 : {pos0} ERREUR !!!")
    #else:
    #    print(f" sel : {sel} pos0 : {pos0}")

    return B

def ConjugateGradient(A,b,Npass,B,sel,pos0,posL,N,nbTraining,cutoff,Phi_tilde_tab,label,cost):
    """
    Conjugate gradient descent

    Optimizes the tensor B with conjugate gradient descent

    Parameters
    ----------
    A : numpy array
        the tensor A constructed for the conjugate gradient
    b : numpy array
        the tensor b constructed for the conjugate gradient
    Npass : int
        The number of maximum pass to do at each step 
    B : numpy array
        the tensor B to optimize
    sel : int
        the number of one of the tensor selected to be in B
    pos0 : int
        the number of the other tensor selected to be in B
    posL : int
        the position of the label index
    N : int
        the number of tensor in the MPS form which is equal to the number of input
    nbTraining : int
        the number of pictures used to train
    cutoff : double
        The truncation error goal when optimizing the MPS
    eps : double (optional)
        to avoid problems of division per zero (default is 10e-10)

    Return
    ------
    B : numpy array
        the new value of B after Conjugate Gradient descent
    """
    
    #Initialization of the lists which will stocks differents things
    p=[] ; pAp=[] ; pA=[] ; r_tab=[]
    i=0 

    #Computation of the gradient
    gradB = DMRG_compute_gradient(A,b,B,sel,pos0,posL,N)

    #Initialization of r and p
    r=-gradB
    p.append(r)

    #Loop over Npass
    while( tl.norm(r,1) > cutoff and i<Npass ): 
        if(sel==0 or (sel==1 and sel>pos0) ):
            pA.append(tl.tenalg.contract(p[i],(0,1,2),A,(0,2,4)))
            pAp.append(float(tl.tenalg.contract(p[i],(0,1,2),pA[i],(0,1,2)))) 
            r_tab.append(tl.tenalg.contract(r,(0,1,2),r,(0,1,2)))
            alpha=r_tab[i]/pAp[i]
            new_r=r-alpha*pA[i]
            if tl.norm(new_r,1) > cutoff :
                r_tab.append(tl.tenalg.contract(new_r,(0,1,2),new_r,(0,1,2)))
                beta=r_tab[i+1]/r_tab[i]
                p.append(new_r+beta*p[i])
            r=new_r
        elif( (sel==posL-1 and sel< pos0) or (sel==posL and sel> pos0) ):
            pA.append(tl.tenalg.contract(A,(0,2,4,6),p[i],(0,1,2,3)))
            pAp.append(float(tl.tenalg.contract(p[i],(0,1,2,3,4),pA[i],(0,1,2,3,4)))) 
            r_tab.append(tl.tenalg.contract(r,(0,1,2,3,4),r,(0,1,2,3,4)))
            alpha=r_tab[i]/pAp[i]
            new_r=r-alpha*pA[i]
            if tl.norm(new_r,1) > cutoff :
                r_tab.append(tl.tenalg.contract(new_r,(0,1,2,3,4),new_r,(0,1,2,3,4)))
                beta=r_tab[i+1]/r_tab[i]
                p.append(new_r+beta*p[i])
            r=new_r
        elif( (sel==posL and sel< pos0) or (sel==posL+1 and sel > pos0)  ):
            pA.append(tl.tenalg.contract(A,(0,2,4,6),p[i],(0,1,3,4)).transpose((0,1,4,2,3)))
            pAp.append(float(tl.tenalg.contract(p[i],(0,1,2,3,4),pA[i],(0,1,2,3,4)))) 
            r_tab.append(tl.tenalg.contract(r,(0,1,2,3,4),r,(0,1,2,3,4)))
            alpha=r_tab[i]/pAp[i]
            new_r=r-alpha*pA[i]
            if tl.norm(new_r,1) > cutoff :
                r_tab.append(tl.tenalg.contract(new_r,(0,1,2,3,4),new_r,(0,1,2,3,4)))
                beta=r_tab[i+1]/r_tab[i]
                p.append(new_r+beta*p[i])
            r=new_r
        elif( (sel==N-2 and sel<pos0 ) or sel==N-1):
            pA.append(tl.tenalg.contract(p[i],(0,1,2),A,(0,2,4)))
            pAp.append(float(tl.tenalg.contract(p[i],(0,1,2),pA[i],(0,1,2)))) 
            r_tab.append(tl.tenalg.contract(r,(0,1,2),r,(0,1,2)))
            alpha=r_tab[i]/pAp[i]
            new_r=r-alpha*pA[i]
            if tl.norm(new_r,1) > cutoff :
                r_tab.append(tl.tenalg.contract(new_r,(0,1,2),new_r,(0,1,2)))
                beta=r_tab[i+1]/r_tab[i]
                p.append(new_r+beta*p[i])
            r=new_r
        else:
            pA.append(tl.tenalg.contract(p[i],(0,1,2,3),A,(0,2,4,6)))
            pAp.append(float(tl.tenalg.contract(p[i],(0,1,2,3),pA[i],(0,1,2,3)))) 
            r_tab.append(tl.tenalg.contract(r,(0,1,2,3),r,(0,1,2,3)))
            alpha=r_tab[i]/pAp[i]
            new_r=r-alpha*pA[i]
            if tl.norm(new_r,1) > cutoff :
                r_tab.append(tl.tenalg.contract(new_r,(0,1,2,3),new_r,(0,1,2,3)))
                beta=r_tab[i+1]/r_tab[i]
                p.append(new_r+beta*p[i])
            r=new_r
        B=B+alpha*p[i]
        i=i+1

    return B

def gradient_descent_fixed_stepsize(A,b,B,sel,pos0,posL,N,alpha,nbTraining,cutoff,Npass,Phi_tilde,loss_function,label):
    """
    Gradient descent with fixed stepsize

    Optimizes the tensor B with a simple gradient descent with fixed stepsize

    Parameters
    ----------
    A : numpy array
        the tensor A constructed for the conjugate gradient
    b : numpy array
        the tensor b constructed for the conjugate gradient
    B : numpy array
        the tensor B to optimize
    sel : int
        the number of one of the tensor selected to be in B
    pos0 : int
        the number of the other tensor selected to be in B
    posL : int
        the position of the label index
    N : int
        the number of tensor in the MPS form which is equal to the number of input
    alpha : double
        the learning rate
    nbTraining : int
        the number of pictures used to train
    cutoff : double
        The truncation error goal when optimizing the MPS
    Npass : int
        The number of maximum pass to do at each step 
    Phi_tilde : numpy array 
        the tensor Phi_tilde 
    loss_function : str 
        the loss function chosen for the optimization 
    label : list of array 
        the labels of all the pictures 

    Return
    ------
    B : numpy array
        the new value of B after Adam
    """
    
    i=0
    gradB=0
    while( (i==0 or tl.norm(gradB,2) > cutoff ) and i<Npass):
        gradB = DMRG_compute_gradient(A,b,B,sel,pos0,posL,N,Phi_tilde,loss_function,label)
        B=B-alpha*gradB/nbTraining
        i=i+1
    return B

def Adam(A,b,B,sel,pos0,posL,N,alpha,nbTraining,cutoff,Npass,Phi_tilde,loss_function,label,beta1 = 0.9, beta2 = 0.999,eps=10**(-10)):
    """
    Adaptative Moment Estimation optimizer

    Optimizes the tensor B with Adam. For more details on the Adam optimizer
    see docs/1412.6980.pdf

    Parameters
    ----------
    A : numpy array
        the tensor A constructed for the conjugate gradient
    b : numpy array
        the tensor b constructed for the conjugate gradient
    B : numpy array
        the tensor B to optimize
    sel : int
        the number of one of the tensor selected to be in B
    pos0 : int
        the number of the other tensor selected to be in B
    posL : int
        the position of the label index
    N : int
        the number of tensor in the MPS form which is equal to the number of input
    alpha : double
        the learning rate
    nbTraining : int
        the number of pictures used to train
    cutoff : double
        The truncation error goal when optimizing the MPS
    Npass : int
        The number of maximum pass to do at each step 
    Phi_tilde : numpy array 
        the tensor Phi_tilde 
    loss_function : str 
        the loss function chosen for the optimization 
    label : list of array 
        the labels of all the pictures 
    beta1 : double
        parameter of Adam algorithm between 0 and 1
    beta 2 : double
        parameter of Adam algorithm between 0 and 1
    eps : double (optional)
        to avoid problems of division per zero (default is 10e-10)

    Return
    ------
    B : numpy array
        the new value of B after Adam
    """
    i=0
    gradB=0 ; vdB = 0; sdB = 0
    while( (i==0 or tl.norm(gradB,2) > cutoff ) and i<Npass):
        i=i+1
        gradB = DMRG_compute_gradient(A,b,B,sel,pos0,posL,N,Phi_tilde,loss_function,label)

        # Moving average of the gradients
        vdB = beta1*vdB +(1-beta1)*gradB

        # Compute bias-corrected first moment estimate
        v_cor = vdB/(1-beta1**i)

        # Moving average of the squared gradients
        sdB = beta2*sdB +(1-beta2)*np.power(gradB,2)

        # Compute bias-corrected second raw moment estimate
        s_cor = sdB/(1-beta2**i)

        # Update parameter
        B=B-alpha*v_cor/(np.sqrt(s_cor)+eps)
        
    return B

