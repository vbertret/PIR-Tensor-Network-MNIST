import numpy as np
import tensorly as tl

def DMRG_creation_B_Atilde(W,sel,pos0):
        """
        Computation of the tensor B and the group of tensors A_tilde

        In the DMRG theory, B corresponds to the tensor we want to optimize
        and A_tilde are the rest of the tensors.

        Parameters
        ----------
        W : list of numpy array
            a list containing all the tensors of the MPS form
        sel : int
            the number of one of the tensor selected to be in B
        pos0 : int
            the number of the other tensor selected to be in B

        Returns
        -------
        B : numpy array
            the tensor B to optimize
        A_tilde : list of numpy array
            the other tensors
        """

        #Selection of the tensors to be in B
        A_tilde=W.copy() ; B_vec=[] ; B_vec.append(A_tilde.pop(sel))
        if(sel<pos0) : B_vec.append(A_tilde.pop(pos0-1)) 
        else : B_vec.append(A_tilde.pop(pos0))
                       
        #Construction of B : contraction of the 2 tensors
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
    
def DMRG_creation_phi_tilde12(A_tilde,Phi,sel,pos0,n,Min,N,nbTraining):
    """
    Computation of the tensor Phi_tilde1 and Phi_tilde2

    Phi_tilde1 and Phi_tilde2 are the tensors respectively on the left and on the right of the tensor B.
    They are making with contraction between left parts of Phi and A_tilde for Phi_tilde1. And the same 
    with right parts for Phi_tilde2.

    Parameters
    ----------
    A_tilde : list of numpy array
        tensors which are not in B
    Phi : list of numpy array
        value of the decision function for the pixels with are not connected with B
    sel : int
        the number of one of the tensor selected to be in B
    pos0 : int
        the number of the other tensor selected to be in B
    n : int
        the number of the training example
    Min : int
        the minimum between sel and pos0
    N : int
        the number of tensor in the MPS form which is equal to the number of inputs
    nbTraining : nt
        the number of training example

    Returns
    -------
    Phi_tilde1 : numpy array
        the tensor Phi_tilde1
    Phi_tilde2 : numpy array
        the tensor Phi_tilde2
    """
    Phi_tilde1=0 ; Phi_tilde2=0

    #Initialization of the static variables of the function (needeed so that the algorithm scales linearly with N)
    if(n==0 and sel == 0 and pos0 == 1):
        DMRG_creation_phi_tilde12.Phi_tilde1_table_to_right=[0]*nbTraining
        DMRG_creation_phi_tilde12.Phi_tilde1_table_to_left=[[0 for i in range(N-2)] for j in range(nbTraining)]
        DMRG_creation_phi_tilde12.Phi_tilde2_table_to_right=[[0 for i in range(N-2)] for j in range(nbTraining)]
        DMRG_creation_phi_tilde12.Phi_tilde2_table_to_left=[0]*nbTraining
    
    ##Construction of Phi_tilde1 to the right
    if(sel==1 and sel<pos0):
        DMRG_creation_phi_tilde12.Phi_tilde1_table_to_right[n]=tl.tenalg.mode_dot(A_tilde[0],Phi[:,0],0) #contractMPS(A_tilde[0:Min],Phi[:,:Min])
        Phi_tilde1=DMRG_creation_phi_tilde12.Phi_tilde1_table_to_right[n]

    if(sel!=0 and sel!=1 and sel<pos0):
        DMRG_creation_phi_tilde12.Phi_tilde1_table_to_right[n]=tl.tenalg.contract(A_tilde[Min-1],1,DMRG_creation_phi_tilde12.Phi_tilde1_table_to_right[n],0)
        DMRG_creation_phi_tilde12.Phi_tilde1_table_to_right[n]=tl.tenalg.mode_dot(DMRG_creation_phi_tilde12.Phi_tilde1_table_to_right[n],Phi[:,Min-1],0)
        Phi_tilde1=DMRG_creation_phi_tilde12.Phi_tilde1_table_to_right[n]

    ##Construction of Phi_tilde1 to the left
    if(sel==(N-1)):
        DMRG_creation_phi_tilde12.Phi_tilde1_table_to_left[n][0]=tl.tenalg.mode_dot(A_tilde[0],Phi[:,0],0)
        for i in range(1,N-2) :
            DMRG_creation_phi_tilde12.Phi_tilde1_table_to_left[n][i]=tl.tenalg.contract(A_tilde[i],1,DMRG_creation_phi_tilde12.Phi_tilde1_table_to_left[n][i-1],0)
            DMRG_creation_phi_tilde12.Phi_tilde1_table_to_left[n][i]=tl.tenalg.mode_dot(DMRG_creation_phi_tilde12.Phi_tilde1_table_to_left[n][i],Phi[:,i],0)
        Phi_tilde1=DMRG_creation_phi_tilde12.Phi_tilde1_table_to_left[n][N-3]

    if(sel>pos0 and  not(sel==1 and sel>pos0)):
        Phi_tilde1=DMRG_creation_phi_tilde12.Phi_tilde1_table_to_left[n][sel-2]

    ##Construction of Phi_tilde2 to the right
    if(sel==0):
        DMRG_creation_phi_tilde12.Phi_tilde2_table_to_right[n][N-3]=tl.tenalg.mode_dot(A_tilde[N-3],Phi[:,N-3],0)
        for i in range(N-4,-1,-1) :
            DMRG_creation_phi_tilde12.Phi_tilde2_table_to_right[n][i]=tl.tenalg.contract(A_tilde[i],2,DMRG_creation_phi_tilde12.Phi_tilde2_table_to_right[n][i+1],0)
            DMRG_creation_phi_tilde12.Phi_tilde2_table_to_right[n][i]=tl.tenalg.mode_dot(DMRG_creation_phi_tilde12.Phi_tilde2_table_to_right[n][i],Phi[:,i],0)
        Phi_tilde2=DMRG_creation_phi_tilde12.Phi_tilde2_table_to_right[n][0]

    if(sel!=0 and sel!=(N-2) and sel<pos0):
        Phi_tilde2=DMRG_creation_phi_tilde12.Phi_tilde2_table_to_right[n][sel]

    ##Construction of Phi_tilde2 to the left
    if(sel==N-2 and sel>pos0):
        DMRG_creation_phi_tilde12.Phi_tilde2_table_to_left[n]=tl.tenalg.mode_dot(A_tilde[N-3],Phi[:,N-3],0)
        Phi_tilde2=DMRG_creation_phi_tilde12.Phi_tilde2_table_to_left[n]

    if(sel != N-1 and sel!=N-2 and sel>pos0):
        DMRG_creation_phi_tilde12.Phi_tilde2_table_to_left[n]=tl.tenalg.contract(A_tilde[sel-1],2,DMRG_creation_phi_tilde12.Phi_tilde2_table_to_left[n],0)
        DMRG_creation_phi_tilde12.Phi_tilde2_table_to_left[n]=tl.tenalg.mode_dot(DMRG_creation_phi_tilde12.Phi_tilde2_table_to_left[n],Phi[:,sel-1],0)
        Phi_tilde2=DMRG_creation_phi_tilde12.Phi_tilde2_table_to_left[n]
        
    return (Phi_tilde1,Phi_tilde2)

def DMRG_creation_Phi_tilde(Phi_tilde1,Phi_tilde2,si,sel,pos0,posL,N):
    """
    Computation of the tensor Phi_tilde

    Phi_tilde is a tensor made of Phi_tilde1 , Phi_tilde 2 and the value
    of the decision function from pixels connecting with B. It's the 
    outer product of the 3 tensors. 

    Parameters
    ----------
    Phi_tilde1 : numpy array
        the tensor Phi_tilde1
    Phi_tilde2 : numpy array
        the tensor Phi_tilde2
    si : list of array
        the value of the decision function from pixels connecting with B
    sel : int
        the number of one of the tensor selected to be in B
    pos0 : int
        the number of the other tensor selected to be in B
    posL : int
        the position of the label index
    N : int
        the number of tensor in the MPS form which is equal to the number of inputs

    Returns
    -------
    Phi_tilde : numpy array
        the tensor Phi_tilde
    """

    if(sel==0 or (sel==1 and sel>pos0) ):
        Phi_tilde=np.multiply.outer(si[:,1],Phi_tilde2) #Phi_tilde2(alpha2,l) => Phi_tilde(s2,alpha2,l)
        Phi_tilde=np.multiply.outer(si[:,0],Phi_tilde) #=> Phi_tilde(s1,s2,alpha2,l)
    elif( (sel==posL-1 and sel< pos0) or (sel==posL and sel> pos0) ):
        Phi_tilde=np.multiply.outer(si[:,0],Phi_tilde1) #=> Phi_tilde1(alpha(i-1)) => Phi_tilde1(si,alpha(i-1))
        Phi_tilde=np.multiply.outer(Phi_tilde,si[:,1]) #=> Phi_tilde(si,alpha(i-1),s(i+1))
        Phi_tilde=np.multiply.outer(Phi_tilde,Phi_tilde2) #=>Phi_tilde(si,alpha(i-1),s(i+1),alpha(i+1))
    elif( (sel==posL and sel< pos0) or (sel==posL+1 and sel > pos0)  ):
        Phi_tilde=np.multiply.outer(si[:,0],Phi_tilde1) #=> Phi_tilde1(alpha(i-1)) => Phi_tilde1(si,alpha(i-1))
        Phi_tilde=np.multiply.outer(Phi_tilde,si[:,1]) #=> Phi_tilde(si,alpha(i-1),s(i+1))
        Phi_tilde=np.multiply.outer(Phi_tilde,Phi_tilde2) #=>Phi_tilde(si,alpha(i-1),s(i+1),alpha(i+1))
    elif( (sel==N-2 and sel<pos0 ) or sel==N-1):
        Phi_tilde=np.multiply.outer(si[:,0],Phi_tilde1) #Phi_tilde1(alpha(N-2),l) =>Phi_tilde1(s(N-1),alpha(N-2),l)
        Phi_tilde=np.multiply.outer(Phi_tilde,si[:,1]) #=> Phi_tilde(s(N-1),alpha(N-2),l,sN)
    else:
        Phi_tilde=np.multiply.outer(si[:,0],Phi_tilde1) #=> Phi_tilde1(alpha(i-1)) => Phi_tilde1(si,alpha(i-1))
        Phi_tilde=np.multiply.outer(Phi_tilde,si[:,1]) #=> Phi_tilde(si,alpha(i-1),s(i+1))
        Phi_tilde=np.multiply.outer(Phi_tilde,Phi_tilde2) #=>Phi_tilde(si,alpha(i-1),s(i+1),alpha(i+1),l)
    return Phi_tilde 

def DMRG_An_bn(Phi_tilde,Phi_tilde1,Phi_tilde2,si,sel,pos0,posL,N,y):
    """
    Computation of the tensor An and bn

    An and bn are tensors needeed to apply conjugate gradient. For more
    details, see the report-2.

    Parameters
    ----------
    Phi_tilde : numpy array
        the tensor Phi_tilde
    Phi_tilde1 : numpy array
        the tensor Phi_tilde1
    Phi_tilde2 : numpy array
        the tensor Phi_tilde2
    si : list of array
        the value of the decision function from pixels connecting with B
    sel : int
        the number of one of the tensor selected to be in B
    pos0 : int
        the number of the other tensor selected to be in B
    posL : int
        the position of the label index
    N : int
        the number of tensor in the MPS form which is equal to the number of inputs
    y : list
        the label associated with the pictures which created Phi_tilde, Phi_tilde1 and Phi_tilde2 (one hot encoding format)

    Returns
    -------
    An : numpy array
        the tensor An
    bn : numpy array
        the tensor bn
    """

    #Computation of the outer product for one of the value of Phi from a pixel connected with B
    temp1=si[:,0].reshape(2,1)
    An=np.dot(temp1,temp1.T)

    #Computation of the outer product for the second pixel
    temp1=si[:,1].reshape(2,1)
    An=np.multiply.outer(An,np.dot(temp1,temp1.T))
    
    #Computation of An and bn 
    if(sel==0 or (sel==1 and sel>pos0) ):
        temp2=np.dot(Phi_tilde2,Phi_tilde2.T)
        An=np.multiply.outer(An,temp2) #An -> (s1,s1,s2,s2,alpha2,alpha2)
        bn=tl.tenalg.mode_dot(Phi_tilde,y,3)
    elif( (sel==posL-1 and sel< pos0) or (sel==posL and sel> pos0) ):
        Phi_tilde1=Phi_tilde1.reshape(Phi_tilde1.shape[0],1) ; Phi_tilde2=Phi_tilde2.reshape(Phi_tilde2.shape[0],1)
        temp2=np.dot(Phi_tilde1,Phi_tilde1.T)
        An=np.multiply.outer(An,temp2)
        temp2=np.dot(Phi_tilde2,Phi_tilde2.T)
        An=np.multiply.outer(An,temp2) #An -> (sj,sj,sj+1,sj+1,alphaj-1,alphaj-1,alphaj+1,alphaj+1)
        An=An.transpose((0,1,4,5,2,3,6,7)) #An -> (sj,sj,alphaj-1,alphaj-1,sj+1,sj+1,alphaj+1,alphaj+1)
        bn=np.multiply.outer(Phi_tilde,y)
    elif( (sel==posL and sel< pos0) or (sel==posL+1 and sel > pos0)  ):
        Phi_tilde1=Phi_tilde1.reshape(Phi_tilde1.shape[0],1) ; Phi_tilde2=Phi_tilde2.reshape(Phi_tilde2.shape[0],1)
        temp2=np.dot(Phi_tilde1,Phi_tilde1.T)
        An=np.multiply.outer(An,temp2)
        temp2=np.dot(Phi_tilde2,Phi_tilde2.T)
        An=np.multiply.outer(An,temp2) #An -> (sj,sj,sj+1,sj+1,alphaj-1,alphaj-1,alphaj+1,alphaj+1)
        An=An.transpose((0,1,4,5,2,3,6,7)) #An -> (sj,sj,alphaj-1,alphaj-1,sj+1,sj+1,alphaj+1,alphaj+1)
        bn=np.multiply.outer(Phi_tilde,y)
        bn=bn.transpose((0,1,4,2,3))
    elif( (sel==N-2 and sel<pos0 ) or sel==N-1):
        temp2=np.dot(Phi_tilde1,Phi_tilde1.T)
        An=np.multiply.outer(An,temp2) #An -> (sN-1,sN-1,sN,sN,alphaN-2,alphaN-2)
        An=An.transpose((0,1,4,5,2,3)) #An -> (sN-1,sN-1,alphaN-2,alphaN-2,sN,sN)
        bn=tl.tenalg.mode_dot(Phi_tilde,y,2)
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
            bn=tl.tenalg.mode_dot(Phi_tilde,y,4)
        else:
            bn=tl.tenalg.mode_dot(Phi_tilde,y,2)
    return (An,bn)

def DMRG_compute_stuff_gradient(Phi_tilde1,Phi_tilde2,si,sel,pos0,posL,N,y):
    """
    Computation of the tensor An, bn and Phi_tilde

    For more details, see the 2 methods above.

    Parameters
    ----------
    Phi_tilde1 : numpy array
        the tensor Phi_tilde1
    Phi_tilde2 : numpy array
        the tensor Phi_tilde2
    si : list of array
        the value of the decision function from pixels connecting with B
    sel : int
        the number of one of the tensor selected to be in B
    pos0 : int
        the number of the other tensor selected to be in B
    posL : int
        the position of the label index
    N : int
        the number of tensor in the MPS form which is equal to the number of inputs
    y : list
        the label associated with the picture which has created Phi_tilde, Phi_tilde1 and Phi_tilde2 (one hot encoding format)

    Returns
    -------
    Phi_tilde : numpy array
        the tensor Phi_tilde
    An : numpy array
        the tensor An
    bn : numpy array
        the tensor bn
    """

    #Computation of Phi_tilde
    Phi_tilde = DMRG_creation_Phi_tilde(Phi_tilde1,Phi_tilde2,si,sel,pos0,posL,N)

    #Computation of An and bn
    An,bn = DMRG_An_bn(Phi_tilde,Phi_tilde1,Phi_tilde2,si,sel,pos0,posL,N,y)

    return (An,bn,Phi_tilde)

def DMRG_compute_cost(B,Phi_tilde,y,sel,pos0,posL,N,loss_function,eps=10**(-10)):
    """ 
    Computation of the value of the loss function for one picture

    Parameters
    ----------
    B : numpy array
        the tensor B to optimize
    Phi_tilde : numpy array
        the tensor Phi_tilde
    y : list
        the label associated with the pictures which created Phi_tilde (one hot encoding format)
    sel : int
        the number of one of the tensor selected to be in B
    pos0 : int
        the number of the other tensor selected to be in B
    posL : int
        the position of the label index
    N : int
        the number of tensor in the MPS form which is equal to the number of input
    loss_function : str
        the loss function chosen for the optimization
    eps : double (optional)
        to avoid problems of division per zero (default is 10e-10)
    
    Returns
    ------
    double:
        the value of the cost for the picture
    """

    #Computation of the decision function
    if(sel==0 or (sel==1 and sel>pos0) ):
        fl=tl.tenalg.contract(B,(0,1,2),Phi_tilde,(0,1,2))
    elif( (sel==posL-1 and sel< pos0) or (sel==posL and sel> pos0) ):
        fl=tl.tenalg.contract(B,(0,1,2,3),Phi_tilde,(0,1,2,3))
    elif( (sel==posL and sel< pos0) or (sel==posL+1 and sel > pos0)  ):
        fl=tl.tenalg.contract(B,(0,1,3,4),Phi_tilde,(0,1,2,3))
    elif( (sel==N-2 and sel<pos0 ) or sel==N-1):
        fl=tl.tenalg.contract(B,(0,1,2),Phi_tilde,(0,1,3))
    else:
        if(sel>posL):
            fl=tl.tenalg.contract(B,(0,1,2,3),Phi_tilde,(0,1,3,4))
        else:
            fl=tl.tenalg.contract(B,(0,1,2,3),Phi_tilde,(0,1,2,3))
    
    #Computation of the loss
    if loss_function =="quadratic":
        cost = (1/2)*np.power(fl-y,2)
    elif loss_function == "cross-entropy":
        cost = -y*np.log(np.exp(fl)/(sum(np.exp(fl)))+eps)
    elif loss_function =="log-quadratic":
        cost = (1/2)*np.log(np.power(fl-y,2)+1)

    #Summation over labels of the loss 
    return sum(cost)

def DMRG_compute_gradient(A,b,B,sel,pos0,posL,N,Phi_tilde=[],loss_function="quadratic",label=[],eps=10**(-10)):
    """ 
    Computation of the gradient.

    For the quadratic loss function, the method uses A,b to compute the gradient whereas for other, 
    the method uses all the value oh Phi_tilde for all the pictures.

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
    Phi_tilde : numpy array (optional)
        the tensor Phi_tilde (default is [])
    loss_function : str (optional)
        the loss function chosen for the optimization (default is 'quadratic')
    label : list of array (optional)
        the labels of all the pictures (default is [])
    eps : double (optional)
        to avoid problems of division per zero (default is 10e-10)

    Returns
    ------
    double:
        the value of the cost for the picture
    """

    if(loss_function == "quadratic"):
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
    elif(loss_function == "cross-entropy"):
        gradB=0
        for i in range(len(Phi_tilde)):
            elem=Phi_tilde[i]
            y=label[i,:]
            if(sel==0 or (sel==1 and sel>pos0) ):
                fl=tl.tenalg.contract(B,(0,1,2),elem,(0,1,2))
                gradf=np.exp(fl)/(sum(np.exp(fl))+eps)-y
                gradB+=tl.tenalg.mode_dot(elem,gradf,3) #=> gradB(s1,s2,alpha2)
            elif( (sel==posL-1 and sel< pos0) or (sel==posL and sel> pos0) ):
                fl=tl.tenalg.contract(B,(0,1,2,3),elem,(0,1,2,3))
                gradf=np.exp(fl)/(sum(np.exp(fl))+eps)-y
                gradB+=np.multiply.outer(elem,gradf) #gradB(si,alpha(i-1),s(i+1),alpha(i+1),l)
            elif( (sel==posL and sel< pos0) or (sel==posL+1 and sel > pos0)  ):
                fl=tl.tenalg.contract(B,(0,1,3,4),elem,(0,1,2,3))
                gradf=np.exp(fl)/(sum(np.exp(fl))+eps)-y
                gradf=np.multiply.outer(elem,gradf) #=>gradf(si,alpha(i-1),s(i+1),alpha(i+1),l)
                gradB+=gradf.transpose((0,1,4,2,3)) #=>gradB(si,alpha(i-1),l,s(i+1),alpha(i+1))
            elif( (sel==N-2 and sel<pos0 ) or sel==N-1):
                fl=tl.tenalg.contract(B,(0,1,2),elem,(0,1,3))
                gradf=np.exp(fl)/(sum(np.exp(fl))+eps)-y
                gradB+=tl.tenalg.mode_dot(elem,gradf,2) #=> gradB(s(N-1),alpha(N-2),sN)
            else:
                if(sel>posL):
                    fl=tl.tenalg.contract(B,(0,1,2,3),elem,(0,1,3,4))
                    gradf=np.exp(fl)/(sum(np.exp(fl))+eps)-y
                    gradB+=tl.tenalg.mode_dot(elem,gradf,2) #gradB(si,alpha(i-1),s(i+1),alpha(i+1))
                else:
                    fl=tl.tenalg.contract(B,(0,1,2,3),elem,(0,1,2,3))
                    gradf=np.exp(fl)/(sum(np.exp(fl))+eps)-y
                    gradB+=tl.tenalg.mode_dot(elem,gradf,4) #gradB(si,alpha(i-1),s(i+1),alpha(i+1))
    elif(loss_function == "log-quadratic"):
        gradB=0
        for i in range(len(Phi_tilde)):
            elem=Phi_tilde[i]
            y=label[i,:]
            if(sel==0 or (sel==1 and sel>pos0) ):
                fl=tl.tenalg.contract(B,(0,1,2),elem,(0,1,2))
                gradf=(fl-y)/(np.power(fl-y,2)+1)
                gradB+=tl.tenalg.mode_dot(elem,gradf,3) #=> gradB(s1,s2,alpha2)
            elif( (sel==posL-1 and sel< pos0) or (sel==posL and sel> pos0) ):
                fl=tl.tenalg.contract(B,(0,1,2,3),elem,(0,1,2,3))
                gradf=(fl-y)/(np.power(fl-y,2)+1)
                gradB+=np.multiply.outer(elem,gradf) #gradB(si,alpha(i-1),s(i+1),alpha(i+1),l)
            elif( (sel==posL and sel< pos0) or (sel==posL+1 and sel > pos0)  ):
                fl=tl.tenalg.contract(B,(0,1,3,4),elem,(0,1,2,3))
                gradf=(fl-y)/(np.power(fl-y,2)+1)
                gradf=np.multiply.outer(elem,gradf) #=>gradf(si,alpha(i-1),s(i+1),alpha(i+1),l)
                gradB+=gradf.transpose((0,1,4,2,3)) #=>gradB(si,alpha(i-1),l,s(i+1),alpha(i+1))
            elif( (sel==N-2 and sel<pos0 ) or sel==N-1):
                fl=tl.tenalg.contract(B,(0,1,2),elem,(0,1,3))
                gradf=(fl-y)/(np.power(fl-y,2)+1)
                gradB+=tl.tenalg.mode_dot(elem,gradf,2) #=> gradB(s(N-1),alpha(N-2),sN)
            else:
                if(sel>posL):
                    fl=tl.tenalg.contract(B,(0,1,2,3),elem,(0,1,3,4))
                    gradf=(fl-y)/(np.power(fl-y,2)+1)
                    gradB+=tl.tenalg.mode_dot(elem,gradf,2) #gradB(si,alpha(i-1),s(i+1),alpha(i+1))
                else:
                    fl=tl.tenalg.contract(B,(0,1,2,3),elem,(0,1,2,3))
                    gradf=(fl-y)/(np.power(fl-y,2)+1)
                    gradB+=tl.tenalg.mode_dot(elem,gradf,4) #gradB(si,alpha(i-1),s(i+1),alpha(i+1))
    
    return gradB

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

    (Phi_tilde1,Phi_tilde2) = DMRG_creation_phi_tilde12(A_tilde,Phi,sel,pos0,n,Min,N,nbTraining)

    print(Phi_tilde1)
