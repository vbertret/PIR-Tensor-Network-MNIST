import numpy as np
import tensorly as tl

def direction(u,s,v,sel,pos0,bond_dim,nmethod):
    """
    Makes the contraction with the eigen values after the Singular Value Decomposition

    If nmethod is 1, then the contraction goes in the same direction as the sweep whereas
    if nmethod is 2 , the contraction goes in the opposite direction.

    Parameters
    ---------
    u : numpy array
        one of the unitary matrix of the SVD
    s : list
        the list of eigen values of the SVD
    v : numpy array
        one of the unitary matrix of the SVD
    sel : int
        the number of one of the tensor selected to be in B
    pos0 : int
        the number of the other tensor selected to be in B
    bond_dim : int
        the number of eigen values to keep for the contraction
    nmethod : int
        the number to choose the direction
    
    Returns
    -------
    u : numpy array
        the new matrix u
    v : numpy array
        the new matrix v
    """

    if(nmethod==1):
        if(sel>pos0):
            v = v[0:bond_dim,:]  ; u =  u[:,0:bond_dim] @ np.diag(s)
        else:
            v = np.diag(s) @ v[0:bond_dim,:]  ; u =  u[:,0:bond_dim]
    if(nmethod==2):
        if(sel<pos0):
            v = v[0:bond_dim,:]  ; u =  u[:,0:bond_dim] @ np.diag(s)
        else:
            v = np.diag(s) @ v[0:bond_dim,:]  ; u =  u[:,0:bond_dim]
    return (u,v)

def SVD_matB(B,sel,pos0,maxalpha,cutoff,nmethod):
    """
    Makes the Singular Value Decomposition of the matrix B

    The tensor was already converted to a matrix. Therefore,
    the method do just the SVD, choose the bond dimension 
    according to the cutoff and the maximal value and make
    the contraction to have new values for the 2 tensors which
    belong to B.

    Parameters
    ----------
    B : numpy array
        The tensor B in matrix form
    sel : int
        The number of one of the tensor selected to be in B
    pos0 : int
        The number of the other tensor selected to be in B
    maxalpha : int
        The maximal value of the bond dimension
    cutoff : double
        The truncation error goal when optimizing the MPS
    nmethod : int
        The number to choose the direction

    Returns
    -------
    u : numpy array
        new values for one of the tensors which belong to B
    v : numpy array
        new values for one of the tensors which belong to B
    bond_dim : int
        the new bond dimension between u and v
    """

    u, s, v = np.linalg.svd(B,full_matrices=False)
    bond_dim=min(np.sum(s>cutoff),maxalpha)
    s=s[0:bond_dim]
    u , v = direction(u,s,v,sel,pos0,bond_dim,nmethod)
    return (u , v, bond_dim)

def SVD_B(sel,pos0,B,posL,N,maxalpha,cutoff,nmethod):
    """
    Makes the Singular Value Decomposition of the tensor B

    It converts the tensor B to a matrix, it does a 
    Singular Value Decomposition on it and then reconstructs 
    2 tensors which will replace the old tensors belonging to
    B.

    Parameters
    ----------
    sel : int
        The number of one of the tensor selected to be in B
    pos0 : int
        The number of the other tensor selected to be in B
    B : numpy array
        The tensor B in matrix form
    posL : int
        the position of the tensor with the label index
    N : int
        the number of tensor in the MPS form which is equal to the number of inputs
    maxalpha : int
        The maximal value of the bond dimension
    cutoff : double
        The truncation error goal when optimizing the MPS
    nmethod : int
        The number to choose the direction

    Returns
    -------
    u : numpy array
        new values for one of the tensors which belong to B
    v : numpy array
        new values for one of the tensors which belong to B
    """
    dim=B.shape
    if(sel==0 or (sel==1 and sel>pos0) ):
        B=B.reshape(dim[0],dim[1]*dim[2])
        u, v , bond_dim = SVD_matB(B,sel,pos0,maxalpha,cutoff,nmethod)
        v= v.reshape(bond_dim,2,dim[2]).transpose((1,0,2))
    elif( (sel==posL-1 and sel< pos0) or (sel==posL and sel> pos0) ):
        B=B.reshape(dim[0]*dim[1],dim[2]*dim[3]*dim[4])
        u, v , bond_dim = SVD_matB(B,sel,pos0,maxalpha,cutoff,nmethod)
        u = u.reshape(2,dim[1],bond_dim)
        v= v.reshape(bond_dim,2,dim[3],dim[4]).transpose((1,0,2,3))
    elif( (sel==N-2 and sel< pos0 )or sel==N-1):
        B=B.reshape(dim[0]*dim[1],dim[2])
        u, v , bond_dim= SVD_matB(B,sel,pos0,maxalpha,cutoff,nmethod)
        u = u.reshape(2,dim[1],bond_dim)
        v= v.reshape(bond_dim,dim[2]).transpose((1,0))
    elif( (sel==posL and sel< pos0) or (sel==posL+1 and sel > pos0) ):
        B=B.reshape(dim[0]*dim[1]*dim[2],dim[3]*dim[4])
        u, v , bond_dim= SVD_matB(B,sel,pos0,maxalpha,cutoff,nmethod)
        u = u.reshape(2,dim[1],dim[2],bond_dim).transpose((0,1,3,2))
        v= v.reshape(bond_dim,2,dim[4]).transpose((1,0,2))
    else:
        B=B.reshape(dim[0]*dim[1],dim[2]*dim[3])
        u, v , bond_dim= SVD_matB(B,sel,pos0,maxalpha,cutoff,nmethod)
        u = u.reshape(2,dim[1],bond_dim)
        v= v.reshape(bond_dim,2,dim[3]).transpose((1,0,2))
    return (u,v)

if __name__ == "__main__":
    #test 1
    sel=5 ; pos0=6 ; posL=8 ; N=9 ; maxalpha = 10 ; cutoff=10**(-10)
    B=np.random.randn(2,10,2,20)
    u1 , v1 = SVD_B(sel,pos0,B,posL,N,maxalpha,cutoff,1)
    u2 , v2 = SVD_B(sel,pos0,B,posL,N,maxalpha,cutoff,2)
    print("##### U1 #####")
    print(u1.shape) 
    print("##### V1 #####")
    print(v1.shape)
    print("##### U2 #####")
    print(u2.shape)
    print("##### V2 #####") 
    print(v2.shape)
    