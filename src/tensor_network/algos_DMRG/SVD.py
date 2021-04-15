import numpy as np
import tensorly as tl

#choix du sens de contraction des valeurs propres
def direction(u,s,v,sel,pos0,bond_dim,nmethod):
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

#Réalisation de la SVD de la matrice B ( après réorganisation ) 
def SVD_matB(B,sel,pos0,maxalpha,cutoff,nmethod):
        u, s, v = np.linalg.svd(B,full_matrices=False)
        bond_dim=min(np.sum(s>cutoff),maxalpha)
        s=s[0:bond_dim]
        u , v = direction(u,s,v,sel,pos0,bond_dim,nmethod)
        return (u , v, bond_dim)

#Réalisation de la SVD sur le tenseur B
def SVD_B(sel,pos0,B,posL,N,maxalpha,cutoff,nmethod):
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
    