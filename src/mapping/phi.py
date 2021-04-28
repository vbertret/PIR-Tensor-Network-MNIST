import numpy as np

def phi(x):
    """
    Mapping function phi

    It mapped a pixel (a double) in another space of larger dimension.
    For more details, read the 2 next documents : 
    -Vladimir Vapnik. The Nature of Statistical Learning Theory. Springer-Verlag New York, 2000.
    -W.Waegeman, T. Pahikkala, A. Airola, T. Salakoski, M. Stock, and B. De Baets. A kernel-based
    framework for learning graded relations from data. Fuzzy Systems, IEEE Transactions on, 20
    (6):1090–1101, Dec 2012.

    Parameter
    ---------
    x : double
        one pixel with value from 0 to 1
    
    Return
    -------
    array:
        the value of the mapping function evaluated at point x
    """
    return np.array([np.cos((np.pi*x)/2),np.sin((np.pi*x)/2)])

def Phi(img):
    """
    Mapping function Phi

    It mapped all the pixel of a picture in another space of larger dimension.
    For more details, go to the method above.

    Parameter
    ---------
    img : array
        all the pixel of a picture with value from 0 to 1
    
    Return
    -------
    array:
        the value of the mapping function evaluated at each pixel of the picture
    """

    return phi(img.reshape(-1,))
    
    

