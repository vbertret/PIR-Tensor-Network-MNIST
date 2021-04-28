import numpy as np
import random
import gzip
import math
import os

def make_dataset_easy():
    """
    Creation of a very easy dataset

    Creation of a dataset with pictures of 3 by 3.
    Pictures with white pixel on the left are labelling with 0
    and other pictures are labelling with 1.

    Returns
    -------
    data : list
        list of pictures
    y : list
        list of labels in one hot encoding format
    """

    #Number of examples
    Nt=8
    data=np.zeros((Nt,3,3))

    #White square on left
    data[0]=np.array([[1,0,0],[1,0,0],[1,0,0]])
    data[1]=np.array([[1,0,0],[0,0,0],[0,0,0]])
    data[2]=np.array([[0,0,0],[1,0,0],[0,0,0]])
    data[3]=np.array([[0,0,0],[0,0,0],[1,0,0]])

    #White square on right
    data[4]=np.array([[0,0,1],[0,0,1],[0,0,1]])
    data[5]=np.array([[0,0,1],[0,0,0],[0,0,0]])
    data[6]=np.array([[0,0,0],[0,0,1],[0,0,0]])
    data[7]=np.array([[0,0,0],[0,0,0],[0,0,1]])

    #Creation of the labels
    y=np.array([[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1]])

    return (data,y)

def make_dataset_random(N,nbExample,nbClass):
    """
    Creation of a random dataset

    Creation of a dataset with input randomly drawn from
    a uniform distribution between 0 and 1. Half of the inputs
    are affected to the label 0 and the other half to 1.

    Parameters
    ----------
    N : int
        the number of features per input
    nbExample : int
        the number of example
    nbClass : int
        the number of Class

    Returns
    -------
    data : list
        list of pictures
    y : list
        list of labels in one hot encoding format
    """

    #Set the seed to have the same dataset each time for the same parameters
    np.random.seed(123)

    #Creation of inputs
    data=np.random.random_sample((nbExample,N))

    #Creation of labels
    y=np.zeros((nbExample,nbClass)) 
    for ex in range(nbExample):
        y[ex,random.randint(0,nbClass-1)]=1

    return (data,y)

def load_MNIST_dataset(path):
    """
    Loads the MNIST dataset.

    Loads the data and the labels of the training and test set
    from the MNIST dataset.

    Parameters
    ----------
    path : str
        path of the folder data in which there is a folder external
        with all the files downloaded on the website http://yann.lecun.com/exdb/mnist/

    Returns
    -------
    train_data : list
        list of pictures from the MNIST training set
    test_data : list
        list of pictures from the MNIST test set
    train_labels : list
        list of labels from the MNIST training set
    test_labels : list
        list of labels from the MNIST test set
    """

    #Training images
    f = gzip.open(f'{path}/external/train-images-idx3-ubyte.gz','r')

    image_size = 28
    num_images = 60000

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    train_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    train_data = train_data.reshape(num_images, image_size, image_size)/256

    f.close()

    #Test images
    f = gzip.open(f'{path}/external/t10k-images-idx3-ubyte.gz','r')

    image_size = 28
    num_images = 10000

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    test_data = test_data.reshape(num_images, image_size, image_size)/256

    f.close()

    #Training labels

    f = gzip.open(f'{path}/external/train-labels-idx1-ubyte.gz','r')

    num_images = 60000

    f.read(8)
    buf = f.read(1 * num_images)
    train_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    f.close()

    #Test labels

    f = gzip.open(f'{path}/external/t10k-labels-idx1-ubyte.gz','r')

    num_images = 10000

    f.read(8)
    buf = f.read(1 * num_images)
    test_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    f.close()

    return (train_data , test_data , train_labels , test_labels)

def convert_one_hot_encoding(labels):
    """
    Converts labels in the one hot encoding format

    Parameters
    ----------
    labels : list
        Some labels. The shape needs to be (nbTraining,1)

    Returns
    -------
    new_labels : list
        Same labels with one encoding format
    """

    #Retrieval of the number of labels
    num_labels=labels.shape[0]

    #Computation of the number of classes
    num_class = np.max(labels) + 1
    new_labels=np.zeros((num_labels,num_class),dtype=np.int64)

    #Convertion of the labels
    for i in range(num_labels):
        new_labels[i,labels[i]]=1

    return new_labels

def make_MNIST_dataset_small(path):
    """
    Creation of a smaller MNIST dataset.

    Uses the method load_MNIST_dataset to load the dataset and
    reduce the size of the pictures by 2 from 28*28 to 14*14.
    It returns the training and test set but it also saves 
    the new pictures in the folder processed in a numpy
    format.

    Parameters
    ----------
    path : str
        path of the folder data in which there is a folder external
        with all the files downloaded on the website http://yann.lecun.com/exdb/mnist/

    Returns
    -------
    train_data : list
        list of pictures from the MNIST training set
    test_data : list
        list of pictures from the MNIST test set
    train_labels : list
        list of labels from the MNIST training set
    test_labels : list
        list of labels from the MNIST test set
    """

    #Loading the images
    train_data , test_data , train_labels , test_labels = load_MNIST_dataset(path)

    #Training images
    num_images = 60000

    new_train_data=np.zeros((num_images,14,14))
    for k in range(num_images):
        for i in range(0,28,2):
            for j in range(0,28,2):
                new_train_data[k,int(i/2),int(j/2) ] = (train_data[k,i,j] + train_data[k,i+1,j] + train_data[k,i,j+1] + train_data[k,i+1,j+1])/4

    #Training images
    num_images = 10000

    new_test_data=np.zeros((num_images,14,14))
    for k in range(num_images):
        for i in range(0,28,2):
            for j in range(0,28,2):
                new_test_data[k,int(i/2),int(j/2) ] = (test_data[k,i,j] + test_data[k,i+1,j] + test_data[k,i,j+1] + test_data[k,i+1,j+1])/4

    
    #Saving the pictures in the folder processed
    np.save(f'{path}/processed/small_train_images.npy',new_train_data)
    np.save(f'{path}/processed/small_test_images.npy',new_test_data)

    return (new_train_data , new_test_data , train_labels , test_labels)

def load_MNIST_dataset_small(path):
    """
    Loads the small MNIST dataset.

    Loads the data and the labels of the training and test set
    from the small MNIST dataset. You have to execute the method
    make_MNIST_dataset_small before.

    Parameters
    ----------
    path : str
        path of the folder data. 

    Returns
    -------
    train_data : list
        list of pictures from the MNIST training set
    test_data : list
        list of pictures from the MNIST test set
    train_labels : list
        list of labels from the MNIST training set
    test_labels : list
        list of labels from the MNIST test set
    """
    
    #Training and Test images

    train_data = np.load(f'{path}/processed/small_train_images.npy')
    test_data = np.load(f'{path}/processed/small_test_images.npy')
    
    #Training labels

    f = gzip.open(f'{path}/external/train-labels-idx1-ubyte.gz','r')

    num_images = 60000

    f.read(8)
    buf = f.read(1 * num_images)
    train_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    f.close()

    #Test labels

    f = gzip.open(f'{path}/external/t10k-labels-idx1-ubyte.gz','r')

    num_images = 10000

    f.read(8)
    buf = f.read(1 * num_images)
    test_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    f.close()

    return (train_data , test_data , train_labels , test_labels)
    
def load_subpart_MNIST_dataset_small(path,digits):
    """
    Loads a subpart of the small MNIST dataset.

    Loads a subpart (only some digits) of the data and the labels of the training and test set
    from the small MNIST dataset. You have to execute the method
    make_MNIST_dataset_small before.

    Parameters
    ----------
    path : str
        path of the folder data. 

    Returns
    -------
    train_data : list
        list of pictures from the MNIST training set
    test_data : list
        list of pictures from the MNIST test set
    train_labels : list
        list of labels from the MNIST training set
    test_labels : list
        list of labels from the MNIST test set
    """

    train_data , test_data , train_labels , test_labels = load_MNIST_dataset_small(path)

    train_index = [i for i in range(len(train_data)) if train_labels[i] in digits]
    test_index = [i for i in range(len(test_data)) if test_labels[i] in digits]

    return (train_data[train_index] , test_data[test_index] , train_labels[train_index] , test_labels[test_index])
    

if __name__ == "__main__":
    path = "data"
    #train_data , test_data , train_labels , test_labels =load_MNIST_dataset(path)

    #new_labels = convert_one_hot_encoding(train_labels)
    #print(new_labels[0],train_labels[0])

    #make_MNIST_dataset_small(path)

    #train_data , test_data , train_labels , test_labels =load_MNIST_dataset_small(path)
    digits=[2,1]
    train_data , test_data , train_labels , test_labels = load_subpart_MNIST_dataset_small(path,digits)
    print(train_labels[0:10])
    print(test_labels[0:10])

    new_labels = convert_one_hot_encoding(train_labels)
    print(new_labels[0],train_labels[0])
