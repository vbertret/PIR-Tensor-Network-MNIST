import numpy as np
import math
import gzip
import matplotlib.pyplot as plt

def make_dataset_easy():

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

    return (data,y)

def make_dataset_random(N,nbExample):
    ##### Creation Dataset #####

    middle=math.floor(nbExample/2)
    
    #Creation des images
    data=np.random.random_sample((nbExample,N))
    #Creation des labels
    y=np.zeros((nbExample,2)) ; y[0:middle,0] = 1 ; y[middle:nbExample,1]=1

    return (data,y)

def load_MNIST_dataset():

    #Training images
    f = gzip.open('data/external/train-images-idx3-ubyte.gz','r')

    image_size = 28
    num_images = 60000

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    train_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    train_data = train_data.reshape(num_images, image_size, image_size)

    f.close()

    #Test images
    f = gzip.open('data/external/t10k-images-idx3-ubyte.gz','r')

    image_size = 28
    num_images = 10000

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    test_data = test_data.reshape(num_images, image_size, image_size)

    f.close()

    #Training labels

    f = gzip.open('data/external/train-labels-idx1-ubyte.gz','r')

    num_images = 60000

    f.read(8)
    buf = f.read(1 * num_images)
    train_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    f.close()

    #Test labels

    f = gzip.open('data/external/t10k-labels-idx1-ubyte.gz','r')

    num_images = 10000

    f.read(8)
    buf = f.read(1 * num_images)
    test_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    f.close()

    return (train_data , test_data , train_labels , test_labels)

def convert_one_hot_encoding(labels):
    num_labels=labels.shape[0]
    num_class = np.max(labels) + 1
    new_labels=np.zeros((num_labels,num_class))

    for i in range(num_labels):
        new_labels[i,labels[i]]=1

    return new_labels

def make_MNIST_dataset_small():

    train_data , test_data , train_labels , test_labels = load_MNIST_dataset()

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

    
    np.save("data/processed/small_train_images.npy",new_train_data)

    return (new_train_data , new_test_data , train_labels , test_labels)


if __name__ == "__main__":
    train_data , test_data , train_labels , test_labels =load_MNIST_dataset()
    print(test_labels.shape)

    new_labels = convert_one_hot_encoding(train_labels)
    print(new_labels[0],train_labels[0])

    train_data , test_data , train_labels , test_labels =make_MNIST_dataset_small()
    plt.imshow(train_data[0])
    plt.show()