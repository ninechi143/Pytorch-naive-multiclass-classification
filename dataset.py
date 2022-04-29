import torch

from torch.utils.data import Dataset
import torchvision

import random
import numpy as np



class train_dataset(Dataset):

    def __init__(self , transform = None):
        # we use MNIST as our example
        cifar10 = torchvision.datasets.CIFAR10(root = "./data" , 
                                                    train = True,
                                                    transform=None,
                                                    download=True)

        train_x = cifar10.data / 255.0     # 50000,32,32,3
        train_y = cifar10.targets          # 50000,

        self.mean , self.std = np.expand_dims(train_x.mean(axis = (0,1,2)) , axis = (1,2)) , np.expand_dims(train_x.std(axis = (0,1,2)) , axis = (1,2))
    


        self.inputs , self.labels = self.axis_process(train_x , train_y)
        # print(self.inputs.shape)   # 50000 , 3 ,  32 , 32
        # print(self.labels.shape)  # 50000 , 1

        print(self.labels.shape)
    
        self.n_samples = self.inputs.shape[0]    

        self.transform = transform


    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):

        if self.transform:
            inputs = self.transform(self.inputs[index] , self.mean , self.std)
        else:
            inputs = self.inputs[index]

        targets = self.labels[index]

        # note that targets is not one-hot encoding, besides, CrossEntropyLoss() need the dtype of targets to be LongTensor
        return torch.from_numpy(inputs).float() , torch.from_numpy(targets).long()



    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


    def axis_process(self , train_x , train_y):
        # data: from 50000 , 32 , 32 , 3  -> 50000 , 3 , 32 , 32
        # labels: from 50000, -> 50000 , 1

        return np.transpose(train_x , (0,3,1,2)) , np.array(train_y)[: , np.newaxis]




# simply normalize
class normalize():

    def __call__(self, inputs , mean , std):

        return (inputs - mean) / (std + 1e-8)





if __name__ == "__main__":


    a = torchvision.datasets.CIFAR10(root = "./data" , 
                                    train = True,
                                    transform=None,
                                    download=True)

    b = torchvision.datasets.CIFAR10(root = "./data" , 
                                    train = False,
                                    transform=None)

    
    ax = a.data
    ay = a.targets

    print(type(ax) , type(ay))     
    print(ax.shape)



    bx = b.data
    by = b.targets
  
    print(bx.shape)


    print(ay[5])   # not yet one-hot coding, so it is just a single number
    print(ax[5])   # not yet normalizer, so the range is from 0 to 255


    # D = train_dataset()
    # print(len(D))

    # (q , y) = D[0]
    # print(type(q))
    # print(type(y))