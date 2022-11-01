import torch
import torchvision as tv
from torch.utils.data import Dataset, DataLoader


meerkat_dir = "/idia/projects/hippo/Koketso/meerkat"
dogbreed_dir = "/idia/projects/hippo/Koketso/dog_breeds"
galaxyzoo_dir = "/idia/projects/hippo/Koketso/galaxy_zoo_sub"
hand_dir = "/idia/projects/hippo/Koketso/Train_Alphabet"

class Custom(Dataset):
    def __init__(self,x,resize = 224,crop = 224,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
        self.x = x
        self.resize = resize
        self.crop = crop
        self.mean = mean
        self.std = std
        self.transform = tv.transforms.Compose([
                            #tv.transforms.ToPILImage(),
                            #tv.transforms.Resize((424,424)),
                            tv.transforms.Resize(self.resize),
                            tv.transforms.CenterCrop(self.crop),          
                            tv.transforms.ToTensor(),
                            #tv.transforms.Grayscale(),
                            #tv.transforms.Normalize(mean=self.mean, std=self.std)
                            ])
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,index):
        
        image = self.x[index]
        
        #image = np.array(image[0].getdata())
        #plt.imshow(image)
        
        #image = self.tv.transforms.ToTensor(image)
        x = self.transform(image[0])
        # defined the transform below
        return x

    
class Custom_labelled(Dataset):
    def __init__(self,x,y,resize = 224,crop = 224,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
        self.x = x
        self.y = y
        self.resize = resize
        self.crop = crop
        self.mean = mean
        self.std = std
        self.transform = tv.transforms.Compose([
                            #tv.transforms.ToPILImage(),
                            #tv.transforms.Resize((424,424)),
                            tv.transforms.Resize(self.resize),
                            tv.transforms.CenterCrop(self.crop),          
                            tv.transforms.ToTensor(),
                            #tv.transforms.Grayscale(),
                            #tv.transforms.Normalize(mean=self.mean, std=self.std)
                            ])
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,index):
        
        image = self.x[index]
        target = self.y[index]
        
        #image = np.array(image[0].getdata())
        #plt.imshow(image)
        
        #image = self.tv.transforms.ToTensor(image)
        x = self.transform(image[0])
        
        # defined the transform below
        return x,target
    
    
def dataset(data):
    if data == 'meerkat':
        Dir = "/idia/projects/hippo/Koketso/meerkat"
    elif data =="dog_breed":
        Dir = "/idia/projects/hippo/Koketso/dog_breeds"
    elif data == "galaxy_zoo":
        Dir = "/idia/projects/hippo/Koketso/galaxy_zoo_sub"
    elif data == "hand_alphabet":
        Dir = "/idia/projects/hippo/Koketso/Train_Alphabet"
    else:
        Dir = data
    return tv.datasets.ImageFolder(Dir)

def transformed(dataset):
    return Custom(dataset) 


def plot_filters_single_channel(t):

    #kernels depth * number of kernels
    nplots = t.shape[0]*t.shape[1]
    ncols = 12

    nrows = 1 + nplots//ncols
    #convert tensor to numpy image
    npimg = np.array(t.numpy(), np.float32)

    count = 0
    fig = plt.figure(figsize=(ncols, nrows))

    #looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    plt.tight_layout()
    plt.show()

def plot_filters_multi_channel(t):

    #get the number of kernals
    num_kernels = t.shape[0]

    #define number of columns for subplots
    num_cols = 12
    #rows = num of kernels
    num_rows = num_kernels

    #set the figure size
    fig = plt.figure(figsize=(num_cols,num_rows))

    #looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)

        #for each kernel, we convert the tensor to numpy
        npimg = np.array(t[i].numpy(), np.float32)
        #standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.savefig('myimage.png', dpi=100)
    plt.tight_layout()
    plt.show()


def plot_weights(model, layer_num, single_channel = True, collated = False):

  #extracting the model features at the particular layer number
  layer = model.features[layer_num]

  #checking whether the layer is convolution layer or not
  if isinstance(layer, torch.nn.Conv2d):
    #getting the weight tensor data
    weight_tensor = model.features[layer_num].weight.data

    if single_channel:
      if collated:
        plot_filters_single_channel_big(weight_tensor)
      else:
        plot_filters_single_channel(weight_tensor)

    else:
      if weight_tensor.shape[1] == 3:
        plot_filters_multi_channel(weight_tensor)
      else:
        print("Can only plot weights with three channels with single channel = False")

  else:
    print("Can only visualize layers which are convolutional")

#visualize weights for alexnet - first conv layer
#plot_weights(resnet50, 0, single_channel = False)

    
