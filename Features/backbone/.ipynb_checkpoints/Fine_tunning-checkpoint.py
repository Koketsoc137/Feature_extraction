import torch
from byol_pytorch import BYOL
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import importlib
import Features.backbone.Custom as Custom
from IPython.display import display,clear_output
import torchvision as tv
import pickle
import kornia.augmentation as K
import kornia
import Features.backbone.MiraBest as mb

import Features.backbone.Test as test
import time
importlib.reload(Custom)
import random
import wandb

#The code parameters:
#augmentation probabilitiees
g_p = 0.3
v_p = 0
h_p = 0
g_r = 0.1
r_r = 1
r_c = 1

#Dataset arguments
batch_size = 100

resize = 300

#training image floder
galaxyzooq_dir = "/idia/projects/hippo/Koketso/galaxyzoo/galaxy_zoo"
#Class validation image folder
galaxyzooq_dir = "/idia/projects/hippo/Koketso/galaxyzoo/resized/galaxy_zoo_class"


#Training arguments
architecure = "Resnet18"
patience = 5
l_r = 1e-4
best_loss = 5000000


wandb.init(
    # set the wandb project where this run will be logged
    project="BYOL Galaxy Zoo Systematic",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": l_r,
    "architecture": architecture,
    "dataset": "Galaxy Zoo Resized",
    "augmentation (Rotationall360)":r_r,
    "augmentation (VFlip)":v_p,
    "augmentation (HFlip)":h_p,
    "augmentation (gblur)":g_p,
    "augmentation (crop)":r_c,

    "epochs": 300,
    "patience":5,
    "batch size":batch_size
    }
)





#Define the model

model = tv.models.resnet18(weights = "IMAGENET1K_V1")

model.fc = torch.nn.Linear(512,100)
model.fc.weight.data.normal_(0,0.01)

model.train()


#Define a random rotation function that excludes the edges of the rotated images
# The augmentations

class RandomRotationWithCrop(K.RandomRotation):
    def __init__(self, degrees, crop_size, output_dim = 244,p = 0.5):
        super(RandomRotationWithCrop, self).__init__(degrees, p = 1)
        #super(RandomRotationWithCrop, self).__init__(crop_size)
        self.rotation_transform = K.RandomRotation(degrees)
        self.crop_transform = K.CenterCrop(crop_size, keepdim = False,align_corners = True)
        self.resize_transform = K.Resize(output_dim)
    def __call__(self, x):
        if random.random() <self.p:
            # Apply random rotation
            x = self.rotation_transform(x)

            # Apply center crop
            x = self.crop_transform(x)
            x = self.resize_transform(x)

        return x
    
    
# Define a feature extractor for classification based validation

    
def features(loader,model):
    time1 = time.time()
    rep = []
    labells = []
    names = []
    images = []
    name = "_"
    label = 0
    i = 0
    with torch.no_grad():
        for image,label,name in loader:                                   #name
            if i*batch_size > 100000:
                break;
            #images.append(image)
            image = image.to(device)
            rep.append(model(image))                 #Modify to model used here
            labells.append(label)

            names.append(name)                      #name
            i+=1

            clear_output(wait = True)
            display(i*batch_size)
            
    #Unwrappping the data
    rep2 = []
    labells2 = []
    rep2 = []
    images2 = []
    names2 = []



    for i in range(len(rep)):
        for j in range(len(rep[i])):
            #images2.append(images[i][j].cpu().numpy()) #Images
            rep2.append(rep[i][j].cpu().numpy())        #Representations
            labells2.append(labells[i][j].item())
            names2.append(names[i][j])                  #Error here if no names

    rep = rep2
    #images = images2 
    labels = labells2

    names = names2
    return rep,labels


#The datasets setup

#training_validation

dataset = Custom.dataset(galaxyzoo_dir)
names = [name[0].split('/')[-1] for name in dataset.imgs]

#classification validation

classification_val_dataset = Custom.dataset(galaxyzooq_dir)

datasets = Custom.train_val_dataset(dataset, val_split = 0.1)

#Traning

transformed_train_dataset = Custom.Custom(datasets['train'],
                                    names = names,
                                    resize = resize,
                                   crop = 244,
                                   )


loader = DataLoader(transformed_train_dataset, 
                    batch_size, 
                    shuffle = True,
                    num_workers = 15)

#validation

transformed_val_dataset = Custom.Custom(datasets['val'],
                                    names = names,
                                    resize = resize,
                                   crop = 244,
                                   )

val_loader = DataLoader(transformed_val_dataset, 
                    batch_size, 
                    shuffle = True,
                    num_workers = 15)


#Classification validation

transformed_classification_val_dataset = Custom.Custom_labelled(classification_val_dataset,
                                    names = names,
                                    resize = resize,
                                   crop = 244,
                                   )



class_loader = DataLoader(transformed_classification_val_dataset, 
                    batch_size, 
                    shuffle = True,
                    num_workers = 15)






#define the augmentations to be used



augment_fn = torch.nn.Sequential(
    
   

        RandomRotationWithCrop(degrees = [0,360],crop_size =200,p =r_r),
    
        kornia.augmentation.RandomResizedCrop([244,244],scale =(0.3,1), p = r_c),
        K.RandomGaussianBlur(kernel_size = [3,3],sigma = [1,2], p =g_p)
    
)



#Define the learner

learner = BYOL(
    model,
    image_size = 244,
    hidden_layer = 'avgpool',
    augment_fn = augment_fn
   
)

# Send to gpu

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
learner = learner.to(device)



# Self_supervised training
for epoch in range(300):
    
    loss_ = 0.0
    learner.train()
    for i,Images in enumerate(loader):
        Images = Images[0]
        #send imaged to device
        images = Images.to(device)
        #optain loss
        loss = learner(images)
        
        #optimization steps
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() #update moving average of target encoder
        loss_ += loss.item()
        loss_per_500 = loss_
        #display(progress)
        if i%5 ==0:
            print("Batch epoch :"+ str(epoch) + " Loss :" + str(loss.item()))
        try:
            if (i+1)%500 ==0:
                wandb.log({"Training loss_per_500": loss_per_500/500})
                loss_per_500 = 0
                
   
        except:
            print("didnt work")
    
    
    #Implementing the early stopping
    
    # Validate
    learner.eval()
    clear_output(wait = True)
    print("Validating")
    with torch.no_grad():
        val_loss = 0
        for i, val_images in enumerate(val_loader):
            
            val_images = val_images[0]
            val_images = val_images.to(device)
            v_loss = learner(val_images)
            val_loss += v_loss.item() 
        wandb.log({"Validation epoch loss": val_loss})
        wandb.log({"Training epoch loss": loss_})
            

        print("Validation loss: ",val_loss)

    if val_loss < best_loss:
        
        best_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
            'augmentations':augment_fn,
            }, "./Features/models_/res18byol_004.pt")
        
    
        counter = 0
    else:
        counter += 1
        
    #Classification validatation
    a,b = features(class_loader,model)
    val_ac = test.KNN_accuracy(a,b)[0].item()
    wandb.log({"Classification Validation":float(val_ac)})
    

    

    # Check if early stopping criteria are met
    if counter >= patience:
        print("Early stopping: No improvement in validation loss for {} epochs".format(patience))
        wandb.finish()
        break
        
