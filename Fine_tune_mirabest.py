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



if __name__ == '__main__': 

    wandb.login(key = "98b1d9a3de157fb207740b99168d17aea96d90ee")
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 22}


    #The code parameters:
    #augmentation probabilitiees
    g_p = 0.5
    v_p = 0.5
    h_p = 0.5
    r_r = 0.7
    r_c = 0.7
    valsplit = 0.1
    l_r = 1e-4

    initial_weights = True
    continuation = True
    model_name = "Resnet_Mirabest_ImageNet"

    epochs = 20
    num_workers = 10

    #Dataset arguments
    batch_size = 32

    resize = 300

    model = tv.models.resnet18(weights = "IMAGENET1K_V1")


    model.fc = torch.nn.Linear(512,100)
    model.fc.weight.data.normal_(0,0.01)

    model.train()



    def features(loader,model):
        time1 = time.time()
        rep = []
        labells = []
        images = []
        label = 0
        i = 0
        with torch.no_grad():
            for image,label in loader:                                   #name
                if i*batch_size > 10000:
                    break;
                #images.append(image)
                image = image.to(device)
                rep.append(model(image, return_embedding = True)[1])                 #Modify to model used here
                
                labells.append(label)

                i+=1

                clear_output(wait = True)
                display(i*batch_size)
                
        #Unwrappping the data
        rep2 = []
        labells2 = []
        rep2 = []
        images2 = []



        for i in range(len(rep)):
            for j in range(len(rep[i])):
                #images2.append(images[i][j].cpu().numpy()) #Images
                rep2.append(rep[i][j].cpu().numpy())        #Representations
                labells2.append(labells[i][j].item())

        rep = rep2
        #images = images2 
        labels = labells2

        return rep,labels




    augment_fn = torch.nn.Sequential(
        
       

            Custom.RandomRotationWithCrop(degrees = [0,360],crop_size =200,p =r_r),
            kornia.augmentation.RandomVerticalFlip( p = v_p),
            kornia.augmentation.RandomHorizontalFlip( p = h_p),
        
            kornia.augmentation.RandomResizedCrop([244,244],scale =(0.7,1), p = r_c),
            K.RandomGaussianBlur(kernel_size = [3,3],sigma = [1,2], p =g_p)
        
    )



    learner = BYOL(
        model,
        image_size = 244,
        hidden_layer = 'avgpool',
        augment_fn = augment_fn
       
    )


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    learner = learner.to(device)





    #Mirabest specific transformations
    resize = 300
    crop = 224
    transform = tv.transforms.Compose([
                                tv.transforms.Resize((resize,resize)),
                                tv.transforms.CenterCrop(224),           # So they are compatible with the dnn models
                                tv.transforms.Grayscale(num_output_channels=3),
                                tv.transforms.ToTensor(),
                                tv.transforms.Grayscale(num_output_channels=3),
                                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
    importlib.reload(mb)



    transformed_train_dataset = mb.MBFRFull(root='./batches', train=True, download=True, transform=transform) 
    transformed_val_dataset = mb.MBFRFull(root='./batches', train=False, download=True, transform=transform)
    transformed_classification_val_dataset = mb.MBFRConfident(root='./batches', train=True, download=True, transform=transform) 
    batch_size = 32


    loader = DataLoader(transformed_train_dataset, 
                        batch_size, 
                        shuffle = True,
                        num_workers = num_workers)

    val_loader = DataLoader(transformed_val_dataset, 
                        batch_size, 
                        shuffle = True,
                        num_workers = num_workers)


    class_loader = DataLoader(transformed_classification_val_dataset, 
                        batch_size, 
                        shuffle = True,
                        num_workers = num_workers)




    patience = 10
    best_loss = 5000000

    opt = torch.optim.Adam(learner.parameters(), lr=l_r)    #3e-7 for gz   -4 for hand  -5 for meerkat_curated
    importlib.reload(wandb)
    wandb.init(
        # set the wandb project where this run will be logged
        project="BYOL Mirabest",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": l_r,
        "architecture": model_name,
        "dataset": "Mirabest New val Test",
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





    #
    if continuation:
        
        try:
            epoch = torch.load("Features/models_/"+model_name+".pt",map_location = "cpu")['epoch']

            model.load_state_dict(torch.load("Features/models_/"+model_name+".pt",map_location = "cpu")['model_state_dict'])
            
            opt.load_state_dict(torch.load("Features/models_/"+model_name+".pt",map_location = "cpu")['optimizer_state_dict'])
            
            save_to = "Features/losses/"+model_name+"accuracies"+".plk"        
            with open(save_to,'rb') as file:
                val_accuracies = pickle.load(file)    

        except:
            epoch = 0
            val_accuracies = []





    # Self_supervised trainin
    a,b = features(class_loader,learner)
    val_accuracies = []

    val_accuracies.append([a,b])
    learner.eval()
    val_ac = test.KNN_accuracy(a,b)[0].item()
    wandb.log({"Classification Validation":float(val_ac)})



    # Self_supervised training
    while epoch <= epochs:
        
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
            if i%50 ==0:
                print("Batch epoch :"+ str(epoch) + " Loss :" + str(loss.item()))
        
        
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
            wandb.log({"Validation epoch loss": val_loss/len(val_loader)*batch_size})
            wandb.log({"Training epoch loss": loss_/len(loader)*batch_size})
            

            print("Validation loss: ",val_loss)

        if val_loss < best_loss:
            
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss,
                'augmentations':augment_fn,
                }, "./Features/models_/"+"best_"+model_name+".pt")
            
        
            counter = 0
        else:
            counter += 1
            
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss,
                'augmentations':augment_fn,
                'optimizer_state_dict': opt.state_dict(),
                }, "./Features/models_/"+model_name+".pt")    
            
        #Classification validatation
        a,b = features(class_loader,learner)
        val_accuracies.append([a,b])
        val_ac = test.KNN_accuracy(a,b)[0].item()
        wandb.log({"Classification Validation":float(val_ac)})
        epoch +=1
        

    """    

        # Check if early stopping criteria are met
        if counter >= patience:
            print("Early stopping: No improvement in validation loss for {} epochs".format(patience))
            wandb.finish()
            break
    """        
    save_to = "Features/losses/"+model_name+"accuracies"+".plk"        
    with open(save_to,'wb') as file:
        pickle.dump(val_accuracies,file)        

