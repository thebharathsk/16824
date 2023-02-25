import torch
from utils import ARGS
import numpy as np
import torchvision
import torch.nn as nn
import utils
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from simple_cnn import get_fc

#define the model
class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1', progress=True)
        
        # TODO define a FC layer here to process the features
        #MY IMPLEMENTATION
        self.resnet.fc = nn.Sequential(*get_fc(512, num_classes, 'none'))
    
    def forward(self, x):
        # TODO return unnormalized log-probabilities here
        #MY IMPLEMENTATION        
        out = self.resnet(x)
        #out = self.fc1(x)
        
        return out

#main function
def main():
    #create some variables
    N = 1000 #no of samples
    C = 20 #no of classes
    F = 512 #feature size
    
    #create list of colors
    np.random.seed(27011996)
    color_mat = np.random.uniform(size=(C, 3))
    
    c = [0, 1, 0.5]
    idx = 0 
    for i in c:
        for j in c:
            for k in c:
                if idx >= 20:
                    break
                else:
                    color_mat[idx, :] = [i, j, k]
                    idx+=1
    
    CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    
    #create list to store features and colors to plot   
    features = np.zeros((N, F))
    colors = np.zeros((N, 3))
        
    #initialize model
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1', progress=True)
    
    # Load the pre-trained resnet18 model
    params = torch.load('resnet_old.pt')
        
    #edit the params
    params['resnet.fc.weight'] = torch.randn([1000,512]) #params['resnet.fc.0.weight']
    params['resnet.fc.bias'] = torch.randn([1000]) #params['resnet.fc.0.bias']
    params.pop('resnet.fc.0.weight')
    params.pop('resnet.fc.0.bias')
    
    new_params = {}
    for k in params.keys():
        tag = k.split('resnet.')[-1]
        new_params[tag] = params[k]
    
    model.load_state_dict(new_params, strict=True)

    # Get all layers of the model except for the last layer
    model = nn.Sequential(*list(model.children())[:-1])
    
    #initialize test dataloader
    dataloader = utils.get_data_loader(
        'voc', train=False, batch_size=1, split='test')
    
    #extract features
    for batch_idx, (data, target, wgt) in tqdm(enumerate(dataloader)):
        #if index exceeds N
        if batch_idx >= N:
            break
        
        #extract features
        fts = model(data).cpu().detach().numpy().flatten()

        #extract relevant colors
        labels = (target==1).cpu().detach().numpy().flatten()

        col = color_mat[labels,:]
        
        #average colors
        col = np.mean(col, axis=0)
        
        #add features and colors to array
        features[batch_idx] = fts
        colors[batch_idx] = col
    
    
    features = (features - features.min())/(features.max() - features.min())
    
    #use tSNE to compute 2D features
    features_2d = TSNE(n_components=2, random_state=0).fit_transform(features)
    
    #plot the features and colors
    plt.xlim(right=-0.5, left=1.25)
    plt.ylim(top=1.25, bottom=-0.25)
    features_2d[:,0] = (features_2d[:,0] - features_2d[:,0].min())/(features_2d[:,0].max() - features_2d[:,0].min())
    features_2d[:,1] = (features_2d[:,1] - features_2d[:,1].min())/(features_2d[:,1].max() - features_2d[:,1].min())
    
    plt.scatter(features_2d[:,0], features_2d[:,1], c=colors)
    
    
    #add legend
    # create an empty dictionary for the legend handles and labels
    legend_dict = {}

    # iterate through the colors dictionary and create a new scatter plot for each group
    for i in range(C):
        handle = plt.scatter([], [], color=colors[i], label=CLASS_NAMES[i])
        legend_dict[CLASS_NAMES[i]] = handle,

    # add the legend
    plt.legend(handles=[h[0] for h in legend_dict.values()], fontsize="x-small")
    plt.title('t-SNE plot on test data')
    
    plt.savefig('tsne.png')
    
if __name__ == "__main__":
    main()