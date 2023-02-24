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
    N = 200 #no of samples
    C = 20 #no of classes
    F = 20 #feature size
    
    #create list of colors
    color_mat = np.random.uniform(size=(C, 3))
    
    #create list to store features and colors to plot   
    features = np.zeros((N, F))
    colors = np.zeros((N, 3))
        
    #initialize model
    model = ResNet(20)
    params = torch.load('resnet.pt')
    model.load_state_dict(params)
    
    #softmax layer
    softmax = nn.Softmax(dim=1)

    #initialize test dataloader
    dataloader = utils.get_data_loader(
        'voc', train=False, batch_size=1, split='test')
    
    #extract features
    for batch_idx, (data, target, wgt) in tqdm(enumerate(dataloader)):
        #if index exceeds N
        if batch_idx >= N:
            break
        
        #extract features
        fts = softmax(model(data)).cpu().detach().numpy().flatten()

        #extract relevant colors
        labels = (target==1).cpu().detach().numpy().flatten()
        col = color_mat[labels]
        
        #average colors
        col = np.mean(col, axis=0)
        
        #add features and colors to array
        features[batch_idx] = fts
        colors[batch_idx] = col
    
    #use tSNE to compute 2D features
    features_2d = TSNE(n_components=2, random_state=0).fit_transform(features)
    
    #plot the features and colors
    plt.scatter(features_2d[:,0], features_2d[:,1], c=colors)
    plt.savefig('tsne.png')
    
if __name__ == "__main__":
    main()