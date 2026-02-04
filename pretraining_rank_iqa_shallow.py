""" This code reproduces pretraining with vgg16 as in Rank-IQA paper, with pairwise ranking loss. """
from comet_ml import Experiment, OfflineExperiment
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from pretraining import DegradedPairDataset, set_window_gpu
import numpy as np 
import os 
from tqdm.auto import tqdm
import collections
import torchvision
from torchvision.models import vgg16, VGG16_Weights

class ShallowFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=256):
        super(ShallowFeatureExtractor, self).__init__()
        
        # 4 convolutional layers
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # -> (32, 224, 224)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # -> (32, 224, 224)  # one channel 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> (64, 112, 112)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # -> (128, 56, 56)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)# -> (256, 28, 28)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layer to produce final feature vector
        # After 4 poolings: 224 -> 112 -> 56 -> 28 -> 14
        self.fc = nn.Linear(256 * 14 * 14, feature_dim)
        # 64 --> 1 
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (32, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # -> (64, 56, 56)
        x = self.pool(F.relu(self.conv3(x)))  # -> (128, 28, 28)
        x = self.pool(F.relu(self.conv4(x)))  # -> (256, 14, 14)
        
        x = x.view(x.size(0), -1)             # Flatten
        features = self.fc(x)                 # Feature vector
        return features

class RankIQA_branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = ShallowFeatureExtractor()
        
        self.head = nn.Sequential(
            nn.Linear(256, 64),  
            nn.ReLU(),            
            nn.Linear(64, 1)    
        )

    def forward(self, x):
        feats = self.features(x)
        score = self.head(feats)
        return score

class SiameseRankIQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = RankIQA_branch()

    def forward(self, x, x_hat):
        s = self.scorer(x)
        s_hat = self.scorer(x_hat)
        return s, s_hat

def ranking_loss(s, s_hat, margin=1.0):  # y ...
    # Want s(x) > s(x_hat), so s - s_hat >= margin
    # diff = s - s_hat
    # loss = torch.clamp(margin - y*diff, min=0)

    diff = s_hat - s
    # loss = torch.clamp(((2*y - 1) * diff) + margin, min=0)
    loss = torch.clamp(diff + margin, min=0)
    return loss.mean()
# loss = torch.clamp(margin - y * (s - s_hat), min=0).mean()

def ranking_accuracy(s, s_hat):
    # s, s_hat: [B, 1]
    return (s > s_hat).float().mean()

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--n_epochs', dest="n_epochs",type=int, default=5, help='number of epochs for training')
    parser.add_argument('--file_path', dest="file_path", type=str, help='file name for saving pretrained model', default=None)

    # comet parameters
    parser.add_argument("--comet", dest="comet", default=1, help="1 for comet ON, 0 for comet OFF")
    parser.add_argument("--name_proj", dest="name_proj", default='medrank-iqa-pretraining', help="define comet ml project folder")
    parser.add_argument("--name_exp", dest="name_exp", default='soft_tissue_window', help="name of comet ml experiment")

    parser.add_argument("--batch_size", dest="batch_size", default=16, help="batch size for train and test")

    parser.add_argument('--device_id', dest="device_id",  default='0', help='gpu device id.')
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=1e-4, help="base learning rate")

    parser.add_argument('--imagenet-initialization', dest="imagenet-initialization",  default=None, help='use (1) or not (None) imagenet weights for VGG16')

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

    base_pretrained_folder = './pretrained_models'
    os.makedirs(base_pretrained_folder, exist_ok=True)

    save_file_name = args.file_path
    if save_file_name is None:
        save_file_name = args.name_exp

    print("Save file name: ", save_file_name)

    # COMET
    experiment = None
    if int(args.comet) == 0:
        # Comet ml integration
        experiment = OfflineExperiment(offline_directory=base_pretrained_folder+ '/COMET_OFFLINE',
                                       project_name=args.name_proj)
    else:
        # matplotlib.use('TkAgg')
        experiment = Experiment(project_name=args.name_proj)

    experiment.set_name(args.name_exp)
    ek = experiment.get_key()

    # Datasets e Dataloaders
    train_dataset = DegradedPairDataset(mode='train')
    test_dataset  = DegradedPairDataset(mode='test')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=4)
    test_dataloader  = DataLoader(dataset=test_dataset,  batch_size=int(args.batch_size), num_workers=4)

    model = SiameseRankIQA()
    model.to(device)
    
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    optim = Adam(model.parameters(), lr=float(args.learning_rate))
    # criterion = nn.CrossEntropyLoss() 

    for epoch in tqdm(range(args.n_epochs)): # il numero di epoche lo scelgo quando mando il file sul terminale
        model.train()
        train_losses = []
        with tqdm(train_dataloader, leave=False, desc="Training") as t:
            # for x, x_hat, y in t:
            for x, x_hat in t:
                x = x.to(device)
                x_hat = x_hat.to(device)
                # y = y.to(device)

                #if x.size(1) == 1:
                #    x = x.repeat(1, 3, 1, 1)
                #    x_hat = x_hat.repeat(1, 3, 1, 1)

                ########
                x = set_window_gpu(x)  # C, H, W
                x_hat = set_window_gpu(x_hat)
                ########

                optim.zero_grad()
                # y_pred = model(x, x_hat)
                # y_pred = model(x, x_hat)
                # loss = criterion(y_pred, y.long())

                s, s_hat = model(x, x_hat)
                # loss = ranking_loss(s, s_hat, y)
                loss = ranking_loss(s, s_hat)
                
                t.set_postfix(loss=loss.item())
                loss.backward()
                optim.step()

                train_losses.append(loss.item())
        
        train_loss_mean = np.mean(train_losses)
        experiment.log_metric("train_loss", train_loss_mean, step=epoch)

        model.eval()
        test_accuracies = []
        test_losses = []
        with tqdm(test_dataloader, leave=False, desc="Test") as t:
            # for x, x_hat, y in t:
            for x, x_hat in t:
                x = x.to(device)
                x_hat = x_hat.to(device)
                # y = y.to(device)

                #if x.size(1) == 1:
                #    x = x.repeat(1, 3, 1, 1)
                #    x_hat = x_hat.repeat(1, 3, 1, 1)

                ########
                x = set_window_gpu(x)  # C, H, W
                x_hat = set_window_gpu(x_hat)
                ########

                with torch.no_grad():
                    # y_pred = model(x, x_hat)
                    # y_pred = model(x, x_hat)
                    s, s_hat = model(x, x_hat)
    
                # loss = criterion(y_pred, y.long())
                # loss = ranking_loss(s, s_hat, y)
                loss = ranking_loss(s, s_hat)
                t.set_postfix(loss=loss.item())
                test_losses.append(loss.item())
                # prediction = torch.argmax(y_pred, dim=1)
                # accuracy = (prediction == y).float().mean()
                accuracy = ranking_accuracy(s, s_hat)
                test_accuracies.append(accuracy.item())

        test_loss_mean = np.mean(test_losses)
        test_acc_mean  = np.mean(test_accuracies)     
        print(f"Epoch {epoch+1} - Test Accuracy: {test_acc_mean:.4f}, Test Loss: {test_loss_mean:.4f}")

        experiment.log_metric("test_loss", test_loss_mean, step=epoch)
        experiment.log_metric("test_accuracy", test_acc_mean, step=epoch)  


    torch.save(model.state_dict(), os.path.join(base_pretrained_folder, save_file_name + '.pth')) # per salvare il modello
    # log model ? 
    experiment.end()
