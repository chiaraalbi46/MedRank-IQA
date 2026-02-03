""" This code reproduces pretraining with vgg16 as in Rank-IQA paper, with pairwise ranking loss. """
from comet_ml import Experiment, OfflineExperiment
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
import torch
from torch import nn
from pretraining import DegradedPairDataset, set_window_gpu
import numpy as np 
import os 
from tqdm.auto import tqdm
import collections
import torchvision
from torchvision.models import vgg16, VGG16_Weights

class Vgg16(nn.Module):
    def __init__(self, imagenet=None):
        super(Vgg16, self).__init__()

        if imagenet is not None:
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            model = torchvision.models.vgg16(pretrained=False, num_classes=1)  # original code (for loading pretrained model on natural images)

        
        self.features = torch.nn.Sequential(
            collections.OrderedDict(
                zip(
                    [
                        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
                        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'
                    ],
                    model.features
                )
            )
        )

        self.classifier = torch.nn.Sequential(
            collections.OrderedDict(
                zip(
                    ['fc6_m', 'relu6_m', 'drop6_m', 'fc7_m', 'relu7_m', 'drop7_m', 'fc8_m'],
                    model.classifier
                )
            )
        )
        if imagenet is not None:
            self.classifier.fc8_m = nn.Linear(4096, 1)  

    def load_model(self, file, debug: bool = False):
        """
        Load model file.

        :param file: the model file to load.
        :param debug: indicate if output the debug info.
        """
        state_dict = torch.load(file)

        dict_to_load = dict()
        for k, v in state_dict.items():  # "v" is parameter and "k" is its name
            for l, p in self.named_parameters():  # "p" is parameter and "l" is its name
                # use parameter's name to match state_dict's params and model's params
                split_k, split_l = k.split('.'), l.split('.')
                if (split_k[0] in split_l[1]) and (split_k[1] == split_l[2]):
                    dict_to_load[l] = torch.from_numpy(np.array(v)).view_as(p)
                    if debug:  # output debug info
                        print(f"match: {split_k} and {split_l}.")

        self.load_state_dict(dict_to_load)

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, start_dim=1, end_dim=-1)  # dont use adaptive avg pooling
        out = self.classifier(out)
        return out

class VGGFeatureExtractor(nn.Module):
    def __init__(self, vgg):
        super().__init__()
        self.features = vgg.features

    def forward(self, x):
        f = self.features(x)              # [B, 512, H, W]
        return f

class RankIQA_branch(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        self.features = VGGFeatureExtractor(vgg_model)  # conv layers of VGG16
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1)  # scalar score
        )

    def forward(self, x):
        feats = self.features(x)
        score = self.head(feats)
        return score

class SiameseRankIQA(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        self.scorer = RankIQA_branch(vgg_model)

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

    vgg = Vgg16(imagenet=args.imagenet_initialization)
    model = SiameseRankIQA(vgg_model=vgg)
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

                if x.size(1) == 1:
                    x = x.repeat(1, 3, 1, 1)
                    x_hat = x_hat.repeat(1, 3, 1, 1)

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

                if x.size(1) == 1:
                    x = x.repeat(1, 3, 1, 1)
                    x_hat = x_hat.repeat(1, 3, 1, 1)

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
