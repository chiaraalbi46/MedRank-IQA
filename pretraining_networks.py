""" Definition of network models to test for pretraining """
# vgg16
# resnet18
# shufflenet
# squeezenet
# mobilenetv2


import collections
import torchvision
from torchvision.models import vgg16, VGG16_Weights
import torch
from torch import nn
import numpy as np 

### VGG16

class Vgg16(nn.Module):
    def __init__(self, imagenet=None):
        super(Vgg16, self).__init__()

        if imagenet is not None:
            print("QUIIIII")
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

            # cambia primo layer per input 1 canale
            # old_conv = model.features[0]
            # model.features[0] = nn.Conv2d(
            #     1,
            #     old_conv.out_channels,
            #     kernel_size=old_conv.kernel_size,
            #     stride=old_conv.stride,
            #     padding=old_conv.padding
            # )

            # # inizializza pesi facendo la media dei canali RGB
            # with torch.no_grad():
            #     model.features[0].weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        else:
            model = torchvision.models.vgg16(pretrained=False, num_classes=1)  # original code (for loading pretrained model on natural images)

            # model.features[0] = nn.Conv2d(
            #     1, 64, kernel_size=3, stride=1, padding=1
            # )

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

### RESNET18

class Resnet18(nn.Module):
    def __init__(self, imagenet=None):
        super(Resnet18, self).__init__()

        if imagenet is not None:
            print("quii")
            model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        else:
            model = torchvision.models.resnet18(weights=None)

        # feature extractor
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            # model.avgpool
        )

        # classifier
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class Resnet18FeatureExtractor(nn.Module):
    def __init__(self, resnet18):
        super().__init__()
        self.features = resnet18.features

    def forward(self, x):
        f = self.features(x)              # [B, 512, H, W]
        return f

class Resnet18RankIQA_branch(nn.Module):
    def __init__(self, resnet18_model):
        super().__init__()
        self.features = Resnet18FeatureExtractor(resnet18_model)  # conv layers of VGG16
        
        self.head = nn.Sequential(
            nn.Flatten(),  # [1, 512]
            #  nn.Linear(512, 1)  #[1, 1]
            nn.Linear(512*7*7, 1)  # [1, 1]
        )

        # self.head = nn.Sequential(
        #     nn.Conv2d(512, 1, kernel_size=1),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten()
        # )

    def forward(self, x):
        feats = self.features(x)
        # print(feats.shape)
        score = self.head(feats)
        return score

class Resnet18SiameseRankIQA(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        self.scorer = Resnet18RankIQA_branch(vgg_model)

    def forward(self, x, x_hat):
        s = self.scorer(x)
        s_hat = self.scorer(x_hat)
        return s, s_hat


### SQUEEZE NET 1.1

class SqueezeNet1_1(nn.Module):
    def __init__(self, imagenet=None):
        super(SqueezeNet1_1, self).__init__()

        if imagenet is not None:
            print("quii")
            model = torchvision.models.squeezenet1_1(weights='IMAGENET1K_V1')  # SqueezeNet1_1_Weights.IMAGENET1K_V1
        else:
            model = torchvision.models.squeezenet1_1(weights=None)

        # feature extractor
        self.features = model.features

        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # classifier
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = self.features(x)
        # x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class SqueezeNet1_1FeatureExtractor(nn.Module):
    def __init__(self, squeezenet1_1):
        super().__init__()
        self.features = squeezenet1_1.features

    def forward(self, x):
        f = self.features(x)              # [B, 512, H, W]
        return f

class SqueezeNet1_1RankIQA_branch(nn.Module):
    def __init__(self, squeezenet1_1_model):
        super().__init__()
        self.features = SqueezeNet1_1FeatureExtractor(squeezenet1_1_model)  
        
        self.head = nn.Sequential(
            nn.Flatten(),  # [1, 512]
            #  nn.Linear(512, 1)  #[1, 1]
            nn.Linear(512*13*13, 1)  # [1, 1]  # 13,13 per immagini 224x224
        )

        # self.head = nn.Sequential(
        #     nn.Conv2d(512, 1, kernel_size=1),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten()  # → [B, 1]
        # )

    def forward(self, x):
        feats = self.features(x)
        # print(feats.shape)
        score = self.head(feats)
        return score

class SqueezeNet1_1SiameseRankIQA(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        self.scorer = SqueezeNet1_1RankIQA_branch(vgg_model)

    def forward(self, x, x_hat):
        s = self.scorer(x)
        s_hat = self.scorer(x_hat)
        return s, s_hat

if __name__ == '__main__':

    model = torchvision.models.squeezenet1_1(weights=None)

    pass