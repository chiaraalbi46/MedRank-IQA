""" This code reproduces pretraining with vgg16 as in Rank-IQA paper, with pairwise ranking loss. """
from comet_ml import Experiment, OfflineExperiment
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
import torch
from torch import nn
from pretraining import rescaling, random, build_degradation, sample_degradation, sample_paired_degradation, TARGET_HW, random_patch_selection
import numpy as np 
import os 
from tqdm.auto import tqdm
import collections
import torchvision
from torchvision.models import vgg16, VGG16_Weights
import glob
from dataset_pretraining_class import DegradedPairDataset_1

from pretraining_networks import Vgg16, SiameseRankIQA, Resnet18, Resnet18SiameseRankIQA, SqueezeNet1_1, SqueezeNet1_1SiameseRankIQA

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Makes CuDNN deterministic (important!)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

base_seed = 42
set_global_seed(base_seed)

g_train = torch.Generator()
g_train.manual_seed(base_seed)

g_test = torch.Generator()
g_test.manual_seed(base_seed + 1)


class DegradedPairDataset(Dataset):
    # def __init__(self, root_path="/Prove/Albisani/TCIA_datasets/train", mode='train'):
    def __init__(self, mode='train', test_case='all'):

        root_path=f"/Prove/Albisani/TCIA_datasets/{mode}_cropped_npy"

        fd_files = glob.glob(f"{root_path}/*_fd.npy")

        pairs = []
        for fd in fd_files:
            ld = fd.replace("_fd.npy", "_ld.npy")
            if os.path.exists(ld):
                pairs.append((fd, ld))
            else:
                print(f"⚠️ Missing LD for {fd}")

        pairs.sort()  # stable order

        self.image_files = [p[0] for p in pairs]
        self.low_dose_images = [p[1] for p in pairs]

        self.test_case = test_case

        # self.postprocess = Windowing(window_level=40, window_width=400)  # soft-tissue window
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        x = np.load(img_path).astype(np.float32)  # H, W  np
        
        # x = torch.from_numpy(x).unsqueeze(0).float()      # 1, H, W
        # x_hat = np.load(self.ld_files[idx])
        # x_hat = torch.from_numpy(x_hat).unsqueeze(0).float()

        quality_label = None
        crop_mode = 'same'

        if self.test_case == 'all':
            # print("Pretraining with synthetic, low dose and same artifact, different levels combinations, crop mode possibly different")
            
            r = random.random()

            if r <= 0.25:
                x = torch.from_numpy(x).unsqueeze(0).float()      # 1, H, W torch

                #### x vs x_hat (low dose)
                low_dose_img_path = self.low_dose_images[idx]

                x_hat = np.load(low_dose_img_path).astype(np.float32)
                x_hat = torch.from_numpy(x_hat).unsqueeze(0).float()  # 1, H, W
            
            elif 0.25 < r <= 0.6:

                #### x vs x_hat
                crop_mode = "same" if random.random() < 0.8 else "different"

                art, level_x_hat, p, parametri = sample_degradation()
                x_hat = build_degradation(artefatto=art, p=p)(x)  # H, W, 1

                x_hat = torch.from_numpy(x_hat).permute(2,0,1).float().contiguous() # 1, H, W torch
                x = torch.from_numpy(x).unsqueeze(0).float()  

            else:
            
                #### x1_hat vs x2_hat
                art1, level_x1_hat, p1, parametri = sample_degradation()
                x1_hat = build_degradation(artefatto=art1, p=p1)(x)
                x1_hat = torch.from_numpy(x1_hat).permute(2,0,1).float().contiguous() # 1, H, W torch

                p2, quality_label = sample_paired_degradation(artefatto=art1, level_x1_hat=level_x1_hat, 
                                                            p1=p1, parametri=parametri)
                x2_hat = build_degradation(artefatto=art1, p=p2)(x)  # H, W, 1
                x2_hat = torch.from_numpy(x2_hat).permute(2,0,1).float().contiguous() # 1, H, W torch

                # x = torch.from_numpy(x).unsqueeze(0).float() 
        
        elif self.test_case == 'all_same':
            # print("Pretraining with synthetic, low dose and same artifact, different levels combinations, crop mode == same ")

            r = random.random()

            if r <= 0.25:
                x = torch.from_numpy(x).unsqueeze(0).float()      # 1, H, W torch

                #### x vs x_hat (low dose)
                low_dose_img_path = self.low_dose_images[idx]

                x_hat = np.load(low_dose_img_path).astype(np.float32)
                x_hat = torch.from_numpy(x_hat).unsqueeze(0).float()  # 1, H, W
            
            elif 0.25 < r <= 0.6:

                #### x vs x_hat
                # random.seed(27)
                # crop_mode = "same" if random.random() < 0.8 else "different"

                art, level_x_hat, p, parametri = sample_degradation()
                x_hat = build_degradation(artefatto=art, p=p)(x)  # H, W, 1

                x_hat = torch.from_numpy(x_hat).permute(2,0,1).float().contiguous() # 1, H, W torch
                x = torch.from_numpy(x).unsqueeze(0).float()  

            else:
            
                #### x1_hat vs x2_hat
                art1, level_x1_hat, p1, parametri = sample_degradation()
                x1_hat = build_degradation(artefatto=art1, p=p1)(x)
                x1_hat = torch.from_numpy(x1_hat).permute(2,0,1).float().contiguous() # 1, H, W torch

                p2, quality_label = sample_paired_degradation(artefatto=art1, level_x1_hat=level_x1_hat, 
                                                            p1=p1, parametri=parametri)
                x2_hat = build_degradation(artefatto=art1, p=p2)(x)  # H, W, 1
                x2_hat = torch.from_numpy(x2_hat).permute(2,0,1).float().contiguous() # 1, H, W torch

                # x = torch.from_numpy(x).unsqueeze(0).float() 
        
        elif self.test_case == 'all_diff':
            crop_mode = 'diff'
            r = random.random()

            if r <= 0.25:
                x = torch.from_numpy(x).unsqueeze(0).float()      # 1, H, W torch

                #### x vs x_hat (low dose)
                low_dose_img_path = self.low_dose_images[idx]

                x_hat = np.load(low_dose_img_path).astype(np.float32)
                x_hat = torch.from_numpy(x_hat).unsqueeze(0).float()  # 1, H, W
            
            elif 0.25 < r <= 0.6:

                #### x vs x_hat
                # random.seed(27)
                # crop_mode = "same" if random.random() < 0.8 else "different"

                art, level_x_hat, p, parametri = sample_degradation()
                x_hat = build_degradation(artefatto=art, p=p)(x)  # H, W, 1

                x_hat = torch.from_numpy(x_hat).permute(2,0,1).float().contiguous() # 1, H, W torch
                x = torch.from_numpy(x).unsqueeze(0).float()  

            else:
            
                #### x1_hat vs x2_hat
                art1, level_x1_hat, p1, parametri = sample_degradation()
                x1_hat = build_degradation(artefatto=art1, p=p1)(x)
                x1_hat = torch.from_numpy(x1_hat).permute(2,0,1).float().contiguous() # 1, H, W torch

                p2, quality_label = sample_paired_degradation(artefatto=art1, level_x1_hat=level_x1_hat, 
                                                            p1=p1, parametri=parametri)
                x2_hat = build_degradation(artefatto=art1, p=p2)(x)  # H, W, 1
                x2_hat = torch.from_numpy(x2_hat).permute(2,0,1).float().contiguous() # 1, H, W torch

                # x = torch.from_numpy(x).unsqueeze(0).float() 

        elif self.test_case == 'synthetic':
            # print("Pretraining only with synthetic perturbations.")  # crop mode same 

            art, level_x_hat, p, parametri = sample_degradation()
            x_hat = build_degradation(artefatto=art, p=p)(x)  # H, W, 1

            x_hat = torch.from_numpy(x_hat).permute(2,0,1).float().contiguous() # 1, H, W torch
            x = torch.from_numpy(x).unsqueeze(0).float()  
        
        elif self.test_case == 'low_dose':
            # print("Pretraining only with low dose perturbations.")  # crop mode same 

            x = torch.from_numpy(x).unsqueeze(0).float()      # 1, H, W torch

            #### x vs x_hat (low dose)
            low_dose_img_path = self.low_dose_images[idx]

            x_hat = np.load(low_dose_img_path).astype(np.float32)
            x_hat = torch.from_numpy(x_hat).unsqueeze(0).float()  # 1, H, W
        
        elif self.test_case == 'same_artifact_diff_levels':
            # print("Pretraining only with same artifact with different levels perturbations.")  # crop mode same 

            #### x1_hat vs x2_hat
            art1, level_x1_hat, p1, parametri = sample_degradation()
            x1_hat = build_degradation(artefatto=art1, p=p1)(x)
            x1_hat = torch.from_numpy(x1_hat).permute(2,0,1).float().contiguous() # 1, H, W torch

            p2, quality_label = sample_paired_degradation(artefatto=art1, level_x1_hat=level_x1_hat, 
                                                        p1=p1, parametri=parametri)
            x2_hat = build_degradation(artefatto=art1, p=p2)(x)  # H, W, 1
            x2_hat = torch.from_numpy(x2_hat).permute(2,0,1).float().contiguous() # 1, H, W torch

        # if r <= 0.25:
        #     x = torch.from_numpy(x).unsqueeze(0).float()      # 1, H, W torch

        #     #### x vs x_hat (low dose)
        #     low_dose_img_path = self.low_dose_images[idx]

        #     x_hat = np.load(low_dose_img_path).astype(np.float32)
        #     x_hat = torch.from_numpy(x_hat).unsqueeze(0).float()  # 1, H, W
        
        # elif 0.25 < r <= 0.6:

        #     #### x vs x_hat
        #     random.seed(27)
        #     crop_mode = "same" if random.random() < 0.8 else "different"

        #     art, level_x_hat, p, parametri = sample_degradation()
        #     x_hat = build_degradation(artefatto=art, p=p)(x)  # H, W, 1

        #     x_hat = torch.from_numpy(x_hat).permute(2,0,1).float().contiguous() # 1, H, W torch
        #     x = torch.from_numpy(x).unsqueeze(0).float()  

        # else:
        
        #     #### x1_hat vs x2_hat
        #     art1, level_x1_hat, p1, parametri = sample_degradation()
        #     x1_hat = build_degradation(artefatto=art1, p=p1)(x)
        #     x1_hat = torch.from_numpy(x1_hat).permute(2,0,1).float().contiguous() # 1, H, W torch

        #     p2, quality_label = sample_paired_degradation(artefatto=art1, level_x1_hat=level_x1_hat, 
        #                                                   p1=p1, parametri=parametri)
        #     x2_hat = build_degradation(artefatto=art1, p=p2)(x)  # H, W, 1
        #     x2_hat = torch.from_numpy(x2_hat).permute(2,0,1).float().contiguous() # 1, H, W torch

        #     # x = torch.from_numpy(x).unsqueeze(0).float() 

        if quality_label is not None:
            if quality_label == 0:
                # x2_hat > x1_hat 
                x = x2_hat
                x_hat = x1_hat
            else:
                # x2_hat < x1_hat 
                x = x1_hat
                x_hat = x2_hat

        # ### windowing and/or normalization
        # x = self.postprocess(x)  # C, H, W
        # x_hat = self.postprocess(x_hat)

        ### random patch
        h, w = x.shape[1], x.shape[2]
        
        if crop_mode == 'same':
            # same crop
            top, left = random_patch_selection(h, w, crop_size=TARGET_HW)

            x_crop = x[:, top:top+TARGET_HW, left:left+TARGET_HW].contiguous()
            x_hat_crop = x_hat[:, top:top+TARGET_HW, left:left+TARGET_HW].contiguous()
        else:
            # different crops
            top1, left1 = random_patch_selection(h, w, crop_size=TARGET_HW)
            top2, left2 = random_patch_selection(h, w, crop_size=TARGET_HW)

            x_crop = x[:, top1:top1+TARGET_HW, left1:left1+TARGET_HW].contiguous()
            x_hat_crop = x_hat[:, top2:top2+TARGET_HW, left2:left2+TARGET_HW].contiguous()

        y = random.randint(0, 1)
        if y == 0:
            return x_crop, x_hat_crop, y
        else:
            return x_hat_crop, x_crop, y

        # return x_crop, x_hat_crop

# class Resnet18(nn.Module):
#     def __init__(self, imagenet=None):
#         super(Resnet18, self).__init__()

#         if imagenet is not None:
#             print("quii")
#             model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
#         else:
#             model = torchvision.models.resnet18(weights=None)

#         # feature extractor
#         self.features = nn.Sequential(
#             model.conv1,
#             model.bn1,
#             model.relu,
#             model.maxpool,
#             model.layer1,
#             model.layer2,
#             model.layer3,
#             model.layer4,
#             # model.avgpool
#         )

#         # classifier
#         self.classifier = nn.Linear(512, 1)

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
    
# class Resnet18FeatureExtractor(nn.Module):
#     def __init__(self, resnet18):
#         super().__init__()
#         self.features = resnet18.features

#     def forward(self, x):
#         f = self.features(x)              # [B, 512, H, W]
#         return f

# class Resnet18RankIQA_branch(nn.Module):
#     def __init__(self, resnet18_model):
#         super().__init__()
#         self.features = Resnet18FeatureExtractor(resnet18_model)  # conv layers of VGG16
        
#         self.head = nn.Sequential(
#             nn.Flatten(),  # [1, 512]
#             #  nn.Linear(512, 1)  #[1, 1]
#             nn.Linear(512*7*7, 1)  # [1, 1]
#         )

#         # self.head = nn.Sequential(
#         #     nn.Conv2d(512, 1, kernel_size=1),
#         #     nn.AdaptiveAvgPool2d(1),
#         #     nn.Flatten()
#         # )

#     def forward(self, x):
#         feats = self.features(x)
#         # print(feats.shape)
#         score = self.head(feats)
#         return score

# class Resnet18SiameseRankIQA(nn.Module):
#     def __init__(self, vgg_model):
#         super().__init__()
#         self.scorer = Resnet18RankIQA_branch(vgg_model)

#     def forward(self, x, x_hat):
#         s = self.scorer(x)
#         s_hat = self.scorer(x_hat)
#         return s, s_hat

# class Vgg16(nn.Module):
#     def __init__(self, imagenet=None):
#         super(Vgg16, self).__init__()

#         if imagenet is not None:
#             print("QUIIIII")
#             model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

#             # cambia primo layer per input 1 canale
#             # old_conv = model.features[0]
#             # model.features[0] = nn.Conv2d(
#             #     1,
#             #     old_conv.out_channels,
#             #     kernel_size=old_conv.kernel_size,
#             #     stride=old_conv.stride,
#             #     padding=old_conv.padding
#             # )

#             # # inizializza pesi facendo la media dei canali RGB
#             # with torch.no_grad():
#             #     model.features[0].weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
#         else:
#             model = torchvision.models.vgg16(pretrained=False, num_classes=1)  # original code (for loading pretrained model on natural images)

#             # model.features[0] = nn.Conv2d(
#             #     1, 64, kernel_size=3, stride=1, padding=1
#             # )

#         self.features = torch.nn.Sequential(
#             collections.OrderedDict(
#                 zip(
#                     [
#                         'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
#                         'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
#                         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
#                         'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
#                         'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'
#                     ],
#                     model.features
#                 )
#             )
#         )

#         self.classifier = torch.nn.Sequential(
#             collections.OrderedDict(
#                 zip(
#                     ['fc6_m', 'relu6_m', 'drop6_m', 'fc7_m', 'relu7_m', 'drop7_m', 'fc8_m'],
#                     model.classifier
#                 )
#             )
#         )
#         if imagenet is not None:
#             self.classifier.fc8_m = nn.Linear(4096, 1)  

#     def load_model(self, file, debug: bool = False):
#         """
#         Load model file.

#         :param file: the model file to load.
#         :param debug: indicate if output the debug info.
#         """
#         state_dict = torch.load(file)

#         dict_to_load = dict()
#         for k, v in state_dict.items():  # "v" is parameter and "k" is its name
#             for l, p in self.named_parameters():  # "p" is parameter and "l" is its name
#                 # use parameter's name to match state_dict's params and model's params
#                 split_k, split_l = k.split('.'), l.split('.')
#                 if (split_k[0] in split_l[1]) and (split_k[1] == split_l[2]):
#                     dict_to_load[l] = torch.from_numpy(np.array(v)).view_as(p)
#                     if debug:  # output debug info
#                         print(f"match: {split_k} and {split_l}.")

#         self.load_state_dict(dict_to_load)

#     def forward(self, x):
#         out = self.features(x)
#         out = torch.flatten(out, start_dim=1, end_dim=-1)  # dont use adaptive avg pooling
#         out = self.classifier(out)
#         return out

# class VGGFeatureExtractor(nn.Module):
#     def __init__(self, vgg):
#         super().__init__()
#         self.features = vgg.features

#     def forward(self, x):
#         f = self.features(x)              # [B, 512, H, W]
#         return f

# class RankIQA_branch(nn.Module):
#     def __init__(self, vgg_model):
#         super().__init__()
#         self.features = VGGFeatureExtractor(vgg_model)  # conv layers of VGG16
        
#         self.head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(4096, 1)  # scalar score
#         )

#     def forward(self, x):
#         feats = self.features(x)
#         score = self.head(feats)
#         return score

# class SiameseRankIQA(nn.Module):
#     def __init__(self, vgg_model):
#         super().__init__()
#         self.scorer = RankIQA_branch(vgg_model)

#     def forward(self, x, x_hat):
#         s = self.scorer(x)
#         s_hat = self.scorer(x_hat)
#         return s, s_hat

def ranking_loss(s, s_hat, y, margin=0.5):  # y ...
    # y = 0 --> s > s_hat

    # Want s(x) > s(x_hat), so s - s_hat >= margin
    diff = s - s_hat
    loss = torch.clamp(((2*y - 1) * diff) + margin, min=0)

    # diff = s_hat - s
    # # loss = torch.clamp(((2*y - 1) * diff) + margin, min=0)
    # loss = torch.clamp(diff + margin, min=0)
    return loss.mean()


# loss = torch.clamp(margin - y * (s - s_hat), min=0).mean()

def ranking_accuracy(s, s_hat, y):
    # s, s_hat: [B, 1]
    # return (s > s_hat).float().mean()
    return ((2 * y - 1) * (s_hat - s) > 0).float().mean()

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--n_epochs', dest="n_epochs",type=int, default=30, help='number of epochs for training')
    parser.add_argument('--file_path', dest="file_path", type=str, help='file name for saving pretrained model', default=None)

    # comet parameters
    parser.add_argument("--comet", dest="comet", default=1, help="1 for comet ON, 0 for comet OFF")
    parser.add_argument("--name_proj", dest="name_proj", default='medrank-iqa-pretraining-NEW', help="define comet ml project folder")
    parser.add_argument("--name_exp", dest="name_exp", default='soft_tissue_window', help="name of comet ml experiment")

    parser.add_argument("--batch_size", dest="batch_size", default=128, help="batch size for train and test")

    parser.add_argument('--device_id', dest="device_id",  default='0', help='gpu device id.')
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=1e-4, help="base learning rate")

    parser.add_argument("--windowing", dest="windowing", type=float, default=1,
                    help="1 is soft tissue windowing, None for norm min max")
    parser.add_argument("--fix_windowing", dest="fix_windowing", type=float, default=None,
                    help="1 is for fix soft tissue windowing for all pairs, None for random windowing among pairs")
    
    parser.add_argument('--imagenet_initialization', dest="imagenet_initialization",  default=None, help='use (1) or not (None) imagenet weights for VGG16')
    parser.add_argument('--test_case', dest="test_case",  default='all', 
                        help='test case for pair combination. all, all_same, synthetic, low_dose, same_artifact_diff_levels.')
    parser.add_argument('--margin', dest="margin",  default=0.5, help='margin for ranking loss')

    
    parser.add_argument('--network_model', dest="network_model",  default='vgg16', help='specify the network to use. vgg16, resnet18')

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

    base_pretrained_folder = './pretrained_models_NEW'
    os.makedirs(base_pretrained_folder, exist_ok=True)

    save_file_name = args.file_path
    if save_file_name is None:
        save_file_name = args.name_exp

    print("Save file name: ", save_file_name)

    rescaling_fn, rescaling_type = rescaling(args.windowing, args.fix_windowing)

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

    ### log useful parameters for the experiment 
    experiment.log_parameters({
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.batch_size),
        "n_epochs": int(args.n_epochs),
        "test_case": args.test_case,
        "windowing": args.windowing,
        "fix_windowing": args.fix_windowing,
        "margin": float(args.margin),
        "imagenet_initialization": args.imagenet_initialization
    })

    print("Test case: ", args.test_case)

    # Datasets e Dataloaders
    # train_dataset = DegradedPairDataset(mode='train', test_case=args.test_case)
    # test_dataset  = DegradedPairDataset(mode='test', test_case=args.test_case)

    train_dataset = DegradedPairDataset_1(mode='train', test_case=args.test_case)
    test_dataset  = DegradedPairDataset_1(mode='test', test_case=args.test_case)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=4,
                                  worker_init_fn=seed_worker,generator=g_train)
    test_dataloader  = DataLoader(dataset=test_dataset,  batch_size=int(args.batch_size), num_workers=4,
                                  worker_init_fn=seed_worker,generator=g_test)

    if args.network_model == 'vgg16':
        vgg = Vgg16(imagenet=args.imagenet_initialization)
        model = SiameseRankIQA(vgg_model=vgg)
    elif args.network_model == 'resnet18':
        vgg = Resnet18(imagenet=args.imagenet_initialization)
        model = Resnet18SiameseRankIQA(vgg_model=vgg)
    elif args.network_model == 'squeezenet1_1':
        vgg = SqueezeNet1_1(imagenet=args.imagenet_initialization)
        model = SqueezeNet1_1SiameseRankIQA(vgg_model=vgg)
        
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
            for x, x_hat, y in t:
            # for x, x_hat in t:
                x = x.to(device)
                x_hat = x_hat.to(device)
                y = y.unsqueeze(1).to(device)

                if x.size(1) == 1:
                    x = x.repeat(1, 3, 1, 1)
                    x_hat = x_hat.repeat(1, 3, 1, 1)

                # ########
                # x = set_window_gpu(x)  # C, H, W
                # x_hat = set_window_gpu(x_hat)
                # ########

                x = rescaling_fn(x)
                x_hat = rescaling_fn(x_hat)

                optim.zero_grad()
                # y_pred = model(x, x_hat)
                # y_pred = model(x, x_hat)
                # loss = criterion(y_pred, y.long())

                s, s_hat = model(x, x_hat)
                loss = ranking_loss(s, s_hat, y)
                # loss = ranking_loss(s, s_hat)
                
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
            for x, x_hat, y in t:
            # for x, x_hat in t:
                x = x.to(device)
                x_hat = x_hat.to(device)
                y = y.unsqueeze(1).to(device)

                if x.size(1) == 1:
                    x = x.repeat(1, 3, 1, 1)
                    x_hat = x_hat.repeat(1, 3, 1, 1)

                # ########
                # x = set_window_gpu(x)  # C, H, W
                # x_hat = set_window_gpu(x_hat)
                # ########

                x = rescaling_fn(x)
                x_hat = rescaling_fn(x_hat)

                with torch.no_grad():
                    # y_pred = model(x, x_hat)
                    # y_pred = model(x, x_hat)
                    s, s_hat = model(x, x_hat)
    
                # loss = criterion(y_pred, y.long())
                loss = ranking_loss(s, s_hat, y)
                # loss = ranking_loss(s, s_hat)
                t.set_postfix(loss=loss.item())
                test_losses.append(loss.item())
                # prediction = torch.argmax(y_pred, dim=1)
                # accuracy = (prediction == y).float().mean()
                accuracy = ranking_accuracy(s, s_hat, y)
                test_accuracies.append(accuracy.item())

        test_loss_mean = np.mean(test_losses)
        test_acc_mean  = np.mean(test_accuracies)     
        print(f"Epoch {epoch+1} - Test Accuracy: {test_acc_mean:.4f}, Test Loss: {test_loss_mean:.4f}")

        experiment.log_metric("test_loss", test_loss_mean, step=epoch)
        experiment.log_metric("test_accuracy", test_acc_mean, step=epoch)  


    torch.save(model.state_dict(), os.path.join(base_pretrained_folder, save_file_name + '.pth')) # per salvare il modello
    # log model ? 
    experiment.end()
