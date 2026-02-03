""" This code aims to refactoring the pretraining code (CodiceFase1Shallow.py), also introducing a different pre-processing step 
(custom crop instead of resize) and working on random crops instead of on full resized images. """

from comet_ml import Experiment, OfflineExperiment

from argparse import ArgumentParser
import torch
import numpy as np
import random
import os
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from dicaugment import Sharpen, RandomBrightnessContrast, GaussNoise
import torch.nn.functional as F

from monai.transforms import Transform
from random import randrange
import glob

TARGET_HW = 224

def set_window_gpu(x, window_level=40, window_width=400):

    low = window_level - window_width / 2
    high = window_level + window_width / 2

    # Clamp in window
    x = torch.clamp(x, low, high)

    # Normalize to [0, 1]
    x = (x - low) / (high - low)

    return x

### MONAI wrapper
class Windowing(Transform):
    def __init__(self, window_level=40, window_width=400, out_range=(0.0, 1.0)):
        self.wl = window_level
        self.ww = window_width
        self.out_min, self.out_max = out_range

    def __call__(self, x):
       
        x_out = set_window_gpu(x, self.wl, self.ww)

        return x_out

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

class SiameseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ShallowFeatureExtractor()

        self.head = nn.Linear(512, 2)

    def forward(self, x, x_hat):      
        z = self.resnet(x)               
        z_hat = self.resnet(x_hat)       
        z_full = torch.cat([z, z_hat], dim=1)
        y_pred = self.head(z_full)
        return y_pred
    
#### keep sample of the artifact separated from application of it 
def sample_degradation():
    parametri = {
        "GaussNoise": {  # no overlaps
            1: dict(var_limit=(0.001, 0.005)),
            2: dict(var_limit=(0.006, 0.01)),
            3: dict(var_limit=(0.02, 0.03)),
            4: dict(var_limit=(0.04, 0.06)),
        },
        "RandomBC": {
            1: dict(brightness_limit=0.005, contrast_limit=0.1),
            2: dict(brightness_limit=0.2, contrast_limit=0.2),
            3: dict(brightness_limit=0.3, contrast_limit=0.5),
            4: dict(brightness_limit=0.6, contrast_limit=0.6),
        },
        "Sharpen": {
            1: dict(alpha=(0.05, 0.1), lightness=0.7),
            2: dict(alpha=(0.15, 0.2), lightness=0.7),
            3: dict(alpha=(0.25, 0.3), lightness=0.7),
            4: dict(alpha=(0.35, 0.4), lightness=0.7),
        },
    }

    artefatto = random.choice(list(parametri.keys()))
    livello = random.choice(list(parametri[artefatto].keys()))
    p = parametri[artefatto][livello]

    return artefatto, livello, p, parametri

def build_degradation(artefatto, p):

    def _apply(img: np.ndarray) -> np.ndarray:
        # img attesa in formato channel-last (H, W) o (H, W, C)
        # Ritorna immagine trasformata con stessa shape

        # Helper per assicurare canale
        def _ensure_channel_last(img_):
            if img_.ndim == 2:
                return img_[:, :, np.newaxis]
            return img_

        # def _maybe_remove_singleton_channel(img_):
        #     if img_.ndim == 3 and img_.shape[2] == 1:
        #         return img_[:, :, 0]
        #     return img_
        
        # salvo min e max originali 
        img_min = float(np.min(img))
        img_max = float(np.max(img))

        if artefatto == "RandomBC":
            
            if img_max == img_min:
                # immagine piatta: niente trasformazione per evitare divisione per zero
                return img.copy()

            # normalizza tra 0 e 1
            img_norm = (img - img_min) / (img_max - img_min)

            # Aggiungi dimensione canale se serve
            img_norm = _ensure_channel_last(img_norm)

            # Applica RandomBrightnessContrast
            rbc = RandomBrightnessContrast(
                brightness_limit=p["brightness_limit"],
                contrast_limit=p["contrast_limit"],
                p=1.0
            )
            transformed = rbc(image=img_norm)['image']

            # Rimuovi canale se presente
            # transformed = _maybe_remove_singleton_channel(transformed)

            # Riporta alla scala originale
            transformed_rescaled = transformed * (img_max - img_min) + img_min
            return transformed_rescaled.astype(np.float32, copy=False)

        elif artefatto == "GaussNoise":
            # Normalizza immagine tra 0 e 1
            # img_min = float(np.min(img))
            # img_max = float(np.max(img))
            if img_max == img_min:
                return img.copy()

            # normalizza tra 0 e 1
            img_norm = (img - img_min) / (img_max - img_min)

            # Aggiungi dimensione canale se serve
            img_norm = _ensure_channel_last(img_norm)

            # Applica GaussNoise
            gauss_noise = GaussNoise(var_limit=p["var_limit"], p=1.0)
            transformed = gauss_noise(image=img_norm)['image']

            # Rimuovi canale se presente
            # transformed = _maybe_remove_singleton_channel(transformed)

            # Riporta alla scala originale
            transformed_rescaled = transformed * (img_max - img_min) + img_min
            return transformed_rescaled.astype(np.float32, copy=False)

        else:  # Sharpen
            # Applica Sharpen a un'immagine 8-bit (0-255)

            # img8 = img
            # if img8.dtype != np.uint8:
            #     # Porta su 0-255 rispettando il range corrente
            #     i_min = float(np.min(img8))
            #     i_max = float(np.max(img8))
            #     if i_max > i_min:
            #         img8 = (img8 - i_min) / (i_max - i_min)
            #     else:
            #         img8 = np.zeros_like(img8, dtype=np.float32)
            #     img8 = np.clip(img8 * 255.0, 0, 255).astype(np.uint8)

            img_norm = (img - img_min) / (img_max - img_min)  # 0–1
            img_uint8 = np.clip(img_norm * 255.0, 0, 255).astype(np.uint8)

            img_uint8 = _ensure_channel_last(img_uint8)

            sharpen = Sharpen(
                alpha=p["alpha"],
                lightness=p["lightness"],
                always_apply=True,
                p=1.0
            )
            transformed = sharpen(image=img_uint8)['image']

            # transformed = _maybe_remove_singleton_channel(transformed)

            transformed_rescaled = transformed.astype(np.float32) / 255.0
            transformed_rescaled = transformed_rescaled * (img_max - img_min) + img_min

            return transformed_rescaled.astype(np.float32, copy=False)

    return _apply

def sample_paired_degradation(artefatto, level_x1_hat, p1, parametri):
    ### this create x2_hat based on x1_hat for comparison between degraded versions

    livelli = list(parametri[artefatto].keys())

    # remove the already used level
    filtered_levels = [l for l in livelli if l != level_x1_hat]
    level_x2_hat = random.choice(filtered_levels)
    p2 = parametri[artefatto][level_x2_hat]

    quality_label = 0  # 0 means that x2_hat has higher quality than x1_hat
    if level_x2_hat > level_x1_hat:  # sigma 2 > sigma 1 
        quality_label = 1  # means that x2_hat has lower quality with respect to x1_hat
    
    return p2, quality_label
    # if artefatto == "GaussNoise" or "RandomBC":
    #     if level_x2_hat > level_x1_hat:  # sigma 2 > sigma 1 
    #         quality_label = 1  # means that x2_hat has lower quality with respect to x1_hat
    #         # return artefatto, p2, quality_label

    # else:
    #     # Sharpen
    #     # keep same lightness of p1 (x1_hat) but uses new alpha from p2
    #     # p2_ = p2.copy()
    #     # p2_['lightness'] = p1['lightness']
    

#########################
def degradazioni():
    #Restituisce una funzione che accetta un'immagine channel-last (H, W[, C])
    #e ritorna l'immagine degradata

    parametri = {
        "GaussNoise": {
            1: dict(var_limit=(0.001, 0.005)),
            2: dict(var_limit=(0.005, 0.01)),
            3: dict(var_limit=(0.01, 0.03)),
            4: dict(var_limit=(0.02, 0.05)),
        },
        "RandomBC": {
            1: dict(brightness_limit=0.005, contrast_limit=0.1),
            2: dict(brightness_limit=0.2, contrast_limit=0.2),
            3: dict(brightness_limit=0.3, contrast_limit=0.5),
            4: dict(brightness_limit=0.6, contrast_limit=0.6),
        },
        "Sharpen": {
            1: dict(alpha=(0.05, 0.1), lightness=(0.5, 1.0)),
            2: dict(alpha=(0.1, 0.2), lightness=(0.5, 1.0)),
            3: dict(alpha=(0.2, 0.3), lightness=(0.5, 1.0)),
            4: dict(alpha=(0.3, 0.5), lightness=(0.5, 1.0)),
        },
    }

    artefatto = random.choice(list(parametri.keys()))
    livello = random.choice(list(parametri[artefatto].keys()))
    p = parametri[artefatto][livello]

    # print("Artifact: ", artefatto, livello, p)

    def _apply(img: np.ndarray) -> np.ndarray:
        # img attesa in formato channel-last (H, W) o (H, W, C)
        # Ritorna immagine trasformata con stessa shape

        # Helper per assicurare canale
        def _ensure_channel_last(img_):
            if img_.ndim == 2:
                return img_[:, :, np.newaxis]
            return img_

        def _maybe_remove_singleton_channel(img_):
            if img_.ndim == 3 and img_.shape[2] == 1:
                return img_[:, :, 0]
            return img_
        
        # salvo min e max originali 
        img_min = float(np.min(img))
        img_max = float(np.max(img))

        if artefatto == "RandomBC":
            
            if img_max == img_min:
                # immagine piatta: niente trasformazione per evitare divisione per zero
                return img.copy()

            # normalizza tra 0 e 1
            img_norm = (img - img_min) / (img_max - img_min)

            # Aggiungi dimensione canale se serve
            img_norm = _ensure_channel_last(img_norm)

            # Applica RandomBrightnessContrast
            rbc = RandomBrightnessContrast(
                brightness_limit=p["brightness_limit"],
                contrast_limit=p["contrast_limit"],
                p=1.0
            )
            transformed = rbc(image=img_norm)['image']

            # Rimuovi canale se presente
            transformed = _maybe_remove_singleton_channel(transformed)

            # Riporta alla scala originale
            transformed_rescaled = transformed * (img_max - img_min) + img_min
            return transformed_rescaled

        elif artefatto == "GaussNoise":
            # Normalizza immagine tra 0 e 1
            # img_min = float(np.min(img))
            # img_max = float(np.max(img))
            if img_max == img_min:
                return img.copy()

            # normalizza tra 0 e 1
            img_norm = (img - img_min) / (img_max - img_min)

            # Aggiungi dimensione canale se serve
            img_norm = _ensure_channel_last(img_norm)

            # Applica GaussNoise
            gauss_noise = GaussNoise(var_limit=p["var_limit"], p=1.0)
            transformed = gauss_noise(image=img_norm)['image']

            # Rimuovi canale se presente
            transformed = _maybe_remove_singleton_channel(transformed)

            # Riporta alla scala originale
            transformed_rescaled = transformed * (img_max - img_min) + img_min
            return transformed_rescaled

        else:  # Sharpen
            # Applica Sharpen a un'immagine 8-bit (0-255)

            img_norm = (img - img_min) / (img_max - img_min)  # 0–1
            img_uint8 = np.clip(img_norm * 255.0, 0, 255).astype(np.uint8)

            img_uint8 = _ensure_channel_last(img_uint8)

            sharpen = Sharpen(
                alpha=p["alpha"],
                lightness=p["lightness"],
                always_apply=True,
                p=1.0
            )
            transformed = sharpen(image=img_uint8)['image']

            transformed = _maybe_remove_singleton_channel(transformed)

            transformed_rescaled = transformed.astype(np.float32) / 255.0
            transformed_rescaled = transformed_rescaled * (img_max - img_min) + img_min

            return transformed_rescaled

    return _apply

def random_patch_selection(h, w, crop_size=TARGET_HW):

    top = randrange(0, max(1, h - crop_size))
    left = randrange(0, max(1, w - crop_size))

    return top, left

class DegradedPairDataset(Dataset):
    # def __init__(self, root_path="/Prove/Albisani/TCIA_datasets/train", mode='train'):
    def __init__(self, mode='train'):

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

        # self.postprocess = Windowing(window_level=40, window_width=400)  # soft-tissue window
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        x = np.load(img_path).astype(np.float32)  # H, W  np
        
        # x = torch.from_numpy(x).unsqueeze(0).float()      # 1, H, W
        # x_hat = np.load(self.ld_files[idx])
        # x_hat = torch.from_numpy(x_hat).unsqueeze(0).float()

        r = random.random()
        quality_label = None
        crop_mode = 'same'
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

        # y = random.randint(0, 1)
        # if y == 0:
        #     return x_crop, x_hat_crop, y
        # else:
        #     return x_hat_crop, x_crop, y

        return x_crop, x_hat_crop

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--n_epochs', dest="n_epochs",type=int, default=5, help='number of epochs for training')
    parser.add_argument('--file_path', dest="file_path", type=str, help='file name for saving pretrained model', default=None)

    # comet parameters
    parser.add_argument("--comet", dest="comet", default=1, help="1 for comet ON, 0 for comet OFF")
    parser.add_argument("--name_proj", dest="name_proj", default='medrank-iqa-pretraining', help="define comet ml project folder")
    parser.add_argument("--name_exp", dest="name_exp", default='soft_tissue_window', help="name of comet ml experiment")

    parser.add_argument("--batch_size", dest="batch_size", default=16, help="batch size for train and test")
    # TODO: aggiungi patience as argument
    # log hyperparametri (lr, patience, lr backbone, )

    parser.add_argument('--device_id', dest="device_id",  default='1', help='gpu device id.')

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

    model = SiameseModel()
    
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    optim = Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss() 

    for epoch in tqdm(range(args.n_epochs)): # il numero di epoche lo scelgo quando mando il file sul terminale
        model.train()
        train_losses = []
        with tqdm(train_dataloader, leave=False, desc="Training") as t:
            for x, x_hat, y in t:
                x = x.to(device)
                x_hat = x_hat.to(device)
                y = y.to(device)

                ########
                x = set_window_gpu(x)  # C, H, W
                x_hat = set_window_gpu(x_hat)
                ########

                optim.zero_grad()
                y_pred = model(x, x_hat)
                # y_pred = model(x, x_hat)
                loss = criterion(y_pred, y.long())
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
                x = x.to(device)
                x_hat = x_hat.to(device)
                y = y.to(device)

                ########
                x = set_window_gpu(x)  # C, H, W
                x_hat = set_window_gpu(x_hat)
                ########

                with torch.no_grad():
                    # y_pred = model(x, x_hat)
                    y_pred = model(x, x_hat)

                loss = criterion(y_pred, y.long())
                t.set_postfix(loss=loss.item())
                test_losses.append(loss.item())
                prediction = torch.argmax(y_pred, dim=1)
                accuracy = (prediction == y).float().mean()
                test_accuracies.append(accuracy.item())

        test_loss_mean = np.mean(test_losses)
        test_acc_mean  = np.mean(test_accuracies)     
        print(f"Epoch {epoch+1} - Test Accuracy: {test_acc_mean:.4f}, Test Loss: {test_loss_mean:.4f}")

        experiment.log_metric("test_loss", test_loss_mean, step=epoch)
        experiment.log_metric("test_accuracy", test_acc_mean, step=epoch)  


    torch.save(model.state_dict(), os.path.join(base_pretrained_folder, save_file_name + '.pth')) # per salvare il modello
    # log model ? 
    experiment.end()
