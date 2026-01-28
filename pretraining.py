""" This code aims to refactoring the pretraining code (CodiceFase1Shallow.py), also introducing a different pre-processing step 
(custom crop instead of resize) and working on random crops instead of on full resized images. """
from comet_ml import Experiment, OfflineExperiment

from argparse import ArgumentParser
import torch
import numpy as np
import random
import os
import json
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from monai.transforms import LoadImage, ScaleIntensity, Compose, ToTensor, Lambda, EnsureChannelFirst
#from monai.networks.nets import resnet18
from torchvision.models import resnet18
from torchvision.transforms import Resize
from monai.data.image_reader import PydicomReader
from tqdm.auto import tqdm
from dicaugment import Sharpen, RandomBrightnessContrast, GaussNoise
import torch.nn.functional as F

from skimage.morphology import closing, disk
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from monai.transforms import Transform
from random import randrange


# DEVICE = "cuda:0"
TARGET_HW = 224

# resize dopo load ? non altero l'artefatto ...
# random crops devono essere stessa posizione (se uso artefatti 'realistici') perchè gli artefatti di ct non sono uniformi 
# random crops anche diversa posizione se uso gli artefatti che proviamo adesso...

# def extract_body_crop_otsu(x, min_size=224):
#     # Otsu separates body from background even if not true air
#     # x = x.detach().cpu().numpy()

#     t = threshold_otsu(x)
#     mask = x > t

#     # Clean
#     mask = closing(mask, disk(7))

#     # Largest connected component
#     labeled = label(mask)
#     regions = regionprops(labeled)
#     largest = max(regions, key=lambda r: r.area)

#     minr, minc, maxr, maxc = largest.bbox

#     # vorrei che il crop fosse almeno min_size x min_size
#     h = maxr - minr
#     w = maxc - minc

#     if h < min_size:
#         pad = min_size - h
#         minr -= pad // 2
#         maxr += pad - pad // 2

#     if w < min_size:
#         pad = min_size - w
#         minc -= pad // 2
#         maxc += pad - pad // 2
    
#     H, W = x.shape
#     minr = max(0, minr)
#     minc = max(0, minc)
#     maxr = min(H, maxr)
#     maxc = min(W, maxc)

#     cropped = x[minr:maxr, minc:maxc]

#     return cropped, (minr, minc, maxr, maxc), mask

# ### MONAI wrapper
# class BodyCropOtsu(Transform):
#     def __init__(self, closing_radius=7, margin=10, min_size=224):
#         self.closing_radius = closing_radius
#         self.margin = margin
#         self.min_size = min_size

#     def __call__(self, x):
#         # x is torch.Tensor (H, W) or (1, H, W)
#         if x.ndim == 3:
#             x = x[0]  # remove channel for processing

#         x_np = x.detach().cpu().numpy()

#         # call function
#         x_out, _, _ = extract_body_crop_otsu(x_np, self.min_size)

#         # # back to tensor
#         # x_out = torch.from_numpy(x_out).unsqueeze(0).to(x.device)

#         ### read the otsu crops json and apply the crop corresponding to the img path

#         return x_out

# def set_window(x, window_level=40, window_width=400):
  
#     low = window_level - window_width // 2
#     high = window_level + window_width // 2
#     x_win = np.clip(x, low, high)

#     # normalize
#     x_norm = (x_win - low) / (high - low)  # 0, 1

#     return x_norm

def set_window_gpu(x, window_level=40, window_width=400):

    low = window_level - window_width / 2
    high = window_level + window_width / 2

    # Clamp in window
    x = torch.clamp(x, low, high)

    # Normalize to [0, 1]
    x = (x - low) / (high - low)

    # # Optional: map to arbitrary range (e.g. [-1,1])
    # if (self.out_min, self.out_max) != (0.0, 1.0):
    #     x = x * (self.out_max - self.out_min) + self.out_min

    return x

### MONAI wrapper
class Windowing(Transform):
    def __init__(self, window_level=40, window_width=400, out_range=(0.0, 1.0)):
        self.wl = window_level
        self.ww = window_width
        self.out_min, self.out_max = out_range

    def __call__(self, x):
        # x: torch.Tensor in HU, shape (1, H, W)
        
        # if x.ndim == 3:
        #     x = x[0]  # remove channel for processing

        # x_np = x.detach().cpu().numpy()

        # x è giù numpy array

        # call function
        # x_out = set_window(x, self.wl, self.ww)
        x_out = set_window_gpu(x, self.wl, self.ww)

        # # back to tensor
        # x_out = torch.from_numpy(x_out).unsqueeze(0).to(x.device)

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
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (32, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # -> (64, 56, 56)
        x = self.pool(F.relu(self.conv3(x)))  # -> (128, 28, 28)
        x = self.pool(F.relu(self.conv4(x)))  # -> (256, 14, 14)
        
        x = x.view(x.size(0), -1)             # Flatten
        features = self.fc(x)                 # Feature vector
        return features


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

            transformed = _maybe_remove_singleton_channel(transformed)

            transformed_rescaled = transformed.astype(np.float32) / 255.0
            transformed_rescaled = transformed_rescaled * (img_max - img_min) + img_min

            return transformed_rescaled

    return _apply

############# maybe these functions could be avoided ...
def ensure_chlast(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x[..., np.newaxis] 
    if x.ndim == 3:
        if x.shape[0] <= 4 and x.shape[1] > 32 and x.shape[2] > 32:
            return np.moveaxis(x, 0, -1) 
        return x  
    raise ValueError(f"Forma immagine non attesa: {x.shape}")

def chlast_to_cd_hw(x_chlast: np.ndarray) -> np.ndarray:
    s_h_w = np.moveaxis(x_chlast, -1, 0)  
    #return s_h_w[np.newaxis, ...]         
    return s_h_w

####################

def random_patch_selection(h, w, crop_size=TARGET_HW):

    top = randrange(0, max(1, h - crop_size))
    left = randrange(0, max(1, w - crop_size))

    # cropped = x[:, :, top:top+crop_size, left:left+crop_size]
    # ritorno gli indici per fare lo stesso crop su x e x_hat

    return top, left

class DegradedPairDataset(Dataset):
    def __init__(self, root_path="/Prove/Albisani/TCIA_datasets/train", mode='train'):
        self.image_files = []
        for patient_folder in os.listdir(root_path):
            patient_path = os.path.join(root_path, patient_folder)
            if os.path.isdir(patient_path):
                image_dir = os.path.join(patient_path, "Full_Dose_Images")
                if os.path.isdir(image_dir):
                    for root, _, files in os.walk(image_dir):
                        for f in files:
                            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tif", ".dcm")):
                                self.image_files.append(os.path.join(root, f))
        
        # filepath to recover otsu crops 
        self.otsu_crop_file = f'/Prove/Albisani/TCIA_datasets/otsu_crops_{mode}.json'
        with open(self.otsu_crop_file) as f:
            self.otsu_crops = json.load(f)

        self.preprocess = Compose([
            LoadImage(image_only=True),  
            # crop con otsu 
            # BodyCropOtsu()
        ])

        self.postprocess = Compose([
            ToTensor(dtype=torch.float32),  # 1, H, W
            Windowing(window_level=40, window_width=400),  # soft-tissue window
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        ### load
        x = self.preprocess(img_path)

        ## retrieve corresponding otsu crop
        minr, minc, maxr, maxc = self.otsu_crops[img_path]
        x = x[minr:maxr, minc:maxc]

        ## passo a numpy per inserire l'artefatto
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)

        x = x.astype(np.float32, copy=False)
        x_chlast = ensure_chlast(x)   # H, W, 1

        #### artifact simulation
        aug = degradazioni()

        x_hat_chlast = aug(x_chlast)
        x_hat_chlast = ensure_chlast(x_hat_chlast)

        x_cd_hw = chlast_to_cd_hw(x_chlast)  # 1, H, W
        # x_cd_hw = np.repeat(x_cd_hw, 3, axis=0)  # non voglio lavorare a tre canali 
        xhat_cd_hw = chlast_to_cd_hw(x_hat_chlast)
        # xhat_cd_hw = np.repeat(xhat_cd_hw, 3, axis=0)

        ### windowing and/or normalization
        x = self.postprocess(x_cd_hw)  # C, H, W
        x_hat = self.postprocess(xhat_cd_hw)

        ### random patch
        h, w = x.shape[1], x.shape[2]
        # 50% same crop, 50% different
        if random.random() < 0.5:
            # same crop
            top, left = random_patch_selection(h, w, crop_size=TARGET_HW)

            x_crop = x[:, top:top+TARGET_HW, left:left+TARGET_HW]
            x_hat_crop = x_hat[:, top:top+TARGET_HW, left:left+TARGET_HW]
        else:
            # different crops
            top1, left1 = random_patch_selection(h, w, crop_size=TARGET_HW)
            top2, left2 = random_patch_selection(h, w, crop_size=TARGET_HW)

            x_crop = x[:, top1:top1+TARGET_HW, left1:left1+TARGET_HW]
            x_hat_crop = x_hat[:, top2:top2+TARGET_HW, left2:left2+TARGET_HW]

        # h, w = x.shape[1], x.shape[2]
        # top, left = random_patch_selection(h, w, crop_size=TARGET_HW)

        # x_crop = x[:, top:top+TARGET_HW, left:left+TARGET_HW]
        # x_hat_crop = x_hat[:, top:top+TARGET_HW, left:left+TARGET_HW]

        y = random.randint(0, 1)
        if y == 0:
            return x_crop, x_hat_crop, y
        else:
            return x_hat_crop, x_crop, y

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

    train_siamese_path = "/Prove/Albisani/TCIA_datasets/train"
    test_siamese_path  = "/Prove/Albisani/TCIA_datasets/test"

    # Datasets e Dataloaders
    train_dataset = DegradedPairDataset(train_siamese_path, mode='train')
    test_dataset  = DegradedPairDataset(test_siamese_path, mode='test')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=4)
    test_dataloader  = DataLoader(dataset=test_dataset,  batch_size=int(args.batch_size), num_workers=4)

    model = SiameseModel()
    model = model.to(device)
    
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

                optim.zero_grad()
                y_pred = model(x, x_hat)
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
                with torch.no_grad():
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
