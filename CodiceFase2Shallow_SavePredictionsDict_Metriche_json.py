# Fase 2 - Fine Tuning

from argparse import ArgumentParser
from torch.optim.lr_scheduler import StepLR
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
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau, pearsonr

DEVICE = "cuda:0"
TARGET_D = 8
TARGET_HW = 224

class ShallowFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=256):
        super(ShallowFeatureExtractor, self).__init__()
        
        # 4 convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # -> (32, 224, 224)
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

        if artefatto == "RandomBC":
            # Normalizza immagine tra 0 e 1
            img_min = float(np.min(img))
            img_max = float(np.max(img))
            if img_max == img_min:
                # immagine piatta: niente trasformazione per evitare divisione per zero
                return img.copy()

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
            img_min = float(np.min(img))
            img_max = float(np.max(img))
            if img_max == img_min:
                return img.copy()

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
            img8 = img
            if img8.dtype != np.uint8:
                # Porta su 0-255 rispettando il range corrente
                i_min = float(np.min(img8))
                i_max = float(np.max(img8))
                if i_max > i_min:
                    img8 = (img8 - i_min) / (i_max - i_min)
                else:
                    img8 = np.zeros_like(img8, dtype=np.float32)
                img8 = np.clip(img8 * 255.0, 0, 255).astype(np.uint8)

            img8 = _ensure_channel_last(img8)

            sharpen = Sharpen(
                alpha=p["alpha"],
                lightness=p["lightness"],
                always_apply=True,
                p=1.0
            )
            transformed = sharpen(image=img8)['image']

            transformed = _maybe_remove_singleton_channel(transformed)
            return transformed.astype(np.uint8)

    return _apply

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
    return s_h_w#[np.newaxis, ...]         

def fix_depth_cd_hw(x_cd_hw: np.ndarray, target_d: int) -> np.ndarray:
    c, s, h, w = x_cd_hw.shape
    if s == target_d:
        return x_cd_hw
    if s < target_d:
        reps = int(np.ceil(target_d / s))
        x_rep = np.repeat(x_cd_hw, reps, axis=1) 
        return x_rep[:, :target_d, :, :]
    start = (s - target_d) // 2
    return x_cd_hw[:, start:start + target_d, :, :]

class DegradedPairDataset(Dataset):
    def __init__(self, root_path="/Prove/Albisani/TCIA_datasets/train"):
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

        self.preprocess = Compose([
            LoadImage(image_only=True),  
        ])

        self.postprocess = Compose([
            ScaleIntensity(minv=0.0, maxv=1.0), #(img -min)/(max -min)
            Resize((TARGET_HW, TARGET_HW)),
            ToTensor(dtype=torch.float32),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        x = self.preprocess(img_path)
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        x = x.astype(np.float32, copy=False)

        x_chlast = ensure_chlast(x)  

        aug = degradazioni()
        x_hat_chlast = aug(x_chlast)

        x_hat_chlast = ensure_chlast(x_hat_chlast)

        x_cd_hw = chlast_to_cd_hw(x_chlast)
        xhat_cd_hw = chlast_to_cd_hw(x_hat_chlast)

        x_cd_hw = fix_depth_cd_hw(x_cd_hw, TARGET_D)
        xhat_cd_hw = fix_depth_cd_hw(xhat_cd_hw, TARGET_D)

        x = self.postprocess(x_cd_hw)
        x_hat = self.postprocess(xhat_cd_hw)

        y = random.randint(0, 1)
        if y == 0:
            return x, x_hat, y
        else:
            return x_hat, x, y

def build_path_score_dict(json_path, images_root, is_test=False):
    with open(json_path, "r") as f:
        data = json.load(f)
    path_score_dict = {}
    if not is_test:
        for img_name, score in data.items():
            img_path = os.path.join(images_root, img_name + ".tif")
            if os.path.exists(img_path):
                path_score_dict[img_path] = float(score)
            else:
                print(f"Warning: immagine {img_name} non trovata in {images_root}")
    else:
        for test_key, test_values in data.items():
            fnames = test_values["fnames"]
            scores = test_values["scores"]
            subfolder = os.path.splitext(test_key)[0]  
            for fname, score in zip(fnames, scores):
                img_path = os.path.join(images_root, subfolder, fname + ".tif")
                if os.path.exists(img_path):
                    path_score_dict[img_path] = float(score)
                else:
                    print(f"Warning: immagine {fname}.tif non trovata in {img_path}")
    return path_score_dict

class FineTuningDictDataset(Dataset):
    def __init__(self, path_score_dict, n=None, return_path=False):
        self.items = list(path_score_dict.items())
        if n is not None:
            self.items = random.sample(self.items, n)

        self.return_path = return_path

        self.loader_tif = LoadImage(image_only=True, reader="PILReader")
        self.loader_dcm = LoadImage(image_only=True, reader=PydicomReader)

        def _load_to_cd_hw(p: str) -> np.ndarray:
            plower = p.lower()
            if plower.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")):
                x = self.loader_tif(p)
            elif plower.endswith(".dcm"):
                x = self.loader_dcm(p)
            else:
                x = LoadImage(image_only=True)(p)

            if not isinstance(x, np.ndarray):
                x = np.asarray(x)
            x = x.astype(np.float32, copy=False)

            x_chlast = ensure_chlast(x)
            x_cd_hw  = chlast_to_cd_hw(x_chlast)
            x_cd_hw  = np.repeat(x_cd_hw, 3, axis=0)
            return x_cd_hw

        self.transform = Compose([
            Lambda(func=_load_to_cd_hw),
            ScaleIntensity(minv=0.0, maxv=1.0),
            Resize((TARGET_HW, TARGET_HW)),
            ToTensor(dtype=torch.float32),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, score = self.items[idx]
        image = self.transform(img_path)
        label = torch.tensor(score, dtype=torch.float32)

        if self.return_path:
            return image, label, img_path
        return image, label

class SiameseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ShallowFeatureExtractor()
        #self.resnet.fc = nn.Linear(512, 16)   
        self.head = nn.Linear(512, 2)   

    def forward(self, x, x_hat):      
        z = self.resnet(x)               
        z_hat = self.resnet(x_hat)       
        z_full = torch.cat([z, z_hat], dim=1)
        y_pred = self.head(z_full)
        return y_pred

class Step2Model(nn.Module):
    def __init__(self, siamese_model):
        super().__init__()
        self.resnet = siamese_model.resnet 
        self.head = nn.Linear(256, 1)

    def forward(self, x):               
        z = self.resnet(x)
        y_pred = torch.sigmoid(self.head(z)) * 4 
        return y_pred

# Percorsi 
train_siamese_path = "/Prove/Albisani/TCIA_datasets/train"
test_siamese_path  = "/Prove/Albisani/TCIA_datasets/test"

train_images_root = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAG2023_train/image"
train_json        = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAG2023_train/train.json"

test_images_root  = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAC2023_test"
test_json         = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAC2023_test/test-ground-truth.json"

# Dizionari path->score
train_dict = build_path_score_dict(train_json, train_images_root, is_test=False)
test_dict  = build_path_score_dict(test_json, test_images_root, is_test=True)

# Argpars
parser = ArgumentParser()
parser.add_argument('n_finetuning', type=int, nargs='?', default=None) #serve per usare un minor numero di immagini del dataset di finetuning
parser.add_argument('file_path', nargs='?', default=None) #se nella riga di comando sul terminale non ci metto niente, lui non 
#carica niente cioè non carica il modello con i pesi salvati, se ci scrivo il nome del file con i pesi salvati invece li carica 
args = parser.parse_args()


# Datasets e Dataloaders
train_dataset = DegradedPairDataset(train_siamese_path)
test_dataset  = DegradedPairDataset(test_siamese_path)

train_dataset_finetuning = FineTuningDictDataset(train_dict, n=args.n_finetuning, return_path=False)
test_dataset_finetuning  = FineTuningDictDataset(test_dict, return_path=True)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
test_dataloader  = DataLoader(dataset=test_dataset,  batch_size=8)

train_dataloader_finetuning = DataLoader(dataset=train_dataset_finetuning, batch_size=8, shuffle=True)
test_dataloader_finetuning  = DataLoader(dataset=test_dataset_finetuning,  batch_size=8)

# Caricamento o meno dei pesi preaddestrati della fase 1
model = SiameseModel()
if args.file_path is not None:
  print("Caricamento pesi")
  model.load_state_dict(torch.load(args.file_path, weights_only=True))

# Step 2 fine-tuning 
model = Step2Model(model)
model = model.to(DEVICE)
if args.file_path is not None:
    for p in model.resnet.parameters():
        p.requires_grad = False

if args.file_path is not None:
    optim = Adam(model.head.parameters(), lr=1e-5, weight_decay=1e-4)
else:
    optim = Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
criterion = nn.MSELoss()

scheduler = StepLR(optim, step_size=50, gamma=0.5)

writer = SummaryWriter()
iteration = 0

for epoch in tqdm(range(100)):
    model.train()
    with tqdm(train_dataloader_finetuning, leave=False, desc="Training") as t:
        for x, y in t:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optim.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y.unsqueeze(1))
            t.set_postfix(loss=loss.item())
            loss.backward()
            optim.step()
            writer.add_scalar("train/loss", loss.item(), iteration)
            iteration += 1

    if True or (epoch == 0) or ((epoch + 1) % 10 == 0):
        model.eval()
        test_losses = []

        all_predictions = []
        all_targets = []
        all_names = []   # per allineamento pred/gt con l'immagine

        with tqdm(test_dataloader_finetuning, leave=False, desc="Test") as t:
            for batch in t:
                x, y, paths = batch
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                with torch.no_grad():
                    y_pred = model(x)

                loss = criterion(y_pred, y.unsqueeze(1))
                t.set_postfix(loss=loss.item())
                test_losses.append(loss.item())

                # Salva pred/gt nello stesso ordine
                all_predictions.append(y_pred.squeeze(1).cpu().numpy())  
                all_targets.append(y.cpu().numpy())                    

                # Salva i nomi immagine (o path) puliti
                all_names.extend([os.path.basename(p) for p in paths])

        # Concatenazione batch
        all_predictions = np.concatenate(all_predictions, axis=0)  
        all_targets     = np.concatenate(all_targets, axis=0)     

        test_loss_mean = float(np.mean(test_losses))
        writer.add_scalar("test/loss", test_loss_mean, iteration)

        # Calcolo metriche ogni epoca
        sp_corr, sp_p = spearmanr(all_predictions, all_targets)
        kd_corr, kd_p = kendalltau(all_predictions, all_targets)
        pr_corr, pr_p = pearsonr(all_predictions, all_targets)

        # Stampa ogni epoca: loss + metriche
        print(
            f"Epoch {epoch+1} - Test Loss: {test_loss_mean:.6f} | "
            f"Spearman r={sp_corr:.6f} (p={sp_p:.2e}) | "
            f"Kendall τ={kd_corr:.6f} (p={kd_p:.2e}) | "
            f"Pearson r={pr_corr:.6f} (p={pr_p:.2e})"
        )

        # Salva le metriche, le predizioni e i ground truth solo all'ultima epoca
        if epoch == 99:  # perché range(100) quindi ultima epoca = 99
            # dizionari pred/gt
            pred_dict = {name: float(pred) for name, pred in zip(all_names, all_predictions)}
            gt_dict   = {name: float(gt)   for name, gt   in zip(all_names, all_targets)}

            with open("predictions_finetuning_dict.json", "w", encoding="utf-8") as f:
                json.dump(pred_dict, f, ensure_ascii=False, indent=2)

            with open("targets_finetuning_dict.json", "w", encoding="utf-8") as f:
                json.dump(gt_dict, f, ensure_ascii=False, indent=2)

            # metriche
            metrics = {
                "spearman_r": float(sp_corr), "spearman_p": float(sp_p),
                "kendall_tau": float(kd_corr), "kendall_p": float(kd_p),
                "pearson_r": float(pr_corr), "pearson_p": float(pr_p),
            }
            with open("correlation_metrics_last_epoch.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

        scheduler.step()



# se successivamente voglio caricare i dizionari, uso questo codice...
# with open("predictions_finetuning_dict.json", "r", encoding="utf-8") as f:
#     pred_dict = json.load(f)
# with open("targets_finetuning_dict.json", "r", encoding="utf-8") as f:
#     gt_dict = json.load(f)

# per esempio per visualizzarle...
# print(pred_dict["qualcheimmagine.tif"], gt_dict["qualcheimmagine.tif"])

# script di evaluation da usare eventualmente in una fase post-processing 
# (nel codice di training l’allineamento tra predizioni e ground truth è già garantito tramite il nome dell’immagine)
# common_keys = sorted(pred_dict.keys())
# preds = np.array([pred_dict[k] for k in common_keys])
# gts = np.array([gt_dict[k]   for k in common_keys])
# spearmanr(preds, gts)

