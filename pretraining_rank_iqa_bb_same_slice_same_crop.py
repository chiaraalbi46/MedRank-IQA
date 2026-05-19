""" Same as pretraining_rank_iqa_balanced_batches_v1.py but with same slice and same crop comparison (other than FD vs real)"""

from comet_ml import Experiment, OfflineExperiment
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from argparse import ArgumentParser
import torch
from torch import nn
# from pretraining import rescaling, random, TARGET_HW, random_patch_selection
import numpy as np 
import os 
from tqdm.auto import tqdm
from torchvision.models import vgg16, VGG16_Weights
import glob
from collections import defaultdict
import collections
import torchvision
import re
import random
from functools import partial

from pretraining_networks import Vgg16, SiameseRankIQA, Resnet18, Resnet18SiameseRankIQA, SqueezeNet1_1, SqueezeNet1_1SiameseRankIQA

from distortions_utils.distortions import gaussian_blur, lens_blur, motion_blur, jpeg, impulse_noise, multiplicative_noise, \
                        jitter, non_eccentricity_patch, pixelate, quantization, \
                        high_sharpen, linear_contrast_change, non_linear_contrast_change

from adaptive_margin import compute_severity, adaptive_ranking_loss

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Makes CuDNN deterministic (important!)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

base_seed = 42
set_global_seed(base_seed)

TARGET_HW = 224

import torch.nn.functional as F

def norm_min_max(x):
    x_min = torch.min(x)
    x_max = torch.max(x)

    x = (x - x_min) / (x_max - x_min)

    return x

def set_window_gpu(x, window_level=40, window_width=400):

    low = window_level - window_width / 2
    high = window_level + window_width / 2

    # Clamp in window
    x = torch.clamp(x, low, high)

    # Normalize to [0, 1]
    x = (x - low) / (high - low)

    return x

def rescaling(wind, fix=None):
    if wind is not None:
        if fix is None: 
            window_levels = [30, 40, 50]
            window_widths = [300, 400, 500]
            wl_id = random.randrange(3)
            ww_id = random.randrange(3)
            # return set_window_gpu(window_level=window_levels[wl_id], window_width=window_widths[ww_id]), 'soft tissue windowing'
            return partial(
                set_window_gpu,
                window_level=window_levels[wl_id],
                window_width=window_widths[ww_id]
            ), 'soft tissue windowing'
        else:
            # fix windowing for all
            # return set_window_gpu(), 'soft tissue windowing' 
            return partial(set_window_gpu), 'soft tissue windowing'
    else:
        return norm_min_max, 'min max'

def random_patch_selection(h, w, crop_size=TARGET_HW):

    top = random.randrange(0, max(1, h - crop_size))
    left = random.randrange(0, max(1, w - crop_size))

    return top, left

def ensure_224(x):
    _, h, w = x.shape

    # Se più piccola → pad
    if h < 224 or w < 224:
        pad_h = max(0, 224 - h)
        pad_w = max(0, 224 - w)
        x = F.pad(x, (0, pad_w, 0, pad_h))  # (left,right,top,bottom)

    return x

######## TORCHIVISION AUGMENTATIONS 
distortion_range = {
    "gaublur": [0.1, 0.5, 1, 2, 5],  # crescente
    "lensblur": [1, 2, 4, 6, 8],  # crescente
    "motionblur": [1, 2, 4, 6, 10],  # crescente
    # "colordiff": [1, 3, 6, 8, 12],
    # "colorshift": [1, 3, 6, 8, 12],
    # "colorsat1": [0.4, 0.2, 0.1, 0, -0.4],
    # "colorsat2": [1, 2, 3, 6, 9],
    # "jpeg2000": [16, 32, 45, 120, 170],
    "jpeg": [43, 36, 24, 7, 4],  # decrescente
    # "whitenoise": [0.001, 0.002, 0.003, 0.005, 0.01],
    # "whitenoiseCC": [0.0001, 0.0005, 0.001, 0.002, 0.003],
    "impulsenoise": [0.001, 0.005, 0.01, 0.02, 0.03],  # crescente
    "multnoise": [0.001, 0.005, 0.01, 0.02, 0.05],  # crescente
    # "brighten": [0.1, 0.2, 0.4, 0.7, 1.1],
    # "darken": [0.05, 0.1, 0.2, 0.4, 0.8],
    # "meanshift": [0, 0.08, -0.08, 0.15, -0.15],
    "jitter": [0.05, 0.1, 0.2, 0.5, 1],  # crescente
    "noneccpatch": [20, 40, 60, 80, 100],  # crescente
    "pixelate": [0.01, 0.05, 0.1, 0.2, 0.5],  # crescente
    "quantization": [20, 16, 13, 10, 7],  # decrescente
    # "colorblock": [2, 4, 6, 8, 10],
    "highsharpen": [1, 2, 3, 6, 12],  # crescente
    # "lincontrchange": [0., 0.15, -0.4, 0.3, -0.6],  # NON MONOTONO
    "lincontrchange": [-0.6, -0.4, 0., 0.15,  0.3],  # crescente
    "nonlincontrchange": [0.4, 0.3, 0.2, 0.1, 0.05],
}

# # NEW 
# distortion_range = {  
#     "gaublur":     [0.1, 0.3, 0.6, 1.0, 1.5],
#     "lensblur":    [1, 2, 3, 4, 5],
#     "motionblur":  [1, 2, 3, 5, 7],
#     "jpeg": [60, 50, 40, 30, 20],
#     "impulsenoise": [0.001, 0.003, 0.005, 0.01, 0.015],
#     "multnoise":    [0.001, 0.003, 0.005, 0.01, 0.02],
#     "jitter": [0.01, 0.03, 0.05, 0.1, 0.2],
#     "pixelate":     [0.01, 0.02, 0.05, 0.1, 0.2],
#     "quantization": [20, 16, 13, 10, 8],
#     "lincontrchange":     [0.0, 0.05, 0.1, 0.2, 0.3],
#     "nonlincontrchange":  [0.2, 0.15, 0.1, 0.05, 0.02],
#     "highsharpen": [1, 2, 3, 4, 6],
#     "noneccpatch": [20, 40, 60, 80],
# }


distortion_functions = {
    "gaublur": gaussian_blur,
    "lensblur": lens_blur,
    "motionblur": motion_blur,
    # "colordiff": color_diffusion,
    # "colorshift": color_shift,
    # "colorsat1": color_saturation1,
    # "colorsat2": color_saturation2,
    # "jpeg2000": jpeg2000,
    "jpeg": jpeg,
    # "whitenoise": white_noise,
    # "whitenoiseCC": white_noise_cc,
    "impulsenoise": impulse_noise,
    "multnoise": multiplicative_noise,
    # "brighten": brighten,
    # "darken": darken,
    # "meanshift": mean_shift,
    "jitter": jitter,
    "noneccpatch": non_eccentricity_patch,
    "pixelate": pixelate,
    "quantization": quantization,
    # "colorblock": color_block,
    "highsharpen": high_sharpen,
    "lincontrchange": linear_contrast_change,
    "nonlincontrchange": non_linear_contrast_change,
}

# def ranking_loss(s, s_hat, y, margin=0.5):  # y ...
#     # y = 0 --> s > s_hat

#     # Want s(x) > s(x_hat), so s - s_hat >= margin
#     diff = s - s_hat
#     loss = torch.clamp(((2*y - 1) * diff) + margin, min=0)

#     # diff = s_hat - s
#     # # loss = torch.clamp(((2*y - 1) * diff) + margin, min=0)
#     # loss = torch.clamp(diff + margin, min=0)
#     return loss.mean()

# def ranking_accuracy(s, s_hat, y):
#     # s, s_hat: [B, 1]
#     # return (s > s_hat).float().mean()
#     return ((2 * y - 1) * (s_hat - s) > 0).float().mean()

# CORRECTED
def ranking_loss(s, s_hat, y, margin=0.5):  # y ...
  
    diff = s - s_hat
    target = (1 - 2*y)

    # loss = torch.clamp((target * diff) + margin, min=0)
    loss = torch.nn.functional.softplus(target * diff)

    return loss.mean()

# CORRECTED
def ranking_accuracy(s, s_hat, y):
    # s, s_hat: [B, 1]

    diff = s - s_hat
    target = (1 - 2*y)

    return (-target * diff > 0).float().mean()

""" CHANGED """
def sample_degradation_(distortion_range):
    parametri = distortion_range

    artefatto = random.choice(list(parametri.keys()))
    # livello = random.choice(parametri[artefatto])
    level_idx = random.randrange(len(parametri[artefatto]))
    livello = parametri[artefatto][level_idx]
    
    return artefatto, livello, level_idx

""" CHANGED """
def sample_paired_degradation_(artefatto, level_x1_hat, level_idx_x1_hat, parametri):
    ### this create x2_hat based on x1_hat for comparison between degraded versions

    livelli = parametri[artefatto]  # la lista di valori da passare alla funzione associata a artefatto

    # remove the already used level
    filtered_levels = [l for l in livelli if l != level_x1_hat]
    level_x2_hat = random.choice(filtered_levels)
    level_idx_x2_hat = livelli.index(level_x2_hat)
    
    # level_idx_x2_hat = random.randrange(len(filtered_levels))  # così non va bene perchè non è rimappato su livelli !! 
    # level_x2_hat = filtered_levels[level_idx_x2_hat]
    # p2 = parametri[artefatto][level_x2_hat]

    quality_label = 0  # 0 means that x2_hat has higher quality than x1_hat
    # if level_x2_hat > level_x1_hat:  # sigma 2 > sigma 1 
    #     quality_label = 1  # means that x2_hat has lower quality with respect to x1_hat

    # # NEW
    # if artefatto not in ['jpeg', 'quantization', 'nonlincontrchange']:
    #     if level_x2_hat > level_x1_hat:  # sigma 2 > sigma 1 
    #         quality_label = 1  # means that x2_hat has lower quality with respect to x1_hat
    # else:
    #     # lista di perturbazioni monotona decrescente (più basso il valore, più la perturbazione impatta)
    #     if level_x2_hat > level_x1_hat:  # sigma 2 > sigma 1 
    #         quality_label = 0
    #     else:
    #         quality_label = 1
    
    # NEW NEW (uso gli i level_idx invece dei level per decidere quality_label)
    if level_idx_x2_hat > level_idx_x1_hat:  # sigma 2 > sigma 1
        quality_label = 1  # means that x2_hat has lower quality with respect to x1_hat (x2_hat è più perturbato di x1_hat)

    return level_x2_hat, quality_label, level_idx_x2_hat

def build_degradation_(artefatto, livello):

    def _apply(img: torch) -> torch:
      
        x_min = torch.min(img)
        x_max = torch.max(img)

        # porto img tra 0 e 1 
        img_01 = (img - x_min) / (x_max - x_min)

        func = distortion_functions[artefatto]
        param = livello

        out = func(img_01, param)

        # rescale
        out_rescaled = out * (x_max - x_min) + x_min

        return out_rescaled.to(torch.float32) # .astype(np.float32, copy=False)

    return _apply

def sample_degradation_real(list_art_levels):

    combination = random.choice(list(list_art_levels.keys()))
    
    return combination

class BaseDataset(Dataset):
    def __init__(self, mode='train'):
        root_path = f"/Prove/Albisani/TCIA_datasets/{mode}_cropped_npy"

        fd_files = glob.glob(f"{root_path}/*_fd.npy")

        pairs = []
        self.class_to_indices = defaultdict(list)

        for fd in sorted(fd_files):

            patient_id = fd.split('/')[-1][:4]

            if patient_id not in ['C095', 'L071', 'C246', 'L273']:

                ld = fd.replace("_fd.npy", "_ld.npy")
                if os.path.exists(ld):
                    pairs.append((fd, ld, patient_id))
                else:
                    print(f"⚠️ Missing LD for {fd}")

        # ora costruisci strutture coerenti
        self.image_files = []
        self.low_dose_images = []

        for idx, (fd, ld, patient_id) in enumerate(pairs):
            self.image_files.append(fd)
            self.low_dose_images.append(ld)
            self.class_to_indices[patient_id].append(idx)
        
        self.patients = list(self.class_to_indices.keys()) # id pazienti 

        self.arts_levels = ['streak_1_noise_0.1', 'streak_1_noise_0.5', 'streak_6_noise_0.1', 'streak_6_noise_0.5',
                            'streak_6_noise_1.0', 'streak_12_noise_0.1', 'streak_12_noise_0.5', 'streak_12_noise_1.0']
        
        self.streak_levels = [1, 6, 12]
        self.noise_levels = [0.1, 0.5, 1.0]
        
        self.artifact_dict = defaultdict(list)
        self.artifact_class_to_indices = defaultdict(lambda: defaultdict(list))
            
        for comb in self.arts_levels:
            files = glob.glob(f"/Prove/Albisani/TCIA_datasets/{mode}_{comb}_npy/*.npy")

            # key = path.split(f"{mode}_")[1].rsplit("_npy", 1)[0] # streak_1_noise_0.1

            for idx, path in enumerate(files):
                self.artifact_dict[comb].append(path)

                patient_id = path.split('/')[-1][:4]

                self.artifact_class_to_indices[comb][patient_id].append(idx)

    def __len__(self):
        return len(self.image_files)

    def load_fd(self, idx):
        x = np.load(self.image_files[idx]).astype(np.float32)
        return torch.from_numpy(x).unsqueeze(0).float()  # 1, H, W

    def load_ld(self, idx):
        x = np.load(self.low_dose_images[idx]).astype(np.float32)
        return torch.from_numpy(x).unsqueeze(0).float()  # 1, H, W
    
    def sample_idx_pair(self, idx):
        label = self.image_files[idx].split('/')[-1][:4]

        if random.random() <= 0.5:
            # stesso paziente,
            idx_1 = random.choice(self.class_to_indices[label])  # una slice a caso dello stesso paziente
        else:
            neg = random.choice([l for l in self.patients if l != label])  # paziente diverso 
            idx_1 = random.choice(self.class_to_indices[neg])  # una slice a caso del paziente diverso 

        return idx_1 # , label
    
    def sample_idx_pair_real(self, idx, comb):
        # simile a sample_idx_pair ma per campionare artefatti realistici 

        label = self.image_files[idx].split('/')[-1][:4]

        if random.random() <= 0.5:
            # stesso paziente
            idx_1 = random.choice(self.artifact_class_to_indices[comb][label])  # una slice a caso dello stesso paziente
        else:
            neg_label = random.choice([l for l in self.patients if l != label])
            idx_1 = random.choice(self.artifact_class_to_indices[comb][neg_label])
        
        return idx_1
    
    def sample_idx_pair_real_real(self, idx, comb1, comb2):
        # per real vs real devo ricampionare due indici (perchè non ho allineamento tra i full dose e quelli realistici). uso idx solo per recuperare l'indicazione sul paziente
        label = self.image_files[idx].split('/')[-1][:4]

        idx1 = random.choice(self.artifact_class_to_indices[comb1][label])

        # stesso paziente/diverso paziente 
        r = random.random()
        if r <= 0.5:
            # stesso paziente - 
            idx_2 = random.choice(self.artifact_class_to_indices[comb2][label])  # una slice a caso dello stesso paziente

        else:
            # diverso paziente 
            neg_label = random.choice([l for l in self.patients if l != label])  # un paziente diverso 

            idx_2 = random.choice(self.artifact_class_to_indices[comb2][neg_label])  # una slice a caso dello stesso paziente
        
        return idx1, idx_2

def random_crop(x, x_hat, cm='random'):
    if cm == 'random':
        crop_mode = "same" if random.random() <= 0.5 else "different"
    else:
        crop_mode = 'same'

    ### random patch
    h, w = x.shape[1], x.shape[2]  # x [1, H, W]
    
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

    x_crop = ensure_224(x_crop)
    x_hat_crop = ensure_224(x_hat_crop)

    return x_crop, x_hat_crop


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_len, pair_types, batch_size):
        self.dataset_len = dataset_len
        self.pair_types = pair_types
        self.batch_size = batch_size

        assert batch_size % len(pair_types) == 0
        self.per_type = batch_size // len(pair_types)

        self.num_batches = dataset_len // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for t in self.pair_types:
                for _ in range(self.per_type):
                    idx = random.randint(0, self.dataset_len - 1)
                    batch.append((idx, t))

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

#### curriculum batch sampler 
class CurriculumBalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_len, schedule, batch_size):
        """
        dataset_len: int -> lunghezza del dataset di base
        schedule: dict -> {epoca: [lista_tipi_coppia_attivi]}
                           es: {0: ['fd_vs_syn'], 5: ['fd_vs_syn', 'syn_vs_syn_easy'], 10: all}
        batch_size: int -> dimensione finale desiderata del batch (es. 120)
        """
        self.dataset_len = dataset_len
        self.schedule = schedule
        self.batch_size = batch_size
        self.num_batches = dataset_len // batch_size
        
        # Inizializzazione allo stato dell'epoca 0
        self.epoch = 0
        self._update_active_types()

    def _update_active_types(self):
        # Determina quali tipi di coppie sono attivi per l'epoca corrente
        milestones = sorted([k for k in self.schedule.keys() if k <= self.epoch])
        latest_milestone = milestones[-1]
        self.active_pair_types = self.schedule[latest_milestone]
        
        # Bilancia dinamicamente il batch in base a quanti tipi sono attivi
        num_active = len(self.active_pair_types)
        assert self.batch_size % num_active == 0, \
            f"Errore: batch_size ({self.batch_size}) non divisibile per il numero di coppie attive ({num_active}) all'epoca {self.epoch}"
        
        self.per_type = self.batch_size // num_active
        print(f"\n[Curriculum] Epoca {self.epoch} -> Tipi Attivi: {self.active_pair_types} | Esempi per tipo nel batch: {self.per_type}")

    def set_epoch(self, epoch):
        """Metodo da chiamare all'inizio di ogni epoca nel loop di training"""
        self.epoch = epoch
        self._update_active_types()

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            # Campiona solo dai tipi attivi in questa epoca
            for t in self.active_pair_types:
                for _ in range(self.per_type):
                    idx = random.randint(0, self.dataset_len - 1)
                    batch.append((idx, t))

            # Rimescola l'ordine interno del batch per non avere tutte le coppie dello stesso tipo vicine
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

class PairGenerator:
    def __call__(self, dataset, idx):
        raise NotImplementedError

## x vs x_hat low dose
class FDvsLD:
    def __call__(self, dataset, idx):
        x = dataset.load_fd(idx)

        # idx_1 = dataset.sample_idx_pair(idx)  # campiono l'altro indice (stesso paziente o diverso paziente)

        # x_hat = dataset.load_ld(idx_1)

        ## se voglio fare stesso confronto 
        x_hat = dataset.load_ld(idx)

        # return x, x_hat, {"type": "fd_ld"}  # quality(x) > quality(x_hat)
        return x, x_hat, {"type": "fd_ld",
                          "art_i": 'fd', 'level_i': -1, 'level_idx_i': -1,
                          "art_j": 'ld', 'level_j': -1, 'level_idx_j': -1,
                          "quality_label": 1}  # x_hat low quality with respect to x  

## x vs x_hat synthetic (uso torchvision ora)
class FDvsSynthetic(PairGenerator):
    def __call__(self, dataset, idx):
        x = dataset.load_fd(idx)

        # idx_1 = dataset.sample_idx_pair(idx)  # campiono l'altro indice (stesso paziente o diverso paziente)
        # x1 = dataset.load_fd(idx_1)

        # applico artefatto random su x1 
        # art, level = sample_degradation_(distortion_range)
        art, level, level_idx = sample_degradation_(distortion_range)
        # x_hat = build_degradation_(artefatto=art, livello=level)(x1)

        ## se voglio fare stesso confronto 
        x_hat = build_degradation_(artefatto=art, livello=level)(x) 
        # e commento idx_1, x1

        # return x, x_hat, {"type": "fd_syn", "art": art, 'level': level, 'level_idx': level_idx}  # quality(x) > quality(x_hat)
        return x, x_hat, {"type": "fd_syn", 
                          "art_i": 'fd', 'level_i': -1, 'level_idx_i': -1,
                          "art_j": art, 'level_j': level, 'level_idx_j': level_idx,
                          "quality_label": 1}  # x_hat low quality with respect to x 
    
## x vs x_hat real
class FDvsReal(PairGenerator):
    def __call__(self, dataset, idx):

        x = dataset.load_fd(idx) # 1, H, W

        comb = sample_degradation_real(dataset.artifact_dict)

        # uso self.arts_levels come riferimento per avere level e level_idx (level tanto non lo uso)
        streak_val, noise_val = re.search(r"streak_(\d+)_noise_([0-9]*\.?[0-9]+)", comb).groups() 
        level_j = int(streak_val) + float(noise_val)
        level_idx_j = dataset.arts_levels.index(comb)

        # TODO: come potrei fare per rendere il confronto più corretto (confronto con una slice più vicina possibile a quella che ho preso) possibile ?

        idx_1 = dataset.sample_idx_pair_real(idx, comb)  # campiono l'altro indice (stesso paziente o diverso paziente)

        # if random.random() <= 0.5:
        #     # stesso paziente
        #     idx_1 = random.choice(dataset.artifact_class_to_indices[comb][label])  # # una slice a caso dello stesso paziente
        # else:
        #     neg_label = random.choice([l for l in dataset.patients if l != label])
        #     idx_1 = random.choice(dataset.artifact_class_to_indices[comb][neg_label])

        img_path_1 = dataset.artifact_dict[comb][idx_1]

        # carico l'immagine con l'artefatto campionato
        x_hat = np.load(img_path_1).astype(np.float32)
        x_hat = torch.from_numpy(x_hat).unsqueeze(0).float()  # 1, H, W

        # return x, x_hat, {"type": "fd_real", "comb": comb}  # quality(x) > quality(x_hat)
        return x, x_hat, {"type": "fd_real", 
                          "art_i": 'fd', 'level_i': -1, 'level_idx_i': -1,
                          "art_j": comb, 'level_j': level_j, 'level_idx_j': level_idx_j, 
                          "quality_label": 1}  # x_hat low quality with respect to x 

## x_hat_1 vs x_hat_2 synthetic
class SynVsSyn(PairGenerator):
    def __call__(self, dataset, idx):

        x = dataset.load_fd(idx)

        # art1, level_x1_hat = sample_degradation_(distortion_range)
        art1, level_x1_hat, level_idx_x1_hat = sample_degradation_(distortion_range)
        # applico l'artefatto campionato a x
        x1_hat = build_degradation_(artefatto=art1, livello=level_x1_hat)(x)

        # level_x2_hat, quality_label = sample_paired_degradation_(
        #     artefatto=art1,
        #     level_x1_hat=level_x1_hat,
        #     parametri=distortion_range
        # )

        level_x2_hat, quality_label, level_idx_x2_hat = sample_paired_degradation_(
            artefatto=art1,
            level_x1_hat=level_x1_hat,
            level_idx_x1_hat=level_idx_x1_hat,
            parametri=distortion_range
        )

        ## se voglio stesso confronto x1_hat e x2_hat devono generarsi entrambe da x
        x2_hat = build_degradation_(artefatto=art1, livello=level_x2_hat)(x)

        # return x1_hat, x2_hat, {"type": "syn_syn", "art1": art1,
        #                          "level_x1_hat": level_x1_hat, "level_x2_hat": level_x2_hat, "quality_label": quality_label,
        #                          "level_idx_x1_hat": level_idx_x1_hat, "level_idx_x2_hat": level_idx_x2_hat} # quality_label

        return x1_hat, x2_hat, {"type": "syn_syn", 
                                "art_i": art1, "level_i": level_x1_hat, "level_idx_i": level_idx_x1_hat,
                                "art_j": art1, "level_j": level_x2_hat, "level_idx_j": level_idx_x2_hat,
                                "quality_label": quality_label} # quality_label

## x_hat_1 vs x_hat 2 real
class RealVsReal(PairGenerator):
    def __call__(self, dataset, idx):

        comb1 = sample_degradation_real(dataset.artifact_dict)

        #### andrebbe in una funzione
        streak_val1, noise_val1 = re.search(r"streak_(\d+)_noise_([0-9]*\.?[0-9]+)", comb1).groups()

        if random.random() <= 0.5:
            # same streak, diff noise
            candidati = []
            for s in dataset.arts_levels:
                match = re.search(r"streak_(\d+)_noise_([0-9]*\.?[0-9]+)", s)
                if match:
                    streak = match.group(1)
                    noise = match.group(2)
                    
                    if streak == streak_val1 and noise != noise_val1:
                        candidati.append(s)

            comb2 = random.choice(candidati)
            streak_val2, noise_val2 = re.search(r"streak_(\d+)_noise_([0-9]*\.?[0-9]+)", comb2).groups()  # streak_val2 == streak_val1

            ## levels and level_idxs
            level_i = float(noise_val1)
            level_idx_i = dataset.noise_levels.index(level_i)
            level_j = float(noise_val2)
            level_idx_j = dataset.noise_levels.index(level_j)

            quality_label = 0
            # if float(noise_val2) < float(noise_val1):  # minore è il noise, minore è la qualità
            #     quality_label = 1  # x2_hat low quality with respect to x1_hat
            if float(noise_val2) > float(noise_val1):  # maggiore è il noise, minore è la qualità
                quality_label = 1  # x2_hat low quality with respect to x1_hat

        else:
            # same noise, diff streak
            candidati = []
            for s in dataset.arts_levels:
                match = re.search(r"streak_(\d+)_noise_([0-9]*\.?[0-9]+)", s)
                if match:
                    streak = match.group(1)
                    noise = match.group(2)
                    
                    if streak != streak_val1 and noise == noise_val1:
                        candidati.append(s)

            comb2 = random.choice(candidati)
            streak_val2, noise_val2 = re.search(r"streak_(\d+)_noise_([0-9]*\.?[0-9]+)", comb2).groups()  # streak_val2 == streak_val1

            ## levels and level_idxs
            level_i = int(streak_val1)
            level_idx_i = dataset.streak_levels.index(level_i)
            level_j = int(streak_val2)
            level_idx_j = dataset.streak_levels.index(level_j)

            quality_label = 0
            if int(streak_val2) > int(streak_val1):  # maggiore è lo streak, minore è la qualità
                quality_label = 1 # # x2_hat low quality with respect to x1_hat
        ####

        # sample slices
        # la slice in realtà la cambio (l'originale non va bene - devo prendere una con l'artefatto che non avrà lo stesso indice)

        # idx_1, idx_2 = dataset.sample_idx_pair_real_real(idx, comb1, comb2)
        # img_path_1 = dataset.artifact_dict[comb1][idx_1]
        # img_path_2 = dataset.artifact_dict[comb2][idx_2]

        # in realtà posso usare un'indice unico per real vs real (sono mappabili una volta trasformate ...dovrebbero)
        # posso campionare un solo indice 
        idx_1 = dataset.sample_idx_pair_real(idx, comb1)  
        img_path_1 = dataset.artifact_dict[comb1][idx_1]
        img_path_2 = dataset.artifact_dict[comb2][idx_1]

        x1_hat = np.load(img_path_1).astype(np.float32)
        x2_hat = np.load(img_path_2).astype(np.float32)

        x1_hat = torch.from_numpy(x1_hat).unsqueeze(0).float()
        x2_hat = torch.from_numpy(x2_hat).unsqueeze(0).float()

        # return x1_hat, x2_hat, {"type": "real_real", "comb1": comb1,
        #                          "comb2": comb2, "quality_label": quality_label} 
        return x1_hat, x2_hat, {"type": "real_real", 
                                "art_i": comb1, 'level_i': level_i, 'level_idx_i': level_idx_i,
                                "art_j": comb2, 'level_j': level_j, 'level_idx_j': level_idx_j, 
                                "quality_label": quality_label} 

# # TODO: modifica ... se metto per tutti quality label in meta, non devo distinguere  
# def build_label(meta): 
#     t = meta["type"]

#     if t in ["fd_ld", "fd_syn", "fd_real"]:
#         return 0  # quality(x) > quality(x_hat)

#     elif t in ["syn_syn", "real_real"]:
#         return meta['quality_label'] # 0 if meta["level1"] < meta["level2"] else 1

#     # elif t == "real_real":
#     #     return compare_artifacts(meta)

#     else:
#         raise ValueError(t)
    
def apply_random_swap(x, x_hat, y):
    # y = random.randint(0, 1)
    # if y == 0:
    #     print("No swap applied")
    #     return x, x_hat, y
    # else:
    #     print("Swap applied")
    #     return x_hat, x, y

    if random.random() < 0.5:
        # print("Swap applied")
        return x_hat, x, 1 - y
    
    return x, x_hat, y

class RankIQADataset(Dataset):
    def __init__(self, base_dataset, generators, pair_types, rescaling_fn, cm):
        self.base = base_dataset  # Contiene le liste file e dizionari artefatti
        self.generators = generators
        self.pair_types = pair_types
        self.rescaling_fn = rescaling_fn
        self.cm = cm  # crop_mode random o same 

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index_info):
        # index_info è la tupla (idx, tipo_coppia) passata dal BatchSampler
        idx, t = index_info
        
        # 1. Generazione Coppia (Logica ex-Generatori)
        gen = self.generators[t]
        x, x_hat, meta = gen(self.base, idx)
        
        # 2. Labeling
        y = meta['quality_label']  # build_label(meta)
        
        # 3. Crop and Swap
        x, x_hat = random_crop(x, x_hat, cm=self.cm)
        # print("Before swap: y =", y)
        x, x_hat, y = apply_random_swap(x, x_hat, y)

        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
            x_hat = x_hat.repeat(3, 1, 1)
            
        if self.rescaling_fn:
            x = self.rescaling_fn(x)
            x_hat = self.rescaling_fn(x_hat)
            
        # return x, x_hat, torch.tensor(y, dtype=torch.float32), t
        return x, x_hat, torch.tensor(y, dtype=torch.float32), t, meta  # ADDED

@torch.no_grad()
def evaluate_detailed(model, loader, device, experiment, epoch, margin):
    model.eval()

    # TODO: sistema
    # m0=0.1
    # alpha=0.3
    # gamma=2.0
    # m0=0.0
    # alpha=1.0
    
    # Dizionari per accumulare metriche divise per tipo
    metrics_by_type = {t: {'loss': [], 'acc': []} for t in loader.dataset.pair_types}
    
    # for x, x_hat, y, pair_types in tqdm(loader, desc=f"Epoch {epoch+1} [Detailed Test]"):
    for x, x_hat, y, pair_types, meta in tqdm(loader, desc=f"Epoch {epoch+1} [Detailed Test]"):
        x, x_hat, y = x.to(device), x_hat.to(device), y.unsqueeze(1).to(device)  # y è B --> lo rendo B, 1
        
        s, s_hat = model(x, x_hat)
        
        # Calcolo loss e acc per l'intero batch
        diff = s - s_hat
        target = (1 - 2 * y) 

        ## hinge loss
        # losses = torch.clamp(((2*y - 1) * diff) + margin, min=0)
        # losses = torch.clamp((target * diff) + margin, min=0)  

        ## soft
        # losses = torch.nn.functional.softplus(target * diff)

        ## adaptive loss
        alpha = 2.0
        severity_i, severity_j = compute_severity(meta=meta, distortion_range=distortion_range, real_arts_levels=loader.dataset.base.arts_levels, real_streak_levels=loader.dataset.base.streak_levels, real_noise_levels=loader.dataset.base.noise_levels, device=device)
        delta = torch.abs(severity_i - severity_j)
        expo = alpha * delta * target * diff
        losses = torch.log(1 + torch.exp(expo))


        # ## NEW
        # severity_i, severity_j = compute_severity(meta=meta, distortion_range=distortion_range, device=device)

        # ####  adaptive_ranking_loss(s, s_hat, y, severity_i, severity_j)  # no perchè fa .mean()
        # delta = torch.abs(severity_i - severity_j)
        # # marg = m0 + alpha * delta
        # # # weight = torch.exp(-gamma * delta)
        # # losses = torch.clamp(((2*y - 1) * diff) + marg, min=0)
        # # losses = weight * losses
        # ####

        # #####  DA ALLINEARE CON diff e target 
        # # target = (2 * y - 1) 
        # # losses = torch.log(1 + torch.exp(-alpha * delta * target * diff))
        # losses =  torch.nn.functional.softplus(-target * diff * (1 + delta))
        # #####

        # accs = ((2 * y - 1) * (s_hat - s) > 0).float()  # NO 
        # accs = ((1 - 2*y) * (s_hat - s) > 0).float()  # OK 
        accs = (-target * diff > 0).float()  # OK per allinearsi a diff e target 

        # Distribuiamo i risultati nei bucket corretti
        for i, t in enumerate(pair_types):
            metrics_by_type[t]['loss'].append(losses[i].item())
            metrics_by_type[t]['acc'].append(accs[i].item())

    # logs
    total_acc = []
    total_loss = []
    for t, results in metrics_by_type.items():
        avg_l = np.mean(results['loss'])
        avg_a = np.mean(results['acc'])

        total_acc.append(avg_a)
        total_loss.append(avg_l)
        
        experiment.log_metric(f"test_{t}_loss", avg_l, step=epoch)
        experiment.log_metric(f"test_{t}_accuracy", avg_a, step=epoch)
        print(f"Type {t:10} | Loss: {avg_l:.4f} | Acc: {avg_a:.4f}")

    avg_total_acc = np.mean(total_acc)
    avg_total_loss = np.mean(total_loss)
    experiment.log_metric("test_total_accuracy", avg_total_acc, step=epoch)
    experiment.log_metric("test_total_loss", avg_total_loss, step=epoch)
    return avg_total_acc, avg_total_loss

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_epochs', dest="n_epochs",type=int, default=30, help='number of epochs for training')
    parser.add_argument('--file_path', dest="file_path", type=str, help='file name for saving pretrained model', default=None)

    # comet parameters
    parser.add_argument("--comet", dest="comet", default=1, help="1 for comet ON, 0 for comet OFF")
    parser.add_argument("--name_proj", dest="name_proj", default='medrank-iqa-pretraining-new', help="define comet ml project folder")
    parser.add_argument("--name_exp", dest="name_exp", default='tmp', help="name of comet ml experiment")

    parser.add_argument("--batch_size", dest="batch_size", default=5, help="batch size for train and test")

    parser.add_argument('--device_id', dest="device_id",  default='0', help='gpu device id.')
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=1e-3, help="base learning rate")

    parser.add_argument("--windowing", dest="windowing", type=float, default=1,
                    help="1 is soft tissue windowing, None for norm min max")
    parser.add_argument("--fix_windowing", dest="fix_windowing", type=float, default=None,
                    help="1 is for fix soft tissue windowing for all pairs, None for random windowing among pairs")

    parser.add_argument('--imagenet_initialization', dest="imagenet_initialization",  default=None, help='use (1) or not (None) imagenet weights for VGG16')
    # parser.add_argument('--test_case', dest="test_case",  default='all', 
    #                 help='test case for pair combination. all, all_same, synthetic, low_dose, same_artifact_diff_levels.')
    parser.add_argument('--margin', dest="margin",  default=0.5, help='margin for ranking loss')

    parser.add_argument('--network_model', dest="network_model",  default='squeezenet1_1', help='specify the network to use. vgg16, resnet18')

    parser.add_argument('--crop_mode', dest="crop_mode",  default='same', help='same or random (compare same or different patches in the pair)')

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
        # "test_case": args.test_case,
        "windowing": args.windowing,
        "fix_windowing": args.fix_windowing,
        "margin": float(args.margin),
        "imagenet_initialization": args.imagenet_initialization,
        "crop_mode": args.crop_mode
    })

    # ##
    # vgg = Vgg16(imagenet=args.imagenet_initialization)
    # model = SiameseRankIQA(vgg_model=vgg)

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

    # optim = Adam(model.parameters(), lr=float(args.learning_rate))
    optim = Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4)

    #### new 
    pair_types = ["fd_ld", "fd_syn", "fd_real", "syn_syn", "real_real"]
    # pair_types = ["fd_ld", "fd_syn", "fd_real"]
    # pair_types = ["fd_ld", "fd_syn"]
    # pair_types = ["fd_syn", "syn_syn"]
    # pair_types = ["syn_syn"]

    # Dataset
    generators = {
        "fd_ld": FDvsLD(),
        "fd_syn": FDvsSynthetic(),
        "fd_real": FDvsReal(),
        "syn_syn": SynVsSyn(),
        "real_real": RealVsReal(),
    }

    # 2. Crea i BaseDataset (quelli che hanno le liste file)
    base_train = BaseDataset(mode="train")
    base_test = BaseDataset(mode="test")

    # 3. Crea i RankIQADataset (quelli che usano i generatori)
    train_ds = RankIQADataset(base_train, generators, pair_types, rescaling_fn, cm=args.crop_mode)
    test_ds = RankIQADataset(base_test, generators, pair_types, rescaling_fn, cm=args.crop_mode)

    # Samplers
    # train_sampler = BalancedBatchSampler(len(train_ds), pair_types, int(args.batch_size))

    # train_schedule = {
    #     0:  ['fd_vs_syn_heavy'],
    #     3:  ['fd_vs_syn_heavy', 'fd_vs_syn_all'],
    #     6:  ['fd_vs_syn_heavy', 'fd_vs_syn_all', 'syn_vs_syn_distant'],
    #     9:  ['fd_vs_syn_heavy', 'fd_vs_syn_all', 'syn_vs_syn_distant', 'syn_vs_syn_close']
    # }
    train_schedule = {
        0:  ['fd_ld'],
        3:  ['fd_ld', 'fd_real'],
        7:  ['fd_ld', 'fd_real', 'fd_syn'],
        10:  ['fd_ld', 'fd_real', 'fd_syn', 'syn_syn'],
        13:  ['fd_ld', 'fd_real', 'fd_syn', 'syn_syn', 'real_real'],
    }

    train_sampler = CurriculumBalancedBatchSampler(
        dataset_len=len(train_ds),
        schedule=train_schedule,
        batch_size=int(args.batch_size)
    )

    # test - campionamento bilanciato o random...
    test_sampler = BalancedBatchSampler(len(test_ds), pair_types, int(args.batch_size))

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=8, pin_memory=True)
    
    test_loader = DataLoader(test_ds, batch_sampler=test_sampler, num_workers=4, pin_memory=True)

    for epoch in range(args.n_epochs):

        model.train()
        train_losses = []

        with tqdm(train_loader, leave=False, desc="Training") as t:

            # Riconfigura il bilanciamento interno del batch all'inizio di ogni epoca
            train_loader.batch_sampler.set_epoch(epoch)

            # for x, x_hat, y, pair_types in t:
            for x, x_hat, y, pair_types, meta in t:
                x, x_hat, y = x.to(device), x_hat.to(device), y.unsqueeze(1).to(device)  # y è B --> lo rendo B, 1
                
                optim.zero_grad()
                s, s_hat = model(x, x_hat) # B, 1 and B,1

                # loss = ranking_loss(s, s_hat, y, margin=float(args.margin))

                ## NEW
                severity_i, severity_j = compute_severity(meta=meta, distortion_range=distortion_range, real_arts_levels=train_loader.dataset.base.arts_levels, real_streak_levels=train_loader.dataset.base.streak_levels, real_noise_levels=train_loader.dataset.base.noise_levels, device=device)
                loss = adaptive_ranking_loss(s, s_hat, y, severity_i, severity_j)

                # t.set_postfix(loss=loss.item())

                with torch.no_grad():
                    s_mean = s.mean().item()
                    sh_mean = s_hat.mean().item()
                    s_min = s.min().item()
                    s_max = s.max().item()
                    # Calcoliamo al volo l'accuracy del batch per monitorarla
                    acc = ranking_accuracy(s, s_hat, y).item()

                # Aggiorna la barra tqdm mostrando loss, accuracy e scala dei punteggi
                t.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{acc:.2f}",
                    s_avg=f"{s_mean:.2f}",
                    sh_avg=f"{sh_mean:.2f}",
                    s_bounds=f"[{s_min:.1f},{s_max:.1f}]"
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                
                train_losses.append(loss.item())
                
        train_loss_mean = np.mean(train_losses)
        experiment.log_metric("train_loss", train_loss_mean, step=epoch)

        ## test
        evaluate_detailed(model, test_loader, device, experiment, epoch, float(args.margin))


    torch.save(model.state_dict(), os.path.join(base_pretrained_folder, save_file_name + '.pth')) # per salvare il modello
    # log model ? 
    experiment.end()