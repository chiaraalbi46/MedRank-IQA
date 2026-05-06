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
from collections import defaultdict
import re

# from distortions_utils.distortions import *

from distortions_utils.distortions import gaussian_blur, lens_blur, motion_blur, jpeg, impulse_noise, multiplicative_noise, \
                        jitter, non_eccentricity_patch, pixelate, quantization, \
                        high_sharpen, linear_contrast_change, non_linear_contrast_change

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

import torch.nn.functional as F

def ensure_224(x):
    _, h, w = x.shape

    # Se più piccola → pad
    if h < 224 or w < 224:
        pad_h = max(0, 224 - h)
        pad_w = max(0, 224 - w)
        x = F.pad(x, (0, pad_w, 0, pad_h))  # (left,right,top,bottom)

    # # Se più grande → resize
    # if x.shape[1] != 224 or x.shape[2] != 224
    #     x = F.interpolate(
    #         x.unsqueeze(0),
    #         size=(224, 224),
    #         mode="bilinear",
    #         align_corners=False
    #     ).squeeze(0)

    return x

######## TORCHIVISION AUGMENTATIONS 
distortion_range = {
    "gaublur": [0.1, 0.5, 1, 2, 5],
    "lensblur": [1, 2, 4, 6, 8],
    "motionblur": [1, 2, 4, 6, 10],
    # "colordiff": [1, 3, 6, 8, 12],
    # "colorshift": [1, 3, 6, 8, 12],
    # "colorsat1": [0.4, 0.2, 0.1, 0, -0.4],
    # "colorsat2": [1, 2, 3, 6, 9],
    # "jpeg2000": [16, 32, 45, 120, 170],
    "jpeg": [43, 36, 24, 7, 4],
    # "whitenoise": [0.001, 0.002, 0.003, 0.005, 0.01],
    # "whitenoiseCC": [0.0001, 0.0005, 0.001, 0.002, 0.003],
    "impulsenoise": [0.001, 0.005, 0.01, 0.02, 0.03],
    "multnoise": [0.001, 0.005, 0.01, 0.02, 0.05],
    # "brighten": [0.1, 0.2, 0.4, 0.7, 1.1],
    # "darken": [0.05, 0.1, 0.2, 0.4, 0.8],
    # "meanshift": [0, 0.08, -0.08, 0.15, -0.15],
    "jitter": [0.05, 0.1, 0.2, 0.5, 1],
    "noneccpatch": [20, 40, 60, 80, 100],
    "pixelate": [0.01, 0.05, 0.1, 0.2, 0.5],
    "quantization": [20, 16, 13, 10, 7],
    # "colorblock": [2, 4, 6, 8, 10],
    "highsharpen": [1, 2, 3, 6, 12],
    "lincontrchange": [0., 0.15, -0.4, 0.3, -0.6],
    "nonlincontrchange": [0.4, 0.3, 0.2, 0.1, 0.05],
}

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

def sample_degradation_(distortion_range):
    parametri = distortion_range

    artefatto = random.choice(list(parametri.keys()))
    livello = random.choice(parametri[artefatto])
    
    return artefatto, livello

def sample_paired_degradation_(artefatto, level_x1_hat, parametri):
    ### this create x2_hat based on x1_hat for comparison between degraded versions

    livelli = parametri[artefatto]  # la lista di valori da passare alla funzione associata a artefatto

    # remove the already used level
    filtered_levels = [l for l in livelli if l != level_x1_hat]
    level_x2_hat = random.choice(filtered_levels)
    # p2 = parametri[artefatto][level_x2_hat]

    quality_label = 0  # 0 means that x2_hat has higher quality than x1_hat
    if level_x2_hat > level_x1_hat:  # sigma 2 > sigma 1 
        quality_label = 1  # means that x2_hat has lower quality with respect to x1_hat
    
    return level_x2_hat, quality_label

    # livelli = list(parametri[artefatto].keys())

    # # remove the already used level
    # filtered_levels = [l for l in livelli if l != level_x1_hat]
    # level_x2_hat = random.choice(filtered_levels)
    # p2 = parametri[artefatto][level_x2_hat]

    # quality_label = 0  # 0 means that x2_hat has higher quality than x1_hat
    # if level_x2_hat > level_x1_hat:  # sigma 2 > sigma 1 
    #     quality_label = 1  # means that x2_hat has lower quality with respect to x1_hat
    
    # return p2, quality_label

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

#########################################  refactoring with generators 




#########################################

class DegradedPairDataset_1(Dataset):
    # def __init__(self, root_path="/Prove/Albisani/TCIA_datasets/train", mode='train'):
    def __init__(self, mode='train', test_case='all'):

        root_path=f"/Prove/Albisani/TCIA_datasets/{mode}_cropped_npy"

        fd_files = glob.glob(f"{root_path}/*_fd.npy")

        pairs = []
        self.class_to_indices = defaultdict(list)
        
        # for i, fd in enumerate(sorted(fd_files)):

        #     patient_id = fd.split('/')[-1][:4]

        #     if patient_id not in ['C095', 'L071', 'C246', 'L273']: #### devo considerare i pazienti che sono falliti nel calcolo degli artefatti realistici

        #         ld = fd.replace("_fd.npy", "_ld.npy")
        #         if os.path.exists(ld):
        #             pairs.append((fd, ld))
        #         else:
        #             print(f"⚠️ Missing LD for {fd}")
                
        #         self.class_to_indices[patient_id].append(i)  # per ogni paziente ho una lista di id delle slices a lui corrispondenti 

        # self.patients = list(self.class_to_indices.keys()) # id pazienti 

        # pairs.sort()  # stable order

        # self.image_files = [p[0] for p in pairs]
        # self.low_dose_images = [p[1] for p in pairs]

        pairs = []

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

        self.test_case = test_case

        ### artefatti realistici 
        # self.root_paths_arts = [f'/Prove/Albisani/TCIA_datasets/{mode}_streak_1_noise_0.1_npy', f'/Prove/Albisani/TCIA_datasets/{mode}_streak_1_noise_0.5_npy',
        #                         f'/Prove/Albisani/TCIA_datasets/{mode}_streak_6_noise_0.1_npy',  f'/Prove/Albisani/TCIA_datasets/{mode}_streak_6_noise_0.5_npy',
        #                         f'/Prove/Albisani/TCIA_datasets/{mode}_streak_6_noise_1.0_npy', f'/Prove/Albisani/TCIA_datasets/{mode}_streak_12_noise_0.1_npy',
        #                         f'/Prove/Albisani/TCIA_datasets/{mode}_streak_12_noise_0.5_npy', f'/Prove/Albisani/TCIA_datasets/{mode}_streak_12_noise_1.0_npy']

        self.arts_levels = ['streak_1_noise_0.1', 'streak_1_noise_0.5', 'streak_6_noise_0.1', 'streak_6_noise_0.5',
                            'streak_6_noise_1.0', 'streak_12_noise_0.1', 'streak_12_noise_0.5', 'streak_12_noise_1.0']

        # mi serve un class_indices per paziente che consideri le slices generate (che sono meno di quelle originali)
        # l''idea dovrebbe essere che posso scegliere artefatto+livello (combinazione) e poi la slice di uno stesso paziente (tra quelle possibili)
        # o una slice di un diverso paziente 
        # mi serve un dizionario con chiave artefatto+livello e values la lista di path delle slices 
        # e mi servirebbe un class_to_indices per ogni artefatto+livello 

        # for r in self.root_paths_arts:
            # files = glob.glob(f"{r}/*.npy")
        
        self.artifact_dict = defaultdict(list)
        self.artifact_class_to_indices = defaultdict(lambda: defaultdict(list))
            
        for comb in self.arts_levels:
            files = glob.glob(f"/Prove/Albisani/TCIA_datasets/{mode}_{comb}_npy/*.npy")

            # key = path.split(f"{mode}_")[1].rsplit("_npy", 1)[0] # streak_1_noise_0.1

            for idx, path in enumerate(files):
                self.artifact_dict[comb].append(path)

                patient_id = path.split('/')[-1][:4]

                self.artifact_class_to_indices[comb][patient_id].append(idx)

        ###

        # self.postprocess = Windowing(window_level=40, window_width=400)  # soft-tissue window
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx] 
        label = img_path.split('/')[-1][:4]
        # print("path: ", anchor_path)

        r = random.random()

        if r <= 0.5:
            # stesso paziente
            idx_1 = random.choice(self.class_to_indices[label])  # una slice a caso dello stesso paziente
            img_path_1 = self.image_files[idx_1]
            # print("Positive path: ", pos_path)
        else:
            # diverso paziente 
            neg_label = random.choice([l for l in self.patients if l != label])  # un paziente diverso 
            idx_1 = random.choice(self.class_to_indices[neg_label])  # una slice a caso del paziente diverso 
            img_path_1 = self.image_files[idx_1]
            # print("Negative path: ", neg_path)

        x = np.load(img_path).astype(np.float32)  # H, W  np

        quality_label = None
        crop_mode = 'same'

        if self.test_case == 'full_syn_real': # full_syn + artefatti realistici
            # posso avere una slice a caso dello stesso paziente (a patto di togliere quelli che sono falliti per ora ...)
            # oppure una slice a caso da paziente diverso - però devo utilizzare class_to_indices diversi 
            # i casi possono essere: x vs low dose, x vs x_hat sintetica, x vs x_hat realistica, x_hat1 vs x_hat2 stesso artefatto sintetico ma diff livello
            # x_hat1 vs x_hat2 stesso artefatto realistico ma diff livello 
            # serve qualcosa tipo sample_degradation e build degradation etc per artefatti realistici - ma no ho già generato i dati :'(
            # però comunque posso mantenere l'idea solo che invece che generarli on the fly li prelevo  

            crop_mode = "same" if random.random() <= 0.5 else "different"
            r = random.random()

            if r <= 0.2:
                x = torch.from_numpy(x).unsqueeze(0).float()      # 1, H, W torch

                #### x vs x_hat (low dose)
                # low_dose_img_path = self.low_dose_images[idx]

                low_dose_img_path = self.low_dose_images[idx_1]  # la seconda slice è low dose 

                x_hat = np.load(low_dose_img_path).astype(np.float32)
                x_hat = torch.from_numpy(x_hat).unsqueeze(0).float()  # 1, H, W
            
            elif 0.2 < r <= 0.4:

                #### x vs x_hat sintetico

                # la degradazione la faccio su img1
                x1 = np.load(img_path_1).astype(np.float32) 
                x1 = torch.from_numpy(x1).unsqueeze(0).float()

                art, level_x_hat = sample_degradation_(distortion_range)
                x_hat = build_degradation_(artefatto=art, livello=level_x_hat)(x1)

                x = torch.from_numpy(x).unsqueeze(0).float()  
            
            elif 0.4 < r <= 0.6:

                #### x vs x_hat realistico

                ## campiono artefatto+livello
                comb = sample_degradation_real(self.artifact_dict)

                ## seleziono la slice
                r = random.random()
                if r <= 0.5:
                    # stesso paziente
                    idx_1 = random.choice(self.artifact_class_to_indices[comb][label])  # una slice a caso dello stesso paziente
                    img_path_1 = self.artifact_dict[comb][idx_1]

                    # idx_1 = random.choice(self.class_to_indices[label])  # una slice a caso dello stesso paziente
                    # img_path_1 = self.image_files[idx_1]
                    # print("Positive path: ", pos_path)
                else:
                    # diverso paziente 
                    neg_label = random.choice([l for l in self.patients if l != label])  # un paziente diverso 
                    idx_1 = random.choice(self.artifact_class_to_indices[comb][neg_label])  # una slice a caso del paziente diverso 
                    img_path_1 = self.artifact_dict[comb][idx_1]
                
                    # idx_1 = random.choice(self.class_to_indices[neg_label])  # una slice a caso del paziente diverso 
                    # img_path_1 = self.image_files[idx_1]
                    # print("Negative path: ", neg_path)

                # la carico
                # x1 = np.load(img_path_1).astype(np.float32) 
                # x1 = torch.from_numpy(x1).unsqueeze(0).float()
                x_hat = np.load(img_path_1).astype(np.float32) 
                x_hat = torch.from_numpy(x_hat).unsqueeze(0).float()

                x = torch.from_numpy(x).unsqueeze(0).float() 

            elif 0.6 < r <= 0.8:

                #### x1_hat vs x2_hat realistico
                # tipo stesso valore di streak e diverso noise 
                # stesso noise e diverso streak 

                comb1 = sample_degradation_real(self.artifact_dict)
                # devo splittare comb in streak e noise con i loro valori e prendere o stesso streak e diverso noise o viceversa

                streak_val1, noise_val1 = re.search(r"streak_(\d+)_noise_([0-9]*\.?[0-9]+)", comb1).groups()

                # creo la combinazione e poi la seleziono
                r = random.random()

                if r <= 0.5:
                    # keep streak_val1 e cambio noise

                    candidati = []
                    for s in self.arts_levels:
                        match = re.search(r"streak_(\d+)_noise_([0-9]*\.?[0-9]+)", s)
                        if match:
                            streak = match.group(1)
                            noise = match.group(2)
                            
                            if streak == streak_val1 and noise != noise_val1:
                                candidati.append(s)

                    comb2 = random.choice(candidati)
                    streak_val2, noise_val2 = re.search(r"streak_(\d+)_noise_([0-9]*\.?[0-9]+)", comb2).groups()  # streak_val2 == streak_val1

                    quality_label = 0
                    if float(noise_val2) < float(noise_val1):  # minore è il noise, minore è la qualità
                        quality_label = 1  # x2_hat low quality with respect to x1_hat

                else:
                    # keep noise_val1 e cambio streak
                    candidati = []
                    for s in self.arts_levels:
                        match = re.search(r"streak_(\d+)_noise_([0-9]*\.?[0-9]+)", s)
                        if match:
                            streak = match.group(1)
                            noise = match.group(2)
                            
                            if streak != streak_val1 and noise == noise_val1:
                                candidati.append(s)

                    comb2 = random.choice(candidati)
                    streak_val2, noise_val2 = re.search(r"streak_(\d+)_noise_([0-9]*\.?[0-9]+)", comb2).groups()  # streak_val2 == streak_val1

                    quality_label = 0
                    if int(streak_val2) > int(streak_val1):  # maggiore è lo streak, minore è la qualità
                        quality_label = 1 # # x2_hat low quality with respect to x1_hat

                ##  ho le combinazioni devo tirare fuori la slice

                # la slice in realtà la cambio (l'originale non va bene - devo prendere una con l'artefatto che non avrà lo stesso indice)
                idx_1 = random.choice(self.artifact_class_to_indices[comb1][label]) 
                img_path_1 = self.artifact_dict[comb1][idx_1]
                
                r = random.random()
                if r <= 0.5:
                    # stesso paziente - 
                    idx_2 = random.choice(self.artifact_class_to_indices[comb2][label])  # una slice a caso dello stesso paziente
                    img_path_2 = self.artifact_dict[comb2][idx_2]

                else:
                    # diverso paziente 
                    neg_label = random.choice([l for l in self.patients if l != label])  # un paziente diverso 

                    idx_2 = random.choice(self.artifact_class_to_indices[comb2][neg_label])  # una slice a caso dello stesso paziente
                    img_path_2 = self.artifact_dict[comb2][idx_2]

                x1_hat = np.load(img_path_1).astype(np.float32) 
                x1_hat = torch.from_numpy(x1_hat).unsqueeze(0).float()

                x2_hat = np.load(img_path_2).astype(np.float32) 
                x2_hat = torch.from_numpy(x2_hat).unsqueeze(0).float()

            else:

                x = torch.from_numpy(x).unsqueeze(0).float()  

                #### x1_hat vs x2_hat sintetico
                art1, level_x1_hat = sample_degradation_(distortion_range)
                x1_hat = build_degradation_(artefatto=art1,livello=level_x1_hat)(x)

                level_x2_hat, quality_label = sample_paired_degradation_(artefatto=art1, level_x1_hat=level_x1_hat, parametri=distortion_range)
                
                ## uso x1 campionata
                x1 = np.load(img_path_1).astype(np.float32) 
                x1 = torch.from_numpy(x1).unsqueeze(0).float()

                x2_hat = build_degradation_(artefatto=art1, livello=level_x2_hat)(x1)  # H, W, 1
               

        elif self.test_case == 'full_syn':  # tolgo artefatti di dicaugment e ne uso altri 
            crop_mode = "same" if random.random() <= 0.5 else "different"
            r = random.random()

            if r <= 0.25:
                x = torch.from_numpy(x).unsqueeze(0).float()      # 1, H, W torch

                #### x vs x_hat (low dose)
                # low_dose_img_path = self.low_dose_images[idx]

                low_dose_img_path = self.low_dose_images[idx_1]  # la seconda slice è low dose 

                x_hat = np.load(low_dose_img_path).astype(np.float32)
                x_hat = torch.from_numpy(x_hat).unsqueeze(0).float()  # 1, H, W

            # elif 0.25 < r <= 0.5:
            #     #### x low dose vs x_hat (synthetic)
            
            elif 0.25 < r <= 0.6:

                #### x vs x_hat
                # crop_mode = "same" if random.random() < 0.8 else "different"

                # la degradazione la faccio su img1
                x1 = np.load(img_path_1).astype(np.float32) 
                x1 = torch.from_numpy(x1).unsqueeze(0).float()

                art, level_x_hat = sample_degradation_(distortion_range)
                x_hat = build_degradation_(artefatto=art, livello=level_x_hat)(x1)

                # art, level_x_hat, p, parametri = sample_degradation()
                # x_hat = build_degradation(artefatto=art, p=p)(x1)  # H, W, 1

                # è già torch quando esce da build_degradation
                # x_hat = torch.from_numpy(x_hat).permute(2,0,1).float().contiguous() # 1, H, W torch
                x = torch.from_numpy(x).unsqueeze(0).float()  

            else:

                x = torch.from_numpy(x).unsqueeze(0).float()  

                #### x1_hat vs x2_hat
                art1, level_x1_hat = sample_degradation_(distortion_range)
                x1_hat = build_degradation_(artefatto=art1,livello=level_x1_hat)(x)
                # x1_hat = torch.from_numpy(x1_hat).permute(2,0,1).float().contiguous() # 1, H, W torch

                level_x2_hat, quality_label = sample_paired_degradation_(artefatto=art1, level_x1_hat=level_x1_hat, parametri=distortion_range)
                
                ## uso x1 campionata
                x1 = np.load(img_path_1).astype(np.float32) 
                x1 = torch.from_numpy(x1).unsqueeze(0).float()

                x2_hat = build_degradation_(artefatto=art1, livello=level_x2_hat)(x1)  # H, W, 1
                # x2_hat = torch.from_numpy(x2_hat).permute(2,0,1).float().contiguous() # 1, H, W torch
            
                # #### x1_hat vs x2_hat
                # art1, level_x1_hat, p1, parametri = sample_degradation()
                # x1_hat = build_degradation(artefatto=art1, p=p1)(x)
                # x1_hat = torch.from_numpy(x1_hat).permute(2,0,1).float().contiguous() # 1, H, W torch

                # p2, quality_label = sample_paired_degradation(artefatto=art1, level_x1_hat=level_x1_hat, 
                #                                             p1=p1, parametri=parametri)
                
                # ## uso x1 campionata
                # x1 = np.load(img_path_1).astype(np.float32) 
                # x2_hat = build_degradation(artefatto=art1, p=p2)(x1)  # H, W, 1
                # x2_hat = torch.from_numpy(x2_hat).permute(2,0,1).float().contiguous() # 1, H, W torch

        elif self.test_case == 'full':
            crop_mode = "same" if random.random() <= 0.5 else "different"
            r = random.random()

            if r <= 0.25:
                x = torch.from_numpy(x).unsqueeze(0).float()      # 1, H, W torch

                #### x vs x_hat (low dose)
                # low_dose_img_path = self.low_dose_images[idx]

                low_dose_img_path = self.low_dose_images[idx_1]  # la seconda slice è low dose 

                x_hat = np.load(low_dose_img_path).astype(np.float32)
                x_hat = torch.from_numpy(x_hat).unsqueeze(0).float()  # 1, H, W

            # elif 0.25 < r <= 0.5:
            #     #### x low dose vs x_hat (synthetic)
            
            elif 0.25 < r <= 0.6:

                #### x vs x_hat
                # crop_mode = "same" if random.random() < 0.8 else "different"

                # la degradazione la faccio su img1
                x1 = np.load(img_path_1).astype(np.float32) 
                art, level_x_hat, p, parametri = sample_degradation()
                x_hat = build_degradation(artefatto=art, p=p)(x1)  # H, W, 1

                x_hat = torch.from_numpy(x_hat).permute(2,0,1).float().contiguous() # 1, H, W torch
                x = torch.from_numpy(x).unsqueeze(0).float()  

            else:
            
                #### x1_hat vs x2_hat
                art1, level_x1_hat, p1, parametri = sample_degradation()
                x1_hat = build_degradation(artefatto=art1, p=p1)(x)
                x1_hat = torch.from_numpy(x1_hat).permute(2,0,1).float().contiguous() # 1, H, W torch

                p2, quality_label = sample_paired_degradation(artefatto=art1, level_x1_hat=level_x1_hat, 
                                                            p1=p1, parametri=parametri)
                
                ## uso x1 campionata
                x1 = np.load(img_path_1).astype(np.float32) 
                x2_hat = build_degradation(artefatto=art1, p=p2)(x1)  # H, W, 1
                x2_hat = torch.from_numpy(x2_hat).permute(2,0,1).float().contiguous() # 1, H, W torch

        elif self.test_case == 'all':
            # print("Pretraining with synthetic, low dose and same artifact, different levels combinations, crop mode possibly different")
            
            r = random.random()

            if r <= 0.25:
                x = torch.from_numpy(x).unsqueeze(0).float()      # 1, H, W torch

                #### x vs x_hat (low dose)
                # low_dose_img_path = self.low_dose_images[idx]

                low_dose_img_path = self.low_dose_images[idx_1]  # la seconda slice è low dose 

                x_hat = np.load(low_dose_img_path).astype(np.float32)
                x_hat = torch.from_numpy(x_hat).unsqueeze(0).float()  # 1, H, W
            
            elif 0.25 < r <= 0.6:

                #### x vs x_hat
                crop_mode = "same" if random.random() < 0.8 else "different"

                # la degradazione la faccio su img1
                x1 = np.load(img_path_1).astype(np.float32) 
                art, level_x_hat, p, parametri = sample_degradation()
                x_hat = build_degradation(artefatto=art, p=p)(x1)  # H, W, 1

                x_hat = torch.from_numpy(x_hat).permute(2,0,1).float().contiguous() # 1, H, W torch
                x = torch.from_numpy(x).unsqueeze(0).float()  

                # art, level_x_hat, p, parametri = sample_degradation()
                # x_hat = build_degradation(artefatto=art, p=p)(x)  # H, W, 1

                # x_hat = torch.from_numpy(x_hat).permute(2,0,1).float().contiguous() # 1, H, W torch
                # x = torch.from_numpy(x).unsqueeze(0).float()  

            else:
            
                #### x1_hat vs x2_hat
                art1, level_x1_hat, p1, parametri = sample_degradation()
                x1_hat = build_degradation(artefatto=art1, p=p1)(x)
                x1_hat = torch.from_numpy(x1_hat).permute(2,0,1).float().contiguous() # 1, H, W torch

                p2, quality_label = sample_paired_degradation(artefatto=art1, level_x1_hat=level_x1_hat, 
                                                            p1=p1, parametri=parametri)
                
                ## uso x1 
                x1 = np.load(img_path_1).astype(np.float32) 
                x2_hat = build_degradation(artefatto=art1, p=p2)(x1)  # H, W, 1
                x2_hat = torch.from_numpy(x2_hat).permute(2,0,1).float().contiguous() # 1, H, W torch


                # x2_hat = build_degradation(artefatto=art1, p=p2)(x)  # H, W, 1
                # x2_hat = torch.from_numpy(x2_hat).permute(2,0,1).float().contiguous() # 1, H, W torch

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

        if quality_label is not None:
            if quality_label == 0:
                # x2_hat > x1_hat 
                x = x2_hat
                x_hat = x1_hat
            else:
                # x2_hat < x1_hat 
                x = x1_hat
                x_hat = x2_hat

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

        x_crop = ensure_224(x_crop)
        x_hat_crop = ensure_224(x_hat_crop)

        y = random.randint(0, 1)
        if y == 0:
            return x_crop, x_hat_crop, y
        else:
            return x_hat_crop, x_crop, y




if __name__ == '__main__':

    # mode = 'test'
    # root_path=f"/Prove/Albisani/TCIA_datasets/{mode}_cropped_npy"

    # fd_files = glob.glob(f"{root_path}/*_fd.npy")

    # # self.class_to_indices = defaultdict(list)
    # # for idx, (_, label) in enumerate(samples):
    # #     self.class_to_indices[label].append(idx)

    # # self.labels = list(self.class_to_indices.keys())

    # pairs = []
    # class_to_indices = defaultdict(list)
    # for i, fd in enumerate(sorted(fd_files)):
    #     ld = fd.replace("_fd.npy", "_ld.npy")
    #     if os.path.exists(ld):
    #         pairs.append((fd, ld))
    #     else:
    #         print(f"⚠️ Missing LD for {fd}")
        
    #     patient_id = fd.split('/')[-1][:4]
    #     class_to_indices[patient_id].append(i)  # per ogni paziente ho una lista di id delle slices a lui corrispondenti 

    # patients = list(class_to_indices.keys()) # id pazienti 

    # pairs.sort()  # stable order

    # image_files = [p[0] for p in pairs]
    # low_dose_images = [p[1] for p in pairs]

    # ####
    # # anchor_path, label = self.samples[index]  # img_path = self.image_files[idx]

    # idx = 10
    # anchor_path = image_files[idx]  #
    # label = anchor_path.split('/')[-1][:4]
    # print("path: ", anchor_path)

    # # positive
    # pos_idx = random.choice(class_to_indices[label])  # una slice a caso dello stesso paziente
    # pos_path = image_files[pos_idx]
    # print("Positive path: ", pos_path)

    # # negative
    # neg_label = random.choice([l for l in patients if l != label])  # un paziente diverso 
    # neg_idx = random.choice(class_to_indices[neg_label])  # una slice a caso del paziente diverso 
    # neg_path = image_files[neg_idx]
    # print("Negative path: ", neg_path)

    mode = 'train'
    root_path=f"/Prove/Albisani/TCIA_datasets/{mode}_cropped_npy"

    fd_files = glob.glob(f"{root_path}/*_ld.npy")

    bad_images = []
    for f in fd_files:
        x = np.load(f).astype(np.float32)
        if x.shape[0] < 224 or x.shape[1] < 224:
            print(f, x.shape)
            bad_images.append(x)

    pass