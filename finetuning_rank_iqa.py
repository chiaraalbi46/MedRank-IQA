""" This code reproduces finetuning with vgg16 as in Rank-IQA paper """

from comet_ml import Experiment, OfflineExperiment
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
import torch
from torch import nn
import numpy as np 
import os 
from tqdm.auto import tqdm
import pandas as pd
import json
from monai.transforms import LoadImage, Lambda
from monai.data.image_reader import PydicomReader
import random
from random import randrange
from torch.optim.lr_scheduler import StepLR
from scipy.stats import spearmanr, kendalltau, pearsonr
from pretraining_rank_iqa import RankIQA_branch, Vgg16

from utils import build_path_score_dict # TODO: with the new csv files this function maybe should be avoided...
 
TARGET_HW = 224

def random_patch_selection(h, w, crop_size=TARGET_HW):

    top = randrange(0, max(1, h - crop_size))
    left = randrange(0, max(1, w - crop_size))

    return top, left

# Sampling JSON helpers 
def load_sampling_dict(sampling_json_path: str) -> dict[int, list[int]]:
    with open(sampling_json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {int(k): [int(x) for x in v] for k, v in d.items()}


def cumulative_indices_for_total(sampling_dict: dict[int, list[int]], total: int) -> list[int]:
    blocks = sorted(sampling_dict.keys())  # e.g. [12,48,240,612]
    out = []
    s = 0
    for b in blocks:
        out.extend(sampling_dict[b])
        s += b
        if s == total:
            return out
        if s > total:
            raise ValueError(
                f"total={total} non è raggiungibile con somme progressive dei blocchi {blocks}. "
                f"Somma superata a {s}."
            )
    raise ValueError(f"total={total} troppo grande. Massimo raggiungibile={s} con blocchi {blocks}.")


def guess_name_column(df: pd.DataFrame) -> str | None:
    preferred = ["img_name", "image", "fname", "filename", "name", "id"]
    cols = list(df.columns)

    for c in preferred:
        if c in cols:
            return c

    for c in cols:
        if df[c].dtype == object:
            sample = df[c].dropna().astype(str).head(20).tolist()
            if any((".tif" in s.lower() or ".tiff" in s.lower() or ".png" in s.lower() or ".jpg" in s.lower())
                   for s in sample):
                return c
            if all(len(s) > 0 for s in sample):
                return c
    return None


def indices_to_train_paths(train_map_csv: str, images_root: str, selected_indices: list[int]) -> list[str]:
    df = pd.read_csv(train_map_csv, index_col=0)
    name_col = guess_name_column(df)

    paths = []
    for i in selected_indices:
        if i not in df.index:
            raise KeyError(f"Indice {i} non presente nel train_map.csv (index_col=0).")

        if name_col is None:
            stem = str(df.loc[i].name)
        else:
            stem = str(df.loc[i, name_col])
        stem = stem.strip()

        lower = stem.lower()
        if lower.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")):
            rel = stem
        else:
            rel = stem + ".tif"

        img_path = os.path.join(images_root, rel)
        paths.append(img_path)

    return paths

def subset_path_score_dict_by_paths(path_score_dict: dict[str, float], keep_paths: list[str]) -> dict[str, float]:
    out = {}
    missing = 0
    for p in keep_paths:
        if p in path_score_dict:
            out[p] = path_score_dict[p]
        else:
            missing += 1
    if missing > 0:
        print(f"Warning: {missing} path selezionati non trovati in path_score_dict (controlla root/estensione).")
    return out


class FineTuningDictDataset(Dataset):
    def __init__(self, path_score_dict, n = None, return_path=False, mode='train'):
        self.items = list(path_score_dict.items())
        if n is not None:
            self.items = random.sample(self.items, n)

        self.loader_tif = LoadImage(image_only=True, reader="PILReader")
        self.loader_dcm = LoadImage(image_only=True, reader=PydicomReader)

        self.return_path = return_path

        # filepath to recover otsu crops 
        self.otsu_crop_file = f'/Prove/Albisani/LDCTIQA_dataset/otsu_crops_finetuning_{mode}.json'
        with open(self.otsu_crop_file) as f:
            self.otsu_crops = json.load(f)

        def _load_to_cd_hw(p: str) -> np.ndarray:
            plower = p.lower()
            if plower.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")):
                x = self.loader_tif(p)
            elif plower.endswith(".dcm"):
                x = self.loader_dcm(p)
            else:
                # fallback: autodetect
                x = LoadImage(image_only=True)(p)

            return x

        self.load = Lambda(func=_load_to_cd_hw)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, score = self.items[idx]

        image = self.load(img_path) 

        ## retrieve corresponding otsu crop
        minr, minc, maxr, maxc = self.otsu_crops[img_path]
        image = image[minr:maxr, minc:maxc]

        image = image.unsqueeze(0)  # 1, H, W

        ## random patch crop of TARGET_HW
        h, w = image.shape[1], image.shape[2]
        top, left = random_patch_selection(h, w, crop_size=TARGET_HW)

        image_crop = image[:, top:top+TARGET_HW, left:left+TARGET_HW]
        image_crop = image_crop.as_tensor()  # metatensor monai --> pytorch tensor (non credo sia fondamentale, però per uniformità...)

        label = torch.tensor(score, dtype=torch.float32)

        if self.return_path:
            return image_crop, label, img_path
        else:
            return image_crop, label

if __name__ == '__main__':

    ###### NB: per ogni test viene creata una cartella con questa struttura a meno che non si passi --dest_folder:
    # testtype_pretrainingmodel_#img 
    # dovre testtype: finetuning / from_scratch (gli equivalenti di sì_pretraining e no_pretraining)
    # pretrainingmodel: nome del file dei pesi
    # #img: n_finetuning se diverso da None, altrimenti è 'all'

    ### command examples
    # python finetuning.py --n_finetuning 12 --file_path './pretrained_models/soft_tissue_window.pth' --> i risultati sono salvati in ./RESULTS/finetuning_soft_tissue_window_12 [finetuning (sì pretraining) 12 img]
    # python finetuning.py --n_finetuning 12  --> i risultati sono salvati in ./RESULTS/from_scratch_soft_tissue_window_12 [from scratch (no pretraining) 12 img]  
    # python finetuning.py --file_path './pretrained_models/soft_tissue_window.pth' --> i risultati sono salvati in ./RESULTS/finetuning_soft_tissue_window_all [finetuning (sì pretraining) all train images]
        # python finetuning.py --n_finetuning 12 --dest_folder 'prova'  --> i risultati sono salvati in ./RESULTS/prova [finetuning 12 img ma con nome cartella 'libero']
    ###
    
    parser = ArgumentParser()
    parser.add_argument('--n_finetuning', dest="n_finetuning", type=int, nargs='?', default=None,
                        help="number of images for training (from scratch or finetuning). if None use all images of the train set.") 
    #serve per usare un minor numero di immagini del dataset di finetuning
    parser.add_argument('sampling_json', nargs='?', default='./sampling_dict_12_48_240_612_seed42.json',
                    help="path to sampling indices json file. sampling_dict_12_48_240_612_*.json")
    
    parser.add_argument('--file_path', dest="file_path", nargs='?', default='./Pytorch_TestRankIQA_pretrained/Rank_tid2013.caffemodel.pt',
                        help="path to the pretrained model (phase 1). if None the model is trained from scratch.") #se nella riga di comando sul terminale non ci metto niente, lui non 
    #carica niente cioè non carica il modello con i pesi salvati, se ci scrivo il nome del file con i pesi salvati invece li carica 
    
    # comet parameters
    parser.add_argument("--comet", dest="comet", default=1, help="1 for comet ON, 0 for comet OFF")
    parser.add_argument("--name_proj", dest="name_proj", default='medrank-iqa-finetuning', help="define comet ml project folder")
    # parser.add_argument("--name_exp", dest="name_exp", default='soft_tissue_window', help="name of comet ml experiment")

    parser.add_argument("--batch_size", dest="batch_size", default=32, help="batch size for train and test")
    parser.add_argument('--n_epochs', dest="n_epochs",type=int, default=100, help='number of epochs for training')

    parser.add_argument('--dest_folder', dest="dest_folder", default=None,
                        help='name (not entire path) of the folder where results for the test are stored. if None, the dest folder name is created from inputs parameters (file_path, n_finetuning).')  
    
    parser.add_argument('--device_id', dest="device_id",  default='0', help='gpu device id.')
    parser.add_argument('--finetune-backbone-lr-multiplier', default=None, type=float, help='learning rate multiplier used to finetune the backbone')
    parser.add_argument('--finetune-backbone-freeze-epochs', default=0, type=float, help='epochs to keep the backbone frozen when finetuning (only used if learning rate multiplier is specified)')
    
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=1e-5, help="base learning rate")

    parser.add_argument('--imagenet-initialization', dest="imagenet-initialization",  default=None, help='use (1) or not (None) imagenet weights for VGG16')
     # '/data/lesc/staff/albisani/MedRank-IQA/Pytorch_TestRankIQA_pretrained/Rank_tid2013.caffemodel.pt'

    args = parser.parse_args()

    # device = "cuda:0"
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

    n_epochs = int(args.n_epochs)

    base_finetuned_folder = './RESULTS_VGG_PRETRAINED'  ### where test subfolders are stored
    os.makedirs(base_finetuned_folder, exist_ok=True)

    train_images_root = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAG2023_train/image"
    train_json        = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAG2023_train/train.json"
    train_map_csv_default = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAG2023_train/train_map.csv"

    test_images_root  = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAC2023_test"
    test_json         = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAC2023_test/test-ground-truth.json"

    train_dict_full = build_path_score_dict(train_json, train_images_root, is_test=False)  # all train images
    # train_dict = build_path_score_dict(train_json, train_images_root, is_test=False)
    test_dict  = build_path_score_dict(test_json, test_images_root, is_test=True)

    # Seleziona sottogruppo di N immagini per fare finetuning usando indici json 
    train_images = 'all'  # if None all available train images are used
    if args.n_finetuning is not None:
        train_images = str(args.n_finetuning)
        if args.sampling_json is None:
            raise ValueError("Hai passato n_finetuning ma non --sampling_json. Serve per caricare gli indici.")

        sampling_dict = load_sampling_dict(args.sampling_json)
        selected_indices = cumulative_indices_for_total(sampling_dict, args.n_finetuning)

        selected_paths = indices_to_train_paths(
            train_map_csv=train_map_csv_default,
            images_root=train_images_root,
            selected_indices=selected_indices
        )

        train_dict = subset_path_score_dict_by_paths(train_dict_full, selected_paths)
        print(f"Fine-tuning subset selezionato: {len(train_dict)}/{len(train_dict_full)} immagini (target={args.n_finetuning})")
    else:
        train_dict = train_dict_full
        print(f"Fine-tuning su tutto il train set: {len(train_dict)} immagini")
    
    train_dataset_finetuning = FineTuningDictDataset(train_dict, n=args.n_finetuning, return_path=False, mode='train')
    test_dataset_finetuning  = FineTuningDictDataset(test_dict, return_path=True, mode='test')

    train_dataloader_finetuning = DataLoader(dataset=train_dataset_finetuning, batch_size=int(args.batch_size), shuffle=True)
    test_dataloader_finetuning  = DataLoader(dataset=test_dataset_finetuning,  batch_size=int(args.batch_size))

    # # Caricamento o meno dei pesi preaddestrati della fase 1
    # model = SiameseModel()
    # if args.file_path is not None:
    #     print("Loading weights")
    #     model.load_state_dict(torch.load(args.file_path, weights_only=True))

    # # Step 2 fine-tuning 
    # model = Step2Model(model)
    # model = model.to(device)
    # if args.file_path is not None:
    #     for p in model.resnet.parameters():
    #         p.requires_grad = False

    if args.file_path is not None:  # questo quando carico pretrained su img naturali 
        if 'Pytorch_TestRankIQA_pretrained' in args.file_path:
            print("Load Rank-IQA weights")
            vgg = Vgg16(imagenet=None)
            vgg.load_model(args.file_path, debug=True)

            model = RankIQA_branch(vgg_model=vgg).to(device)

        else:
            print("Load our pretrained weights")
            ## questo quando carico con modello pretrained su ct (siamese)
            vgg = Vgg16(imagenet=args.imagenet_initialization)
            model = RankIQA_branch(vgg_model=vgg).to(device)

            state_dict = torch.load(args.file_path, map_location=device)
            # model.load_state_dict(state_dict, strict=True)
            scorer_state_dict = {
                    k.replace("scorer.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("scorer.")
                }
            model.load_state_dict(scorer_state_dict, strict=True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    dest_folder_name = f'from_scratch_{train_images}'
    learning_rate = float(args.learning_rate)
    if args.file_path is not None:
        optim = Adam(model.head.parameters(), lr=learning_rate, weight_decay=1e-4)
        if args.finetune_backbone_lr_multiplier is not None:
            # optim_backbone = Adam(model.resnet.parameters(), lr=learning_rate * args.finetune_backbone_lr_multiplier, weight_decay=1e-4)
            optim_backbone = Adam(model.features.parameters(), lr=learning_rate * args.finetune_backbone_lr_multiplier, weight_decay=1e-4)
        else:
            optim_backbone = None
        pretrained_model = args.file_path.split('/')[-1].split('.')[0]
        dest_folder_name = f'finetuning_{pretrained_model}_{train_images}'
    else:
        optim = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        optim_backbone = None
        
    criterion = nn.MSELoss()

    dest_folder = os.path.join(base_finetuned_folder, dest_folder_name)
    name_exp = dest_folder_name
    if args.dest_folder is not None:
        dest_folder = os.path.join(base_finetuned_folder, args.dest_folder) # use the name passed as args.dest_folder
        name_exp = args.dest_folder

    os.makedirs(dest_folder, exist_ok=True)
    print("Destination folder: ", dest_folder)

    scheduler = StepLR(optim, step_size=n_epochs//3, gamma=0.5)
    if optim_backbone is not None:
        scheduler_backbone = StepLR(optim_backbone, step_size=n_epochs//3, gamma=0.5)
    else:
        scheduler_backbone = None
    # scheduler = CosineAnnealingLR(optim, T_max=n_epochs)  

    # COMET
    experiment = None
    if int(args.comet) == 0:
        # Comet ml integration 
        experiment = OfflineExperiment(offline_directory=base_finetuned_folder+ '/COMET_OFFLINE',
                                       project_name=args.name_proj)
    else:
        # matplotlib.use('TkAgg')
        experiment = Experiment(project_name=args.name_proj) 
        # if .comet.config is not correctly loaded pass explicitly your COMET_API_KEY
        # experiment = Experiment(project_name=args.name_proj, api_key='YOUR_COMET_API_KEY') 

    experiment.set_name(name_exp)  # comet experiment has the same name of the destination folder
    ek = experiment.get_key()

    optim_backbone_freeze = args.finetune_backbone_freeze_epochs

    for epoch in tqdm(range(n_epochs)):
        model.train()
        train_losses = []
        with tqdm(train_dataloader_finetuning, leave=False, desc="Training") as t:
            for x, y in t:
                x = x.to(device)
                y = y.to(device)

                if x.size(1) == 1:
                    x = x.repeat(1, 3, 1, 1)

                optim.zero_grad()
                if optim_backbone is not None and epoch >= optim_backbone_freeze:
                    optim_backbone.zero_grad()
                y_pred = model(x)
                # y_pred = torch.clamp(y_pred, 0, 4)  # gt scores are between 0 and 4
                loss = criterion(y_pred, y.unsqueeze(1))
                t.set_postfix(loss=loss.item())
                loss.backward()
                optim.step()
                if optim_backbone is not None and epoch >= optim_backbone_freeze:
                    optim_backbone.step()

                train_losses.append(loss.item())
        
        train_loss_mean = np.mean(train_losses)
        experiment.log_metric("train_loss", train_loss_mean, step=epoch)

        if True or (epoch == 0) or ((epoch + 1) % 10 == 0):
            model.eval()
            test_losses = []

            all_predictions = []
            all_targets = []
            all_names = []   # per allineamento pred/gt con l'immagine

            with tqdm(test_dataloader_finetuning, leave=False, desc="Test") as t:
                for batch in t:
                    x, y, paths = batch
                    x = x.to(device)
                    y = y.to(device)

                    if x.size(1) == 1:
                        x = x.repeat(1, 3, 1, 1)

                    with torch.no_grad():
                        y_pred = model(x)

                    loss = criterion(y_pred, y.unsqueeze(1))
                    t.set_postfix(loss=loss.item())
                    test_losses.append(loss.item())

                    # Salva pred/gt nello stesso ordine
                    y_pred = torch.clamp(y_pred, 0, 4)  # gt scores are between 0 and 4
                    all_predictions.append(y_pred.squeeze(1).cpu().numpy())  
                    all_targets.append(y.cpu().numpy())                    

                    # Salva i nomi immagine (o path) puliti
                    all_names.extend([os.path.basename(p) for p in paths])

            # Concatenazione batch
            all_predictions = np.concatenate(all_predictions, axis=0)  
            all_targets     = np.concatenate(all_targets, axis=0)     

            test_loss_mean = np.mean(test_losses)
            # writer.add_scalar("test/loss", test_loss_mean, iteration)
            experiment.log_metric("test_loss", test_loss_mean, step=epoch)

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

            experiment.log_metric("spearman", sp_corr, step=epoch)
            experiment.log_metric("kendall", kd_corr, step=epoch)
            experiment.log_metric("pearson", pr_corr, step=epoch)

            if epoch == (n_epochs - 1):  
                # dizionari pred/gt
                pred_dict = {name: float(pred) for name, pred in zip(all_names, all_predictions)}
                # gt_dict   = {name: float(gt)   for name, gt   in zip(all_names, all_targets)}

                with open(os.path.join(dest_folder, "predictions_finetuning_dict.json"), "w", encoding="utf-8") as f:
                    json.dump(pred_dict, f, ensure_ascii=False, indent=2)

                # metriche
                metrics = {
                    "spearman_r": float(sp_corr), "spearman_p": float(sp_p),
                    "kendall_tau": float(kd_corr), "kendall_p": float(kd_p),
                    "pearson_r": float(pr_corr), "pearson_p": float(pr_p),
                }
                with open(os.path.join(dest_folder, f"correlation_metrics_{epoch}.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)

        scheduler.step()  # should be every epoch
        if scheduler_backbone is not None and epoch >= optim_backbone_freeze:
            scheduler_backbone.step()  # should be every epoch