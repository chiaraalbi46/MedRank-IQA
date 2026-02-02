""" This code aims to refactoring the finetuning code (CodiceFase2Shallow_SavePredictionsDict_Metriche_json.py), also introducing a different way for evaluating
results based on multiple crops. """

from comet_ml import Experiment, OfflineExperiment

from argparse import ArgumentParser
from torch.optim.lr_scheduler import StepLR
import torch
import numpy as np
import random
import os
import json
import pandas as pd
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from monai.transforms import LoadImage, Lambda, Compose, ToTensor
from monai.data.image_reader import PydicomReader
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau, pearsonr

from pretraining import TARGET_HW, random_patch_selection, SiameseModel
from utils import build_path_score_dict  # TODO: with the new csv files this function maybe should be avoided...


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
       Salva il modello all'epoca con la migliore validation loss."""

    def __init__(self, patience=4, verbose=False, delta=1e-4, path='best_model.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 4
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 1e-4
            path (str): Path for the checkpoint to be saved to.
                        Default: 'best_model.pth'
            trace_func (function): trace print function.
                                   Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_epoch = None

    def __call__(self, val_loss: float, model: torch.nn.Module, epoch_1based: int):
        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.best_epoch = epoch_1based
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self.best_epoch = epoch_1based
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.path} ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


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


def remaining_indices(train_map_csv: str, used_indices: list[int]) -> list[int]:
    df = pd.read_csv(train_map_csv, index_col=0)
    all_idx = set(df.index.tolist())
    used = set(int(i) for i in used_indices)
    remaining = sorted(list(all_idx - used))
    return remaining


def build_dict_from_indices(
    train_map_csv: str,
    images_root: str,
    indices: list[int],
    full_path_score_dict: dict[str, float],
) -> dict[str, float]:
    paths = indices_to_train_paths(train_map_csv, images_root, indices)
    d = subset_path_score_dict_by_paths(full_path_score_dict, paths)
    return d


class FineTuningDictDataset(Dataset):
    def __init__(self, path_score_dict, n=None, return_path=False, mode='train'):
        self.items = list(path_score_dict.items())
        if n is not None:
            self.items = random.sample(self.items, n)

        self.loader_tif = LoadImage(image_only=True, reader="PILReader")
        self.loader_dcm = LoadImage(image_only=True, reader=PydicomReader)

        self.return_path = return_path

        # filepath to recover otsu crops
        self.otsu_crop_file = f'/Prove/Albisani/LDCTIQA_dataset/otsu_crops_finetuning_{mode}.json'
        if not os.path.exists(self.otsu_crop_file):
            fallback = '/Prove/Albisani/LDCTIQA_dataset/otsu_crops_finetuning_train.json'
            print(f"[FineTuningDictDataset] Warning: {self.otsu_crop_file} non trovato. Uso fallback: {fallback}")
            self.otsu_crop_file = fallback

        with open(self.otsu_crop_file) as f:
            self.otsu_crops = json.load(f)

        def _load(p: str):
            plower = p.lower()
            if plower.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")):
                x = self.loader_tif(p)
            elif plower.endswith(".dcm"):
                x = self.loader_dcm(p)
            else:
                # fallback: autodetect
                x = LoadImage(image_only=True)(p)

            return x

        self.load = Lambda(func=_load)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, score = self.items[idx]

        image = self.load(img_path)

        # retrieve corresponding otsu crop
        minr, minc, maxr, maxc = self.otsu_crops[img_path]
        image = image[minr:maxr, minc:maxc]

        image = image.unsqueeze(0)  # 1, H, W

        # random patch crop of TARGET_HW
        h, w = image.shape[1], image.shape[2]
        top, left = random_patch_selection(h, w, crop_size=TARGET_HW)

        image_crop = image[:, top:top + TARGET_HW, left:left + TARGET_HW]
        image_crop = image_crop.as_tensor()  # metatensor monai --> pytorch tensor (non credo sia fondamentale, però per uniformità...)

        label = torch.tensor(score, dtype=torch.float32)

        if self.return_path:
            return image_crop, label, img_path
        return image_crop, label


class Step2Model(nn.Module):
    def __init__(self, siamese_model):
        super().__init__()
        self.resnet = siamese_model.resnet
        # self.head = nn.Linear(256, 1)
        self.head = nn.Sequential(
            nn.Linear(256, 64),  
            nn.ReLU(),            
            nn.Linear(64, 1)    
        )

    def forward(self, x):
        z = self.resnet(x)
        # y_pred = torch.sigmoid(self.head(z)) * 4  # gli score sono tra 0 e 4
        y_pred = self.head(z)  # no activation
        return y_pred


def evaluate_loss(model, dataloader, device, criterion) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y.unsqueeze(1))
            losses.append(loss.item())
    return float(np.mean(losses)) if len(losses) else float("inf")

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

    parser = ArgumentParser()
    parser.add_argument('--n_finetuning', dest="n_finetuning", type=int, nargs='?', default=None,
                        help="number of images for training (from scratch or finetuning). if None use all images of the train set.")
    # serve per usare un minor numero di immagini del dataset di finetuning

    parser.add_argument('sampling_json', nargs='?', default='./sampling_dict_12_48_240_612_seed42.json',
                        help="path to sampling indices json file. sampling_dict_12_48_240_612_*.json")

    parser.add_argument('--file_path', dest="file_path", nargs='?', default=None,
                        help="path to the pretrained model (phase 1). if None the model is trained from scratch.")
    # se nella riga di comando sul terminale non ci metto niente, lui non carica niente cioè non carica il modello con i pesi salvati
    # se ci scrivo il nome del file con i pesi salvati invece li carica

    # comet parameters
    parser.add_argument("--comet", dest="comet", default=1, help="1 for comet ON, 0 for comet OFF")
    parser.add_argument("--name_proj", dest="name_proj", default='medrank-iqa-finetuning', help="define comet ml project folder")
    # parser.add_argument("--name_exp", dest="name_exp", default='soft_tissue_window', help="name of comet ml experiment")

    parser.add_argument("--batch_size", dest="batch_size", default=32, help="batch size for train and test")
    parser.add_argument('--n_epochs', dest="n_epochs", type=int, default=100, help='number of epochs for training')

    parser.add_argument('--dest_folder', dest="dest_folder", default=None,
                        help='name (not entire path) of the folder where results for the test are stored. if None, the dest folder name is created from inputs parameters (file_path, n_finetuning).')

    parser.add_argument('--device_id', dest="device_id", default='0', help='gpu device id.')
    parser.add_argument('--finetune-backbone-lr-multiplier', default=None, type=float, help='learning rate multiplier used to finetune the backbone')
    parser.add_argument('--finetune-backbone-freeze-epochs', default=0, type=float, help='epochs to keep the backbone frozen when finetuning (only used if learning rate multiplier is specified)')

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

    n_epochs = int(args.n_epochs)

    base_finetuned_folder = './RESULTS'  # where test subfolders are stored

    train_images_root = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAG2023_train/image"
    train_json = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAG2023_train/train.json"
    train_map_csv_default = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAG2023_train/train_map.csv"

    test_images_root = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAC2023_test"
    test_json = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAC2023_test/test-ground-truth.json"

    train_dict_full = build_path_score_dict(train_json, train_images_root, is_test=False)  # all train images
    # train_dict = build_path_score_dict(train_json, train_images_root, is_test=False)
    test_dict = build_path_score_dict(test_json, test_images_root, is_test=True)

    # Seleziona sottogruppo di N immagini per fare finetuning usando indici json
    train_images = 'all'  # if None all available train images are used
    val_dict = None

    if args.n_finetuning is not None:
        train_images = str(args.n_finetuning)
        if args.sampling_json is None:
            raise ValueError("Hai passato n_finetuning ma non --sampling_json. Serve per caricare gli indici.")

        sampling_dict = load_sampling_dict(args.sampling_json)
        used_train_indices = cumulative_indices_for_total(sampling_dict, args.n_finetuning)

        train_dict = build_dict_from_indices(
            train_map_csv=train_map_csv_default,
            images_root=train_images_root,
            indices=used_train_indices,
            full_path_score_dict=train_dict_full
        )
        print(f"Fine-tuning subset selezionato (TRAIN): {len(train_dict)}/{len(train_dict_full)} immagini (target={args.n_finetuning})")

        # Validation = tutte le immagini del train non campionate nel subset
        val_indices = remaining_indices(train_map_csv_default, used_train_indices)
        val_dict = build_dict_from_indices(
            train_map_csv=train_map_csv_default,
            images_root=train_images_root,
            indices=val_indices,
            full_path_score_dict=train_dict_full
        )
        print(f"Validation (complemento): {len(val_dict)} immagini (non campionate)")
    else:
        train_dict = train_dict_full
        print(f"Fine-tuning su tutto il train set: {len(train_dict)} immagini")

    train_dataset_finetuning = FineTuningDictDataset(train_dict, n=args.n_finetuning, return_path=False, mode='train')
    train_dataloader_finetuning = DataLoader(dataset=train_dataset_finetuning, batch_size=int(args.batch_size), shuffle=True)

    val_dataloader_finetuning = None
    if val_dict is not None and len(val_dict) > 0:
        val_dataset_finetuning = FineTuningDictDataset(val_dict, n=None, return_path=False, mode='val')
        val_dataloader_finetuning = DataLoader(dataset=val_dataset_finetuning, batch_size=int(args.batch_size), shuffle=False)

    test_dataset_finetuning = FineTuningDictDataset(test_dict, return_path=True, mode='test')
    test_dataloader_finetuning = DataLoader(dataset=test_dataset_finetuning, batch_size=int(args.batch_size), shuffle=False)

    # Caricamento o meno dei pesi preaddestrati della fase 1
    model = SiameseModel()
    if args.file_path is not None:
        print("Loading weights")
        model.load_state_dict(torch.load(args.file_path, weights_only=True))

    # Step 2 fine-tuning
    model = Step2Model(model)
    model = model.to(device)
    if args.file_path is not None:
        for p in model.resnet.parameters():
            p.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    dest_folder_name = f'from_scratch_{train_images}'
    learning_rate = 1e-5
    if args.file_path is not None:
        optim = Adam(model.head.parameters(), lr=learning_rate, weight_decay=1e-4)
        if args.finetune_backbone_lr_multiplier is not None:
            optim_backbone = Adam(model.resnet.parameters(), lr=learning_rate * args.finetune_backbone_lr_multiplier, weight_decay=1e-4)
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
        dest_folder = os.path.join(base_finetuned_folder, args.dest_folder)  # use the name passed as args.dest_folder
        name_exp = args.dest_folder

    os.makedirs(dest_folder, exist_ok=True)
    print("Destination folder: ", dest_folder)

    # scheduler = StepLR(optim, step_size=50, gamma=0.5)
    scheduler = StepLR(optim, step_size=max(1, n_epochs // 3), gamma=0.5)
    if optim_backbone is not None:
        scheduler_backbone = StepLR(optim_backbone, step_size=n_epochs//3, gamma=0.5)
    else:
        scheduler_backbone = None
    # scheduler = CosineAnnealingLR(optim, T_max=n_epochs)

    # COMET
    experiment = None
    if int(args.comet) == 0:
        # Comet ml integration
        experiment = OfflineExperiment(
            offline_directory=base_finetuned_folder + '/COMET_OFFLINE',
            project_name=args.name_proj
        )
    else:
        experiment = Experiment(project_name=args.name_proj)
        # if .comet.config is not correctly loaded pass explicitly your COMET_API_KEY
        # experiment = Experiment(project_name=args.name_proj, api_key='Fwcd8Z62iWdyhdkt7y0gYSVQw')

    experiment.set_name(name_exp)  # comet experiment has the same name of the destination folder
    ek = experiment.get_key()

    optim_backbone_freeze = args.finetune_backbone_freeze_epochs

    early_stopping_enabled = (val_dataloader_finetuning is not None)
    best_path = os.path.join(dest_folder, "best_model.pth")
    best_epoch_file = os.path.join(dest_folder, "best_epoch.txt")

    early_stopper = None
    if early_stopping_enabled:
        early_stopper = EarlyStopping(
            patience=5, #10, 20
            verbose=True,
            delta=1e-4,
            path=best_path,
            trace_func=print
        )

    for epoch in tqdm(range(n_epochs)):
        model.train()
        train_losses = []
        with tqdm(train_dataloader_finetuning, leave=False, desc="Training") as t:
            for x, y in t:
                x = x.to(device)
                y = y.to(device)
                optim.zero_grad()
                if optim_backbone is not None and epoch >= optim_backbone_freeze:
                    optim_backbone.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y.unsqueeze(1))
                t.set_postfix(loss=loss.item())
                loss.backward()
                optim.step()
                if optim_backbone is not None and epoch >= optim_backbone_freeze:
                    optim_backbone.step()

                train_losses.append(loss.item())
                # writer.add_scalar("train/loss", loss.item(), iteration)
                # iteration += 1

        train_loss_mean = np.mean(train_losses)
        experiment.log_metric("train_loss", train_loss_mean, step=epoch)

        if early_stopping_enabled:
            val_loss_mean = evaluate_loss(model, val_dataloader_finetuning, device, criterion)
            experiment.log_metric("val_loss", val_loss_mean, step=epoch)

            print(
                f"Epoch {epoch+1} - Train Loss: {train_loss_mean:.6f} | "
                f"Val Loss: {val_loss_mean:.6f}"
            )

            early_stopper(val_loss_mean, model, epoch_1based=epoch + 1)

            # salva epoca con miglior validation loss
            if early_stopper.best_epoch is not None:
                with open(best_epoch_file, "w", encoding="utf-8") as f:
                    f.write(str(early_stopper.best_epoch))

            if early_stopper.early_stop:
                print(
                    f"Early stopping at epoch {epoch+1}. "
                    f"Best model saved at epoch {early_stopper.best_epoch} to: {best_path} "
                    f"(best_val_loss={early_stopper.best_val_loss:.6f})"
                )
                break

        #test ogni epoca
        model.eval()
        test_losses = []

        all_predictions = []
        all_targets = []
        all_names = []  # per allineamento pred/gt con l'immagine

        with tqdm(test_dataloader_finetuning, leave=False, desc="Test") as t:
            for batch in t:
                x, y, paths = batch
                x = x.to(device)
                y = y.to(device)

                with torch.no_grad():
                    y_pred = model(x)

                loss = criterion(y_pred, y.unsqueeze(1))
                t.set_postfix(loss=loss.item())
                test_losses.append(loss.item())

                # Salva pred/gt nello stesso ordine
                y_pred = torch.clamp(y_pred, 0, 4)
                all_predictions.append(y_pred.squeeze(1).cpu().numpy())
                all_targets.append(y.cpu().numpy())

                # Salva i nomi immagine (o path) puliti
                all_names.extend([os.path.basename(p) for p in paths])

        # Concatenazione batch
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

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

        experiment.log_metric("spearman", float(sp_corr), step=epoch)
        experiment.log_metric("kendall", float(kd_corr), step=epoch)
        experiment.log_metric("pearson", float(pr_corr), step=epoch)

        scheduler.step()  # should be every epoch
        if scheduler_backbone is not None and epoch >= optim_backbone_freeze:
            scheduler_backbone.step()  # should be every epoch

    # test finale sul modello con la migliore validation test
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    model.eval()
    test_losses = []
    all_predictions = []
    all_targets = []
    all_names = []

    with tqdm(test_dataloader_finetuning, leave=False, desc="Final Test") as t:
        for batch in t:
            x, y, paths = batch
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                y_pred = model(x)

            loss = criterion(y_pred, y.unsqueeze(1))
            t.set_postfix(loss=loss.item())
            test_losses.append(loss.item())

            y_pred = torch.clamp(y_pred, 0, 4)
            all_predictions.append(y_pred.squeeze(1).cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_names.extend([os.path.basename(p) for p in paths])

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    test_loss_mean = np.mean(test_losses)
    sp_corr, sp_p = spearmanr(all_predictions, all_targets)
    kd_corr, kd_p = kendalltau(all_predictions, all_targets)
    pr_corr, pr_p = pearsonr(all_predictions, all_targets)

    print(
        f"FINAL - Test Loss: {test_loss_mean:.6f} | "
        f"Spearman r={sp_corr:.6f} (p={sp_p:.2e}) | "
        f"Kendall τ={kd_corr:.6f} (p={kd_p:.2e}) | "
        f"Pearson r={pr_corr:.6f} (p={pr_p:.2e})"
    )

    pred_dict = {name: float(pred) for name, pred in zip(all_names, all_predictions)}
    with open(os.path.join(dest_folder, "predictions_finetuning_dict.json"), "w", encoding="utf-8") as f:
        json.dump(pred_dict, f, ensure_ascii=False, indent=2)

    metrics = {
        "test_loss": float(test_loss_mean),
        "spearman_r": float(sp_corr), "spearman_p": float(sp_p),
        "kendall_tau": float(kd_corr), "kendall_p": float(kd_p),
        "pearson_r": float(pr_corr), "pearson_p": float(pr_p),
        "best_val_loss": float(early_stopper.best_val_loss) if early_stopper is not None else None,
        "best_epoch": int(early_stopper.best_epoch) if early_stopper is not None else None
    }
    with open(os.path.join(dest_folder, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)