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
from pretraining_networks import RankIQA_branch, Vgg16, Resnet18RankIQA_branch, Resnet18, SqueezeNet1_1RankIQA_branch, SqueezeNet1_1

from early_stopping import EarlyStopping, evaluate_loss, load_sampling_dict, cumulative_indices_for_total, guess_name_column, indices_to_train_paths, subset_path_score_dict_by_paths, build_dict_from_indices, remaining_indices

from utils import build_path_score_dict # TODO: with the new csv files this function maybe should be avoided...

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Makes CuDNN deterministic (important!)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

base_seed = 42
set_global_seed(base_seed)
 
TARGET_HW = 224

def random_patch_selection(h, w, crop_size=TARGET_HW):

    top = randrange(0, max(1, h - crop_size))
    left = randrange(0, max(1, w - crop_size))

    return top, left


class FineTuningDictDataset(Dataset):
    def __init__(self, path_score_dict, n = None, return_path=False, mode='train'):
        self.items = list(path_score_dict.items())
        if n is not None:
            self.items = random.sample(self.items, n)

        self.loader_tif = LoadImage(image_only=True, reader="PILReader")
        self.loader_dcm = LoadImage(image_only=True, reader=PydicomReader)
        self.loader_fallback = LoadImage(image_only=True)

        self.return_path = return_path

        # filepath to recover otsu crops 
        self.otsu_crop_file = f'/Prove/Albisani/LDCTIQA_dataset/otsu_crops_finetuning_{mode}.json'
        
        ####### per validation
        if not os.path.exists(self.otsu_crop_file):
            fallback = '/Prove/Albisani/LDCTIQA_dataset/otsu_crops_finetuning_train.json'
            print(f"[FineTuningDictDataset] Warning: {self.otsu_crop_file} non trovato. Uso fallback: {fallback}")
            self.otsu_crop_file = fallback
        #######
        
        with open(self.otsu_crop_file) as f:
            self.otsu_crops = json.load(f)

        # def _load_to_cd_hw(p: str) -> np.ndarray:
        #     plower = p.lower()
        #     if plower.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")):
        #         x = self.loader_tif(p)
        #     elif plower.endswith(".dcm"):
        #         x = self.loader_dcm(p)
        #     else:
        #         # fallback: autodetect
        #         x = LoadImage(image_only=True)(p)

        #     return x

        # self.load = Lambda(func=_load_to_cd_hw)

    def _load_image(self, p: str):
        plower = p.lower()
        if plower.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")):
            return self.loader_tif(p)
        elif plower.endswith(".dcm"):
            return self.loader_dcm(p)
        return self.loader_fallback(p)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, score = self.items[idx]

        # image = self.load(img_path) 
        image = self._load_image(img_path)

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
    
    parser.add_argument('--file_path', dest="file_path", nargs='?', default=None,  # './Pytorch_TestRankIQA_pretrained/Rank_tid2013.caffemodel.pt',
                        help="path to the pretrained model (phase 1). if None the model is trained from scratch.") #se nella riga di comando sul terminale non ci metto niente, lui non 
    #carica niente cioè non carica il modello con i pesi salvati, se ci scrivo il nome del file con i pesi salvati invece li carica 
    
    # comet parameters
    parser.add_argument("--comet", dest="comet", default=1, help="1 for comet ON, 0 for comet OFF")
    parser.add_argument("--name_proj", dest="name_proj", default='medrank-iqa-finetuning-NEW', help="define comet ml project folder")
    # parser.add_argument("--name_exp", dest="name_exp", default='soft_tissue_window', help="name of comet ml experiment")

    parser.add_argument("--batch_size", dest="batch_size", default=32, help="batch size for train and test")
    parser.add_argument('--n_epochs', dest="n_epochs",type=int, default=100, help='number of epochs for training')

    parser.add_argument('--dest_folder', dest="dest_folder", default=None,
                        help='name (not entire path) of the folder where results for the test are stored. if None, the dest folder name is created from inputs parameters (file_path, n_finetuning).')  
    
    parser.add_argument('--device_id', dest="device_id",  default='0', help='gpu device id.')
    parser.add_argument('--finetune-backbone-lr-multiplier', default=None, type=float, help='learning rate multiplier used to finetune the backbone')
    parser.add_argument('--finetune-backbone-freeze-epochs', default=0, type=float, help='epochs to keep the backbone frozen when finetuning (only used if learning rate multiplier is specified)')
    
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=1e-4, help="base learning rate")

    parser.add_argument("--patience", dest="patience", type=int, default=10,
                    help="early stopping patience (used only if validation is enabled)")

    parser.add_argument('--imagenet_initialization', dest="imagenet_initialization",  default=None, help='use (1) or not (None) imagenet weights for VGG16')
     # '/data/lesc/staff/albisani/MedRank-IQA/Pytorch_TestRankIQA_pretrained/Rank_tid2013.caffemodel.pt'

    parser.add_argument('--network_model', dest="network_model",  default='vgg16', help='specify the network to use. vgg16, resnet18')

    args = parser.parse_args()

    # device = "cuda:0"
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

    n_epochs = int(args.n_epochs)

    # base_finetuned_folder = './RESULTS_VGG_PRETRAINED'  ### where test subfolders are stored
    # base_finetuned_folder = './RESULTS_VGG_PRETRAINED_NEW'

    if args.network_model == 'vgg16':
        base_finetuned_folder = './RESULTS_VGG'
    elif args.network_model == 'resnet18':
        base_finetuned_folder = './RESULTS_RESNET18'
    else:
        base_finetuned_folder = './RESULTS_SQUEEZENET1_1'
        
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
    val_dict = None
    if args.n_finetuning is not None:
        train_images = str(args.n_finetuning)
        if args.sampling_json is None:
            raise ValueError("Hai passato n_finetuning ma non --sampling_json. Serve per caricare gli indici.")

        sampling_dict = load_sampling_dict(args.sampling_json)

        # selected_indices = cumulative_indices_for_total(sampling_dict, args.n_finetuning)

        # selected_paths = indices_to_train_paths(
        #     train_map_csv=train_map_csv_default,
        #     images_root=train_images_root,
        #     selected_indices=selected_indices
        # )
        # train_dict = subset_path_score_dict_by_paths(train_dict_full, selected_paths)

        used_train_indices = cumulative_indices_for_total(sampling_dict, args.n_finetuning)

        train_dict = build_dict_from_indices(
            train_map_csv=train_map_csv_default,
            images_root=train_images_root,
            indices=used_train_indices,
            full_path_score_dict=train_dict_full
        )

        print(f"Fine-tuning subset selezionato: {len(train_dict)}/{len(train_dict_full)} immagini (target={args.n_finetuning})")

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
    test_dataset_finetuning  = FineTuningDictDataset(test_dict, return_path=True, mode='test')

    train_dataloader_finetuning = DataLoader(dataset=train_dataset_finetuning, batch_size=int(args.batch_size), shuffle=True)
    test_dataloader_finetuning  = DataLoader(dataset=test_dataset_finetuning,  batch_size=int(args.batch_size))

    val_dataloader_finetuning = None
    if val_dict is not None and len(val_dict) > 0:
        val_dataset_finetuning = FineTuningDictDataset(val_dict, n=None, return_path=False, mode='val')
        val_dataloader_finetuning = DataLoader(dataset=val_dataset_finetuning, batch_size=int(args.batch_size), shuffle=False)


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
            print(f"Load our {args.network_model} pretrained weights")
            ## questo quando carico con modello pretrained su ct (siamese)

            if args.network_model == 'vgg16':
                vgg = Vgg16(imagenet=args.imagenet_initialization)
                model = RankIQA_branch(vgg_model=vgg).to(device)
            elif args.network_model == 'resnet18':
                vgg = Resnet18(imagenet=args.imagenet_initialization)
                model = Resnet18RankIQA_branch(resnet18_model=vgg).to(device)
            else:
                vgg = SqueezeNet1_1(imagenet=args.imagenet_initialization)
                model = SqueezeNet1_1RankIQA_branch(squeezenet1_1_model=vgg).to(device)

            state_dict = torch.load(args.file_path, map_location=device)
            # model.load_state_dict(state_dict, strict=True)
            scorer_state_dict = {
                    k.replace("scorer.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("scorer.")
                }
            model.load_state_dict(scorer_state_dict, strict=True)
    
    else:

        if args.network_model == 'vgg16':
            print("Test VGG16 from scratch")
            vgg = Vgg16(imagenet=args.imagenet_initialization)
            model = RankIQA_branch(vgg_model=vgg).to(device)
        elif args.network_model == 'resnet18':
            print("Test Resnet18 from scratch")
            vgg = Resnet18(imagenet=args.imagenet_initialization)
            model = Resnet18RankIQA_branch(resnet18_model=vgg).to(device)
        elif args.network_model == 'squeezenet1_1':
            print("Test SqueezeNet 1.1 from scratch")
            vgg = SqueezeNet1_1(imagenet=args.imagenet_initialization)
            model = SqueezeNet1_1RankIQA_branch(squeezenet1_1_model=vgg).to(device)

    dest_folder_name = f'from_scratch_{train_images}_{args.network_model}'
    learning_rate = float(args.learning_rate)


    # if args.file_path is not None:
    #     optim = Adam(model.head.parameters(), lr=learning_rate, weight_decay=1e-4)
    #     if args.finetune_backbone_lr_multiplier is not None:
    #         # optim_backbone = Adam(model.resnet.parameters(), lr=learning_rate * args.finetune_backbone_lr_multiplier, weight_decay=1e-4)
    #         optim_backbone = Adam(model.features.parameters(), lr=learning_rate * args.finetune_backbone_lr_multiplier, weight_decay=1e-4)
    #     else:
    #         optim_backbone = None
    #     pretrained_model = args.file_path.split('/')[-1].split('.')[0]
    #     dest_folder_name = f'finetuning_{pretrained_model}_{train_images}'
        
    # else:
    #     optim = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    #     # optim_backbone = None
    #     if args.finetune_backbone_lr_multiplier is not None:
    #         # TODO: problema , doppi gradienti (features sia in optim che in optim backbone)
    #         # optim_backbone = Adam(model.resnet.parameters(), lr=learning_rate * args.finetune_backbone_lr_multiplier, weight_decay=1e-4)
    #         optim_backbone = Adam(model.features.parameters(), lr=learning_rate * args.finetune_backbone_lr_multiplier, weight_decay=1e-4)
    #     else:
    #         optim_backbone = None

    ## one optimizer
    param_groups = []
    if args.file_path is not None:
        param_groups.append({"params": model.head.parameters(), "lr": learning_rate})
        if args.finetune_backbone_lr_multiplier is not None:
            param_groups.append({"params": model.features.parameters(), "lr": learning_rate * args.finetune_backbone_lr_multiplier})
        pretrained_model = args.file_path.split('/')[-1].split('.')[0]
        dest_folder_name = f'finetuning_{pretrained_model}_{train_images}'
    else:
        if args.finetune_backbone_lr_multiplier is not None:
            param_groups.append({"params": model.features.parameters(), "lr": learning_rate * args.finetune_backbone_lr_multiplier})
            param_groups.append({"params": model.head.parameters(), "lr": learning_rate})
        else:
            param_groups.append({"params": model.parameters(), "lr": learning_rate})

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    optim = Adam(param_groups, weight_decay=1e-4)
        
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()

    dest_folder = os.path.join(base_finetuned_folder, dest_folder_name)
    name_exp = dest_folder_name
    if args.dest_folder is not None:
        dest_folder = os.path.join(base_finetuned_folder, args.dest_folder) # use the name passed as args.dest_folder
        name_exp = args.dest_folder

    os.makedirs(dest_folder, exist_ok=True)
    print("Destination folder: ", dest_folder)

    scheduler = StepLR(optim, step_size=n_epochs//3, gamma=0.5)
    # if optim_backbone is not None:
    #     scheduler_backbone = StepLR(optim_backbone, step_size=n_epochs//3, gamma=0.5)
    # else:
    #     scheduler_backbone = None
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

    experiment.log_parameters({
        "learning_rate": float(learning_rate),
        "patience": int(args.patience),
        "finetune_backbone_lr_multiplier": None if args.finetune_backbone_lr_multiplier is None else float(args.finetune_backbone_lr_multiplier),
        "finetune_backbone_freeze_epochs": float(args.finetune_backbone_freeze_epochs),
        "batch_size": int(args.batch_size),
        "n_epochs": int(args.n_epochs),
        "n_finetuning": None if args.n_finetuning is None else int(args.n_finetuning),
        "pretrained_file_path": None if args.file_path is None else str(args.file_path),
        "imagenet_initialization": args.imagenet_initialization,
        "network_model": args.network_model,
        "pretrained_file_path": None if args.file_path is None else str(args.file_path),
    })

    optim_backbone_freeze = args.finetune_backbone_freeze_epochs

    early_stopping_enabled = None  # (val_dataloader_finetuning is not None)
    best_path = os.path.join(dest_folder, "best_model.pth")

    early_stopper = None
    if early_stopping_enabled:
        early_stopper = EarlyStopping(
            patience=int(args.patience),
            verbose=True,
            delta=1e-4,
            path=best_path,
            trace_func=print
        )

    last_epoch = 0

    for epoch in tqdm(range(n_epochs)):
        model.train()

        # Gestione del congelamento/scongelamento dinamico del backbone
        if args.finetune_backbone_lr_multiplier is not None:
            freeze_backbone = epoch < optim_backbone_freeze  # se l'epoca è minore di quella di freeze, allora backbone congelato
            for p in model.features.parameters():
                p.requires_grad = not freeze_backbone

        train_losses = []
        last_epoch = epoch
        with tqdm(train_dataloader_finetuning, leave=False, desc="Training") as t:
            for x, y in t:
                x = x.to(device)
                y = y.to(device)

                if x.size(1) == 1:
                    x = x.repeat(1, 3, 1, 1)

                optim.zero_grad()
                # if optim_backbone is not None and epoch >= optim_backbone_freeze:
                #     optim_backbone.zero_grad()
                y_pred = model(x)
                # y_pred = torch.clamp(y_pred, 0, 4)  # gt scores are between 0 and 4
                loss = criterion(y_pred, y.unsqueeze(1))
                t.set_postfix(loss=loss.item())
                loss.backward()
                optim.step()
                # if optim_backbone is not None and epoch >= optim_backbone_freeze:
                #     optim_backbone.step()

                train_losses.append(loss.item())
        
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

            if early_stopper.early_stop:
                print(
                    f"Early stopping at epoch {epoch+1}. "
                    f"Best model saved at epoch {early_stopper.best_epoch} to: {best_path} "
                    f"(best_val_loss={early_stopper.best_val_loss:.6f})"
                )
                break

        model.eval()
        test_losses = []
        all_predictions = []
        all_targets = []
        all_names = []   # per allineamento pred/gt con l'immagine

        with tqdm(test_dataloader_finetuning, leave=False, desc="Test") as t:
            for batch in t:
                # x, y, paths = batch
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
                # y_pred = torch.clamp(y_pred, 0, 4)  # gt scores are between 0 and 4

                y_pred_np = y_pred.squeeze(1).cpu().numpy()
                y_np = y.cpu().numpy()

                all_predictions.append(y_pred_np)  
                all_targets.append(y_np)   

                # all_predictions.append(y_pred.squeeze(1).cpu().numpy())  
                # all_targets.append(y.cpu().numpy())                    

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

        # experiment.log_metric(f"{dist_type}_spearman", sp_corr, step=epoch)
        # experiment.log_metric(f"{dist_type}_kendall", kd_corr, step=epoch)
        # experiment.log_metric(f"{dist_type}_pearson", pr_corr, step=epoch)

        scheduler.step()  # should be every epoch
        # if scheduler_backbone is not None and epoch >= optim_backbone_freeze:
        #     scheduler_backbone.step()  # should be every epoch
    
    # test finale sul modello con la migliore validation test
    if os.path.exists(best_path):
        print("Loading best model for final evaluation...")
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        torch.save(model.state_dict(), best_path)

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

            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)

            with torch.no_grad():
                y_pred = model(x)

            loss = criterion(y_pred, y.unsqueeze(1))
            t.set_postfix(loss=loss.item())
            test_losses.append(loss.item())

            # y_pred = torch.clamp(y_pred, 0, 4)
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
        "best_epoch": int(early_stopper.best_epoch) if early_stopper is not None else last_epoch
    }
    with open(os.path.join(dest_folder, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


        #     if epoch == (n_epochs - 1):  
        #         # dizionari pred/gt
        #         pred_dict = {name: float(pred) for name, pred in zip(all_names, all_predictions)}
        #         # gt_dict   = {name: float(gt)   for name, gt   in zip(all_names, all_targets)}

        #         with open(os.path.join(dest_folder, "predictions_finetuning_dict.json"), "w", encoding="utf-8") as f:
        #             json.dump(pred_dict, f, ensure_ascii=False, indent=2)

        #         # metriche
        #         metrics = {
        #             "spearman_r": float(sp_corr), "spearman_p": float(sp_p),
        #             "kendall_tau": float(kd_corr), "kendall_p": float(kd_p),
        #             "pearson_r": float(pr_corr), "pearson_p": float(pr_p),
        #         }
        #         with open(os.path.join(dest_folder, f"correlation_metrics_{epoch}.json"), "w", encoding="utf-8") as f:
        #             json.dump(metrics, f, indent=2)

        # scheduler.step()  # should be every epoch
        # if scheduler_backbone is not None and epoch >= optim_backbone_freeze:
        #     scheduler_backbone.step()  # should be every epoch