""" Linear probing on pretrained models """
# predizione artefatto
# predizione scores 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pretraining_networks import Vgg16, SiameseRankIQA, RankIQA_branch, Resnet18, Resnet18SiameseRankIQA, Resnet18RankIQA_branch, SqueezeNet1_1, SqueezeNet1_1SiameseRankIQA, SqueezeNet1_1RankIQA_branch

import random
from random import randrange
import numpy as np 
import os
import json
from monai.transforms import LoadImage, Lambda
from monai.data.image_reader import PydicomReader

import h5py # pip install h5py


from utils import build_path_score_dict 

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

def debug_vgg_shapes(vgg, x):
    print("\n=== DEBUG VGG SHAPES ===")
    for name, layer in vgg.features._modules.items():
        x = layer(x)
        print(f"{name:10s} -> {tuple(x.shape)}")
    return x

class VGGFeatureExtractor(nn.Module):
    def __init__(self, vgg, layer_name):
        super().__init__()
        self.features = vgg.features
        self.layer_name = layer_name

    def forward(self, x):
        for name, layer in self.features._modules.items():
            x = layer(x)  # output del layer 
            if name == self.layer_name:
                return x

        raise ValueError(f"Layer {self.layer_name} not found")

# per più layers
class VGGMultiLayerExtractor(nn.Module):
    def __init__(self, vgg, layers):
        super().__init__()
        self.features = vgg.features
        self.classifier = vgg.classifier
        self.layers = set(layers)

    def forward(self, x):
        outputs = {}

        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in self.layers:
                outputs[name] = x
                print(f"Extracting {name} shape {x.shape}")

        # flatten
        x = torch.flatten(x, start_dim=1, end_dim=-1) 

        for name, layer in self.classifier._modules.items():
            x = layer(x)

            if name in self.layers:
                outputs[name] = x
                print(f"Extracting {name} shape {x.shape}")

        return outputs

def process_features(feat_dict):
    out = {}

    # for k, v in feat_dict.items():
    #     v = torch.nn.functional.adaptive_avg_pool2d(v, (1, 1))  # capire se è ottimale ...
    #     v = v.view(v.size(0), -1)  # [B, C]
    #     out[k] = v.detach().cpu().numpy()

    for k, v in feat_dict.items():

        # CONV FEATURES: [B,C,H,W]
        if v.ndim == 4:
            v = torch.nn.functional.adaptive_avg_pool2d(v, (1, 1))
            v = v.view(v.size(0), -1)

        # FC FEATURES: [B,D]
        elif v.ndim == 2:
            pass

        else:
            raise ValueError(f"Unexpected shape for {k}: {v.shape}")

        out[k] = v.detach().cpu().numpy()

    return out

class HDF5FeatureWriter:
    def __init__(self, save_path, feature_dims, N, layers, dtype="float32"):
        self.f = h5py.File(save_path, "w")
        self.N = N
        self.idx = 0

        # salva metadata
        self.f.attrs["layers"] = list(layers)

        # features
        self.datasets = {}
        for name, dim in feature_dims.items():
            self.datasets[name] = self.f.create_dataset(
                name,
                shape=(N, dim),
                dtype=dtype,
                compression="gzip",
                chunks=(64, dim)
            )

        # scores
        self.scores = self.f.create_dataset(
            "scores",
            shape=(N,),
            dtype=dtype
        )

        # paths (stringhe)
        self.paths = self.f.create_dataset(
            "paths",
            shape=(N,),
            dtype=h5py.string_dtype(encoding="utf-8")
        )

        self.preds = self.f.create_dataset(
            "preds",
            shape=(N,),
            dtype="float32"
        )

    def write_batch(self, feats, scores, preds, paths):
        b = len(paths)

        for k, v in feats.items():
            self.datasets[k][self.idx:self.idx+b] = v

        self.scores[self.idx:self.idx+b] = scores
        self.preds[self.idx:self.idx+b] = preds
        self.paths[self.idx:self.idx+b] = np.array(paths, dtype=object)

        self.idx += b
    
    def close(self):
        self.f.close()

def infer_feature_dims(model, dataloader, device):
    x, _, _ = next(iter(dataloader))

    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)

    x = x.to(device)

    with torch.no_grad():
        feats = process_features(model(x))

    return {k: v.shape[1] for k, v in feats.items()}

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--n_finetuning', dest="n_finetuning", type=int, nargs='?', default=None,
                        help="number of images for training (from scratch or finetuning). if None use all images of the train set.") 
    #serve per usare un minor numero di immagini del dataset di finetuning
    # parser.add_argument('sampling_json', nargs='?', default='./sampling_dict_12_48_240_612_seed42.json',
    #                 help="path to sampling indices json file. sampling_dict_12_48_240_612_*.json")
    
    parser.add_argument('--file_path', dest="file_path", nargs='?', default=None,  # './Pytorch_TestRankIQA_pretrained/Rank_tid2013.caffemodel.pt',
                        help="path to the pretrained model (phase 1). if None the model is trained from scratch.") #se nella riga di comando sul terminale non ci metto niente, lui non 
    #carica niente cioè non carica il modello con i pesi salvati, se ci scrivo il nome del file con i pesi salvati invece li carica 

    parser.add_argument("--batch_size", dest="batch_size", default=32, help="batch size for train and test")
    
    parser.add_argument('--device_id', dest="device_id",  default='0', help='gpu device id.')

    parser.add_argument('--imagenet_initialization', dest="imagenet_initialization",  default=None, help='use (1) or not (None) imagenet weights for VGG16')
     # '/data/lesc/staff/albisani/MedRank-IQA/Pytorch_TestRankIQA_pretrained/Rank_tid2013.caffemodel.pt'

    parser.add_argument('--network_model', dest="network_model",  default='vgg16', help='specify the network to use. vgg16, resnet18, squeezenet1_1')

    parser.add_argument('--data_type', dest="data_type",  default='train', help='specify if you want to extract features from test or train set')

    parser.add_argument('--base_save_path', dest="base_save_path", default='/Prove/Albisani/medrank_iqa_network_features_analysis',
                        help='path where .h5 files are stored')

    args = parser.parse_args()

    ## dataset train e test come per finetuning
    train_images_root = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAG2023_train/image"
    train_json        = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAG2023_train/train.json"
    train_map_csv_default = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAG2023_train/train_map.csv"

    test_images_root  = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAC2023_test"
    test_json         = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAC2023_test/test-ground-truth.json"

    train_dict_full = build_path_score_dict(train_json, train_images_root, is_test=False)  # all train images
    # train_dict = build_path_score_dict(train_json, train_images_root, is_test=False)
    test_dict  = build_path_score_dict(test_json, test_images_root, is_test=True)

    train_images = 'all'  # if None all available train images are used
    val_dict = None
    train_dict = train_dict_full

    train_dataset_finetuning = FineTuningDictDataset(train_dict, n=args.n_finetuning, return_path=True, mode='train')  # ho messo return path=True
    test_dataset_finetuning  = FineTuningDictDataset(test_dict, return_path=True, mode='test')

    train_dataloader_finetuning = DataLoader(dataset=train_dataset_finetuning, batch_size=int(args.batch_size), shuffle=True)
    test_dataloader_finetuning  = DataLoader(dataset=test_dataset_finetuning,  batch_size=int(args.batch_size))

    ## recupero il modello da cui voglio estrarre le features
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

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
        else:
            print("Test SqueezeNet 1.1 from scratch")
            vgg = SqueezeNet1_1(imagenet=args.imagenet_initialization)
            model = SqueezeNet1_1RankIQA_branch(squeezenet1_1_model=vgg).to(device)
        
    
    ## ora estraggo le features dal livello che mi interessa 
    # layers = ["conv1_2", "conv2_2", "conv3_3", "conv4_3", "conv5_3"]
    # feature_extractor = VGGFeatureExtractor(vgg=vgg, layer_name='conv2_2')
    # vgg e non RankIQA_branch sennò non ho  i nomi dei layer espliciti

    # model.eval()

    # ######################### debug
    # # prendi una batch
    # images, _ = next(iter(train_dataloader_finetuning))
    # images = images.to(device)

    # if images.size(1) == 1:
    #     images = images.repeat(1, 3, 1, 1)

    # with torch.no_grad():
    #     debug_vgg_shapes(vgg, images)
    
    # with torch.no_grad():
    #     feat = feature_extractor(images)

    # print("Extracted feature:", feat.shape)

    # #########################

    base_save_path = args.base_save_path # 
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)
    
    model_name = args.file_path.split('/')[-1].replace('.pth', '')
    data_type = str(args.data_type)

    dataloader = test_dataloader_finetuning
    if data_type == 'test':
        print("Test set feature extraction")
    else:
        print("Train set feature extraction")
        dataloader = train_dataloader_finetuning
    
    save_path = os.path.join(base_save_path, f'features_{args.network_model}_{model_name}_{data_type}.h5')

    if args.network_model == 'vgg16': 
        layers = ['relu2_2', 'relu3_3', 'relu5_3', 'relu6_m','relu7_m']
    elif args.network_model == 'resnet18':
        layers = ['3','5','7']
    else:
        # squeezenet 1.1
        layers = ['2', '7', '12']

    print("Layers to be extracted: ", layers)

    feature_extractor = VGGMultiLayerExtractor(vgg=vgg, layers=layers)

    model.eval()
    feature_extractor.eval()

    N = len(dataloader.dataset)

    # feature_dims = infer_feature_dims(model, test_dataloader_finetuning, device)
    feature_dims = infer_feature_dims(feature_extractor, dataloader, device)

    writer = HDF5FeatureWriter(save_path, feature_dims, N, layers)

    ## save extracted features 
    with tqdm(dataloader, leave=False, desc=f"{data_type}") as t:
        for batch in t:
            x, y, paths = batch
            x = x.to(device)
            y = y.to(device)

            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)

            with torch.no_grad():
                feats = process_features(feature_extractor(x))
                y_pred = model(x)  # è la predizione che fa il modello pretrainato senza ulteriore finetuning 
            
            y_pred = torch.clamp(y_pred, 0, 4)  # gt scores are between 0 and 4
            y_pred_np = y_pred.squeeze(1).cpu().numpy()  # pred

            y_np = y.cpu().numpy()  # gt 

            writer.write_batch(
                feats,
                y_np,
                y_pred_np,
                paths
            )

    writer.close()