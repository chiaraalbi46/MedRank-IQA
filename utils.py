""" Define matching between image names and artifact types for LDCTIQA train and test set. """

import math
from skimage.morphology import closing, disk
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from monai.transforms import LoadImage
import json 
import os

def get_train_map(base_train_path):

    # image names and scores
    json_train_file = os.path.join(base_train_path, 'train.json')
    with open(json_train_file, 'r') as file:
        train_data = json.load(file)  # keys are names, values are iqa scores

    # artifacts
    artifacts_file = os.path.join(base_train_path, 'dist_ans_train.csv')
    arts = pd.read_csv(artifacts_file)

    ### path to image folder
    images_root = os.path.join(base_train_path, 'image')

    rows = []
    for key in train_data:
        parts = key.split('_')
        image_id = '_'.join(parts[:3])  # L096_402_1
        col_id = parts[3]  # 1

        file_path = os.path.join(images_root, key + '.tif')

        art_code = arts.loc[arts['image'] == image_id, col_id].iloc[0]
        noise_level = art_code[1:2]
        streak_level = art_code[3:4]

        rows.append([file_path, key, art_code, noise_level, streak_level, train_data[key]])

    result = pd.DataFrame(rows, columns=['image_filepath', 'image_name', 'code', 'noise_level', 'streak_level', 'score'])

    dest_train_file = os.path.join(base_train_path, 'train_map.csv')
    result.to_csv(dest_train_file)


def get_test_map(base_test_path):
    json_test_file = os.path.join(base_test_path, 'test-ground-truth.json')
    with open(json_test_file, 'r') as file:
        test_data = json.load(file)  # keys are names, values are iqa scores

    ## artifacts
    artifacts_file = os.path.join(base_test_path, 'dist_ans_test.csv')
    arts = pd.read_csv(artifacts_file)

    rows = []
    for test_key, test_values in test_data.items():

        # cicla sui test_0, test_1, test_2
        fnames = test_values["fnames"]
        scores = test_values["scores"]

        subfolder = os.path.splitext(test_key)[0]  

        for fname, score in zip(fnames, scores):
            pat_id, slice_id, row_id, col_id = fname.split('_')
            image_id = f"{pat_id}_{int(slice_id):03d}_{row_id}"  # L096_402_1
            # nb: 'S058279_70_1' nel json mentre 'S058279_070_1' nel csv artefatti

            file_path = os.path.join(base_test_path, subfolder, fname + ".tif")  # im is saved as in json

            art_code = arts.loc[arts['image'] == image_id, col_id].iloc[0]
            noise_level = art_code[1:2]
            streak_level = art_code[3:4]

            rows.append([file_path, fname, art_code, noise_level, streak_level, score])

    result = pd.DataFrame(rows, columns=['image_filepath', 'image_name', 'code', 'noise_level', 'streak_level', 'score'])

    dest_test_file = os.path.join(base_test_path, 'test_map.csv')
    result.to_csv(dest_test_file)

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


def dicts_equal_float(d1, d2, tol=1e-9):
    if d1.keys() != d2.keys():
        return False
    return all(math.isclose(d1[k], d2[k], rel_tol=tol) for k in d1)

def extract_body_crop_otsu(x, min_size=224):
    # Otsu separates body from background even if not true air
    # x = x.detach().cpu().numpy()

    t = threshold_otsu(x)
    mask = x > t

    # Clean
    mask = closing(mask, disk(7))

    # Largest connected component
    labeled = label(mask)
    regions = regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)

    minr, minc, maxr, maxc = largest.bbox

    # vorrei che il crop fosse almeno min_size x min_size
    h = maxr - minr
    w = maxc - minc

    if h < min_size:
        pad = min_size - h
        minr -= pad // 2
        maxr += pad - pad // 2

    if w < min_size:
        pad = min_size - w
        minc -= pad // 2
        maxc += pad - pad // 2
    
    H, W = x.shape
    minr = max(0, minr)
    minc = max(0, minc)
    maxr = min(H, maxr)
    maxc = min(W, maxc)

    cropped = x[minr:maxr, minc:maxc]

    return cropped, (minr, minc, maxr, maxc), mask

def save_crop_otsu_indices_pretraining(mode):
    """ Scan train and test images of pretraining dataset (separated) and save indices of Otsu crop in json for each image. """
    # these json should be loaded in pretraining for performing the Otsu crop on gpu instead of on cpu 

    root_path=f"/Prove/Albisani/TCIA_datasets/{mode}"

    image_files = []
    otsu_crops = {}
    for patient_folder in os.listdir(root_path):
        patient_path = os.path.join(root_path, patient_folder)
        if os.path.isdir(patient_path):
            image_dir = os.path.join(patient_path, "Full_Dose_Images")
            if os.path.isdir(image_dir):
                for root, _, files in os.walk(image_dir):
                    for f in files:
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tif", ".dcm")):
                            img_path = os.path.join(root, f)
                            print(img_path)
                            image_files.append(img_path)
                            #### compute Otsu crop 
                            
                            im = LoadImage(image_only=True)(img_path)

                            im_np = im.detach().cpu().numpy()

                            # call function
                            x_out, (minr, minc, maxr, maxc), mask = extract_body_crop_otsu(im_np)

                            # save json
                            otsu_crops[img_path] = [int(minr), int(minc), int(maxr), int(maxc)]

    json_path = os.path.join('/Prove/Albisani/TCIA_datasets', f"otsu_crops_{mode}.json")
    with open(json_path, "w") as f:
        json.dump(otsu_crops, f, indent=2)

    print(f"Saved Otsu crop boxes for {len(otsu_crops)} images to {json_path}")



def save_crop_otsu_indices_finetuning(mode):
    """ Scan train and test images of finetuning dataset (separated) and save indices of Otsu crop in json for each image. """

    if mode == 'train':

        train_images_root = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAG2023_train/image"
        train_json        = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAG2023_train/train.json"

        train_dict = build_path_score_dict(train_json, train_images_root, is_test=False)

        items = list(train_dict.items())
    else:
        test_images_root  = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAC2023_test"
        test_json         = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAC2023_test/test-ground-truth.json"

        test_dict  = build_path_score_dict(test_json, test_images_root, is_test=True)

        items = list(test_dict.items())

    loader_tif = LoadImage(image_only=True, reader="PILReader")

    otsu_crops = {}
    for idx in range(len(items)):
        img_path, score = items[idx]
        print(img_path)
        image = loader_tif(img_path) 

        im_np = image.detach().cpu().numpy()

        # call function
        x_out, (minr, minc, maxr, maxc), mask = extract_body_crop_otsu(im_np)

        # save json
        otsu_crops[img_path] = [int(minr), int(minc), int(maxr), int(maxc)]

    json_path = os.path.join('/Prove/Albisani/LDCTIQA_dataset', f"otsu_crops_finetuning_{mode}.json")
    with open(json_path, "w") as f:
        json.dump(otsu_crops, f, indent=2)

    print(f"Saved Otsu crop boxes for {len(otsu_crops)} images to {json_path}")

    
if __name__ == "__main__":
    import pandas as pd

    ############################# train
    base_train_path = '/Prove/Albisani/LDCTIQA_dataset/LDCTIQAG2023_train'

    # get_train_map(base_train_path)

    map_file = os.path.join(base_train_path, 'train_map.csv')
    df_train = pd.read_csv(map_file, index_col=0)

    image_score_dict_train = dict(zip(df_train['image_filepath'], df_train['score']))  # this create exactly same dict as build_path_score_dict

    train_images_root = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAG2023_train/image"
    train_json        = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAG2023_train/train.json"
    train_dict = build_path_score_dict(train_json, train_images_root, is_test=False)

    if dicts_equal_float(image_score_dict_train, train_dict, tol=1e-9):
        print("train dictionary is ok ")

    ############################# test 
    base_test_path = '/Prove/Albisani/LDCTIQA_dataset/LDCTIQAC2023_test'

    # get_test_map(base_test_path)

    map_file_test = os.path.join(base_test_path, 'test_map.csv')
    df_test = pd.read_csv(map_file_test, index_col=0)

    image_score_dict_test = dict(zip(df_test['image_filepath'], df_test['score'])) 

    test_images_root  = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAC2023_test"
    test_json         = "/Prove/Albisani/LDCTIQA_dataset/LDCTIQAC2023_test/test-ground-truth.json"
    test_dict  = build_path_score_dict(test_json, test_images_root, is_test=True)

    if dicts_equal_float(image_score_dict_test, test_dict, tol=1e-9):
        print("test dictionary is ok ")
