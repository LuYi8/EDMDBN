from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import  StratifiedGroupKFold
from merlib.data import TEMP_DIR_PATH
from merlib.data.base import ME_DATASET,DATASET_CHOICES,MEGC2019CD,FPS200CD, \
CASME2_info, SAMM_info, SMICHSE_info, SMICHS_info,FiveEmotion, ThreeEmotion,MMEW_info,FPS200CD2
from merlib.data.clean_doc import build_cleaned_doc
import warnings
from PIL import Image
import shutil

def main(args):
    orign_path_list=[]
    now_path_list=[]
    dataset_name = args.dataset_name
    data_root = Path(args.data_root)
    new_data_root = Path(args.data_root).parent/ (Path(args.data_root).name+'_unique_path')
    new_data_root.mkdir(parents=True,exist_ok=True)
    dataset_info = ME_DATASET[dataset_name]

    df:pd.DataFrame = build_cleaned_doc(dataset_name, dataset_info.doc_path)
    orign_path_list=Path()/ data_root/ df['path']
    now_path_list=Path()/ new_data_root/ df['unique_path']
    for img_path, now_path in zip(orign_path_list,now_path_list):
        img_path= Path(img_path)
        now_path = Path(now_path)
        shutil.copy2(img_path, now_path)
    pass


def get_args():
    parser = argparse.ArgumentParser(
        'transform images path to new path'
        ,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'dataset_name', choices=DATASET_CHOICES)
    parser.add_argument(
        'data_root',type=str)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)