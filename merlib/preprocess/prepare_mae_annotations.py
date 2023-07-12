from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import  train_test_split

from utils.base import ME_DATASET,DATASET_CHOICES, ThreeEmotion
from utils.clean_doc import build_cleaned_doc
import warnings

WRITE_CLOUMNS = ['path', 'start_frame', 'apex_frame', 'end_frame', 'label']
GROUP_HEAD='group'
RANDOM_SEED=42 

def generate_unsupervised_annotations(save_path: Path, df: pd.DataFrame, unique=True):
    assert df.isna().values.any() == False
    df['label'] = 0

    if unique:
        warnings.warn('unique为True, 将在原标注的视频路径后加上 {startframe}_{endframe}', UserWarning)
        df['path'] =df['unique_path']

    save_path = save_path / 'unsupervised_annotations'
    if not save_path.exists():
        save_path.mkdir()

    train_ids, val_ids = train_test_split( df.index, test_size=0.15, random_state=RANDOM_SEED)
    df.loc[train_ids].to_csv(save_path/'train.txt',
                                    sep=' ', header=False, columns=WRITE_CLOUMNS, index=False)
    df.loc[val_ids].to_csv(save_path/'val.txt',
                                sep=' ', header=False, columns=WRITE_CLOUMNS, index=False)



def get_args():
    parser = argparse.ArgumentParser(
        'prepare annotations.txt for the specified dataset')
    parser.add_argument(
        'dataset_name', choices=DATASET_CHOICES)
    parser.add_argument(
        '--save_path', default=None)
    parser.add_argument('-nc', '--num_classes', default=2, type=int)
    args = parser.parse_args()
    return args

def main(args):
    num_classes = args.num_classes
    dataset_name = args.dataset_name
    dataset_info = ME_DATASET[dataset_name]

    save_path = Path(
        args.save_path) if args.save_path else dataset_info.save_path
    if save_path and not save_path.exists():
        print("{} doesn't exist, it will be created!".format(save_path.resolve()))
        save_path.mkdir()

    df:pd.DataFrame = build_cleaned_doc(dataset_name, dataset_info.doc_path)
    if df is not None: print(df.info())
    # 有监督二分类
    generate_unsupervised_annotations(save_path, df)
if __name__ == '__main__':
    args = get_args()
    main(args)