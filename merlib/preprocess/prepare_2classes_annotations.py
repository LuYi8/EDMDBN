###################
# 生成宏表情和微表情的二分类标注文件
from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.model_selection import  StratifiedGroupKFold,LeavePGroupsOut

from data.base import ME_DATASET,DATASET_CHOICES, ThreeEmotion
from data.clean_doc import build_cleaned_doc
import warnings

WRITE_CLOUMNS = ['path', 'start_frame', 'apex_frame', 'end_frame', 'label']
GROUP_HEAD='group'
RANDOM_SEED=42 

def transform_labels_to_ids(num_classes: int, df: pd.DataFrame):
    if num_classes == 2:
        micro = ['micro', 'micro-expression', 'micro expression']

        df['label'] = df['expression_type'].map(
            lambda x: (1 if str(x).lower() in micro else 0))
        return df

def generate_micro_and_macro_annotations(save_path: Path, df: pd.DataFrame):
    assert df.isna().values.any() == False
    df = transform_labels_to_ids(2, df)    # type: ignore


    save_path = save_path / 'micro_and_macro_annotations'
    
    if not save_path.exists():
        save_path.mkdir()

    num_groups=len(set(df.group))
    num_group_left= num_groups//10  if num_groups//10 >=2 else 2
    print("{} train, {} val".format(num_groups-num_group_left,num_group_left)  )
    
    for train_ids, val_ids in LeavePGroupsOut(n_groups=num_group_left).split(df.path,df.label,df.group):
        df.loc[train_ids].to_csv(save_path/'train.txt',
                                        sep=' ', header=False, columns=WRITE_CLOUMNS, index=False)
        df.loc[val_ids].to_csv(save_path/'val.txt',
                                    sep=' ', header=False, columns=WRITE_CLOUMNS, index=False)
        return

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
    if dataset_info.unique:
        warnings.warn('unique为True, 将在原标注的视频路径后加上 {startframe}_{endframe}', UserWarning)
        df['path'] =df['unique_path']
    # 有监督二分类
    generate_micro_and_macro_annotations(save_path, df)
if __name__ == '__main__':
    args = get_args()
    main(args)