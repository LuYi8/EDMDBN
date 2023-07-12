from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import  StratifiedGroupKFold
from merlib.data import TEMP_DIR_PATH
from merlib.data.base import ME_DATASET,DATASET_CHOICES,MEGC2019CD,FPS200CD, \
CASME2_info, SAMM_info, SMICHSE_info, SMICHS_info,FiveEmotion, ThreeEmotion,MMEW_info,FPS200CD2
from merlib.data.clean_doc import build_cleaned_doc
import warnings

WRITE_CLOUMNS = ['path', 'start_frame', 'apex_frame', 'end_frame', 'label']
GROUP_HEAD='group'
RANDOM_SEED=42 
CROSS_DATABASE=[ MEGC2019CD.name,FPS200CD.name,FPS200CD2.name]
def generate_annotations_explanation(df:pd.DataFrame,file_path:Path,n_class_id2str_fun=ThreeEmotion.id2str):
    details={}
    details['num_samples']=len(df)
    details['num_groups']=len(set(df[GROUP_HEAD]))
    _=len(set(df['label']))
    details['num_labels']='total: {},标签含义'.format(_) + ", ".join([ "{}:{}".format(x,n_class_id2str_fun(x)) for x in range(_)])
    num_value_dict=df['label'].value_counts()
    details['distribution_of_labels']='各标签数量：' + ", ".join([ "{}:{}".format(x,num_value_dict[x]) for x in range(_)])
    pd.Series(details).to_csv(file_path,header=None,sep=':')
    pass


def generate_kfolds_annotations(num_classes: int, save_path: Path, df: pd.DataFrame):
    """ Generate loso and 5,10 folds cross validation annotations text files.

    根据num_classes 和相关标注文件产生的dataframe, 在指定的save_path下生成几类标注文件或文件夹
    1) (n)classes-5folds-annotations文件夹
    2) (n)classes-10folds-annotations文件夹
    3) (n)classes-loso-annotations文件夹 loso,leave one subject out,留一法
    """
    assert df.isna().values.any() == False
    assert save_path.exists()
    if 'dataset_name' in df:
        # for cross database
        WRITE_CLOUMNS.append('dataset_name')
    # 对应 all 
    cur_save_path= save_path/'{}classes-all_samples.txt'.format(num_classes)
    df.to_csv(cur_save_path,
                            sep=' ', header=False, columns=WRITE_CLOUMNS, index=False)
    # 对应5&10-folds 和loso
    for n in [5, 10, 'loso']:
        _ = 'loso' if isinstance(n, str) else "{}folds".format(n)
        cur_save_path = save_path / '{}classes-{}-annotations'.format(num_classes, _)

        if not cur_save_path.exists():
            cur_save_path.mkdir()

        n_split = len(set(df.group)) if isinstance(n, str) else n
        _ = StratifiedGroupKFold(n_split).split(df.label, df.label, df.group)

        for idx, (train_ids, val_ids) in enumerate(_):

            df.loc[train_ids].to_csv(cur_save_path/'fold{}_train.txt'.format(idx),
                                     sep=' ', header=False, columns=WRITE_CLOUMNS, index=False)
            df.loc[val_ids].to_csv(cur_save_path/'fold{}_val.txt'.format(idx),
                                    sep=' ', header=False, columns=WRITE_CLOUMNS, index=False)


def transform_labels_to_ids(num_classes: int, df: pd.DataFrame,dataset_name=None):
    if num_classes == 3:
        df['label'] = df['emotion_label'].map(ThreeEmotion.transform)

        # df=df.dropna(axis=0)
        df = df[~df.label.isna()]  # 过滤未转化的sample
        df = df.reset_index(drop=True)
        df['label'] = df['label'].map(ThreeEmotion.str2id)
        return df
    if num_classes == 5 or num_classes == 4:
        df['label'] = df['emotion_label'].map(FiveEmotion(dataset_name).transform)
        df = df[~df.label.isna()]  # 过滤情感类型为other的sample
        df = df.reset_index(drop=True)
        df['label'] = df['label'].map(FiveEmotion(dataset_name).str2id)
        return df

def main(args):
    num_classes = args.num_classes
    dataset_name = args.dataset_name
    dataset_info = ME_DATASET[dataset_name]

    # debug模式下，剪裁的数据保存在当前目录的/temp/dataset_name下，否则使用数据集元信息中save_path
    save_path: Path = (TEMP_DIR_PATH/dataset_name) if args.debug else Path( dataset_info.save_path) 
    
    # 如果额外指定了save_path,则优先保存在此目录
    if args.save_path:
        save_path = Path(args.save_path)
    
    save_dir_name=''
    if args.use_given_faces_data:
        save_dir_name='given_faces_data_annotations'
    else:
        warnings.warn('使用自己处理，而非已处理的数据集，将在原标注的视频路径后加上 {startframe}_{endframe}进行唯一标注样本', UserWarning)
        save_dir_name='unique_path_data_annotations'

    save_path=save_path/save_dir_name
    if save_path and not save_path.exists():
        print("{} doesn't exist, it will be created!".format(save_path.resolve()))
        save_path.mkdir(parents=True)

    if dataset_name not in CROSS_DATABASE:
        # 对单个数据库生成有监督标注
        df:pd.DataFrame = build_cleaned_doc(dataset_name, dataset_info.doc_path)
        # if df is not None: print(df.info())
        df = transform_labels_to_ids(num_classes, df,dataset_name)
        if not args.use_given_faces_data:
            df['path']=df['unique_path']

        generate_kfolds_annotations(num_classes, save_path, df)

        n_class_id2str_fun=ThreeEmotion.id2str
        if args.num_classes==5:
            n_class_id2str_fun=FiveEmotion(args.dataset_name).id2str
        if args.num_classes==4 and args.dataset_name==MMEW_info.name:
            n_class_id2str_fun=FiveEmotion(args.dataset_name).id2str
        generate_annotations_explanation(df,save_path/'{}classes_annotations_explanation.txt'.format(num_classes), n_class_id2str_fun)
        
    # 生成交叉数据库的标注信息
    else:
        if dataset_name == MEGC2019CD.name:
            data_root_path={
                CASME2_info.name:'/home/neu-wang/mby/database/CASME2/dlib_crop_twice',
                SMICHS_info.name:SMICHS_info.raw_data_path,
                SAMM_info.name: '/home/neu-wang/mby/database/SAMM/dlib_crop_twice'
            }
        elif dataset_name == FPS200CD.name:
            data_root_path={
            CASME2_info.name:'/home/neu-wang/mby/database/CASME2/dlib_crop_twice',
            # MMEW_info.name: MMEW_info.raw_data_path,
           MMEW_info.name: '/home/neu-wang/mby/database/MMEW/dlib_crop_twice',
            SAMM_info.name: '/home/neu-wang/mby/database/SAMM/dlib_crop_twice'
            }
        elif dataset_name == FPS200CD2.name:
            print(dataset_name)
            data_root_path={
            CASME2_info.name:'/home/neu-wang/mby/database/CASME2/dlib_crop_twice',
            # MMEW_info.name: MMEW_info.raw_data_path,
            SAMM_info.name: '/home/neu-wang/mby/database/SAMM/dlib_crop_twice'
            }
        df= build_cleaned_doc(dataset_name,dataset_info.doc_path, data_root_path)
        df = transform_labels_to_ids(num_classes, df)
        generate_kfolds_annotations(num_classes, save_path, df)
        generate_annotations_explanation(df,save_path/'{}classes_annotations_explanation.txt'.format(num_classes))

def get_args():
    parser = argparse.ArgumentParser(
        'prepare supervised nclass kfolds annotations.txt for the specified dataset'
        ,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'dataset_name', choices=DATASET_CHOICES)
    parser.add_argument('--no-debug', action='store_false',dest='debug', help='根据debug模式确定将处理的数据保存在当下的temp目录还是指定目录')
    parser.add_argument('--debug',default=True, action='store_true', help='根据debug模式确定将处理的数据保存在当下的temp目录还是指定目录')

    parser.add_argument('--use_given_faces_data',default=False,action='store_true',help='如果使用数据集携带的、预处理好的人脸数据，则路径是非unique_path')

    parser.add_argument(
        '--save_path', default=None,help='如果不指定，在debug模式下，剪裁的数据保存在当前目录的/temp/dataset_name下，否则使用数据集元信息中save_path')
    parser.add_argument('-nc', '--num_classes', default=3, type=int,help='确定几分类')
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
