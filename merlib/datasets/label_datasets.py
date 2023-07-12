from pathlib import Path
from random import choices
from typing import Any, Dict, List, Optional, Type
import torch.utils.data
from .utils.me_video_dataset import VideoFrameDataset
from torch.utils.data.dataloader import DataLoader
from .transforms import get_me_transform
from .utils.sampler import ImbalancedDatasetSampler
from pytorch_lightning import LightningDataModule
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np

from merlib.data.base import ME_DATASET, DATASET_CHOICES
FACE_REGISTRATION_METHOD=['openface']


def add_dataset_specific_args(parent_parser):
    parser = parent_parser.add_argument_group(title="dataset",description="mer dataset params")
    # dataloader 参数
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    # 指定标注文件和数据集路径
    parser.add_argument('--data_root', default='/home/neu-wang/mby/database/CASME2/dlib_crop_twice2', type=str,
                        help='dataset path')
    parser.add_argument('--annotation_path', default='/home/neu-wang/mby/database/CASME2/unsupervised-annotations.txt', type=str,
                        help='annotation_file')
    
    # 控制统一帧数量和大小
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--num_frames', default=16, type=int,
                        help='images frames size')
    # 控制帧采样方式
    # 0-all frames, 1-apex_frame, 2-onset,apex frames, 3-onset,apex,offsetframes, 4-frames_perseqment
    parser.add_argument('--sampling_strategy', default=4, type=int,choices=[0,1,2,3,4])
    parser.add_argument('--num_segments', default=10, type=int)
    parser.add_argument('--frames_per_segment', default=1, type=int)
    # 控制插值策略
    parser.add_argument('--interpolation_strategy',choices=['FixSequenceInterpolation'],default='FixSequenceInterpolation')
    # 离线增强
    parser.add_argument('--use_offline_aug',default=False,action="store_true",help="启用离线数据增强")
    # 数据增强
    parser.add_argument('--random_scale',type=float, nargs='+',default=[0.7,1.0])
    # 其它
    # 人脸配准方法
    parser.add_argument(
        '--registration_method','-reg', choices=FACE_REGISTRATION_METHOD, default=None,help="face registration")
    # 通过dataset_name获取对应的图片后缀
    parser.add_argument(
        '--dataset_name', choices=DATASET_CHOICES, default=None,help="help to get img suffix")
    # 是否启用光流
    parser.add_argument('--optical_flow', action='store_true',
                            help='feed gray image and x,y optical flow to three channels ')
    parser.set_defaults(optical_flow=False)
    # 是否启用加权sampler
    parser.add_argument('--use_weighted_sampler', action='store_true',
                            help='use weighted sampler')
    parser.set_defaults(use_weighted_sampler=False)

    return parent_parser

def get_blanced_class_weight(path:str):
    x=pd.read_csv(path,sep=' ',header=None)
    labels=x[4] # 根据annotation标注文件
    classes=np.array(sorted(set(labels)))
    # 样本均衡后的权重
    weight = compute_class_weight('balanced', classes=classes, y=labels)
    print('class weight: ',weight)
    return list(weight.data)
from typing import Union
def trans2augtrain_file(train_annotation_file_path:Union[str,Path]):
            # aug_df
    def generate_offline_aug_train_anno(train_annotation_file_path:Union[str,Path],aug_anno_path):
        import pandas as pd
        orign_df=pd.read_csv(train_annotation_file_path,sep=' ',header=None)
        derived_df1=orign_df.copy()
        derived_df1[3]=derived_df1[2]
        # derived_df1
        derived_df2=orign_df.copy()
        derived_df2[1]=derived_df2[2]
        # derived_df2
        aug_df=pd.concat([orign_df,derived_df1,derived_df2],ignore_index=True)
        # aug_df
        aug_df.to_csv(aug_anno_path,header=False,index=False,sep=' ')
        


    name=Path(train_annotation_file_path).name.split('_')
    name.insert(1,'offline_aug')
    name='_'.join(name)
    aug_anno_path=Path(train_annotation_file_path).parent/name
    if not aug_anno_path.exists():
        generate_offline_aug_train_anno(train_annotation_file_path,aug_anno_path)
    assert aug_anno_path.exists()
    return aug_anno_path

def get_all_data():
    pass
def get_holdoutdatamodules_from_args(args,transform=None):
    """
    接受一个path,指定一个文件夹路径，内容包括，train.txt, test.txt。
    """
    annotation_path = Path(args.annotation_path)
    assert annotation_path.exists() and annotation_path.is_dir()
    train_annotation_file = annotation_path / "train.txt" 
    
    val_annotation_file = annotation_path / "val.txt" 
    assert train_annotation_file.exists()
    return DataModule(args,
                         train_annotation_file=train_annotation_file.resolve().as_posix(),
                         val_annotation_file=val_annotation_file.resolve().as_posix() if val_annotation_file.exists() else None,
                         transform=transform)

def get_kfolddatamodules_from_args(args,resume_fold=0,transform=None):
    annotation_path = Path(args.annotation_path)
    assert annotation_path.exists() and annotation_path.is_dir()

    train_annotation_list = list(annotation_path.glob("fold*_train.txt"))
    val_annotation_list = list(annotation_path.glob("fold*_val.txt"))
    # assert len(train_annotation_list) == len(val_annotation_list)
    print("进行数据集 {} 的 {} 折 交叉验证训练".format(
        args.dataset_name, len(val_annotation_list)))
    for i in range(resume_fold,len(val_annotation_list)):
        
        train_annotation_file = annotation_path / "fold{}_train.txt".format(i)
        if args.use_offline_aug:
            train_annotation_file=trans2augtrain_file(train_annotation_file)
        val_annotation_file = annotation_path / "fold{}_val.txt".format(i)
        yield DataModule(args,
                         train_annotation_file=train_annotation_file.resolve().as_posix(),
                         val_annotation_file=val_annotation_file.resolve().as_posix(),
                         transform=transform
                         )


class DataModule(LightningDataModule):
    def __init__(self, args, train_annotation_file:str, val_annotation_file:str,transform):
        super().__init__()
        self.weights= get_blanced_class_weight(train_annotation_file)
        self.train_loader = get_me_dataloader(
            args, train_annotation_file, is_train=True,transform=transform)
        self.val_loader = get_me_dataloader(
            args, val_annotation_file, is_train=False,transform=transform) if val_annotation_file else None
        self.labels=ME_DATASET[args.dataset_name].get_labels(args.num_classes)
    def prepare_data(self) -> None:
        # download the data.
        # only called on 1 GPU/TPU in distributed
        # Note we do not make any state assignments in this function
        # print('prepare')
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # make assignments here (val/train/test split)
        # Setup expects a 'stage' arg which is used to separate logic for 'fit' and 'test'.
        # Note this runs across all GPUs and it is safe to make state assignments here
        # print('setup')
        pass

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()


def get_me_dataloader(args, annotation_file: str, is_train=True,transform=None):
    num_samples=args.num_segments*args.frames_per_segment
    if transform is None:
        transform=get_me_transform(args.input_size, is_train,num_samples,args.optical_flow,args.interpolation_strategy,args.random_scale
                                   ,registration_method=args.registration_method)
    dataset = VideoFrameDataset(
        root_path=args.data_root,
        annotationfile_path=annotation_file,
        num_segments=args.num_segments,
        frames_per_segment=args.frames_per_segment,
        imagefile_template=ME_DATASET[args.dataset_name].imagefile_template,
        transform= transform,
        sampling_strategy=args.sampling_strategy,
        test_mode=not is_train,
       )
    
    use_weighted_sampler=args.use_weighted_sampler
    sampler=ImbalancedDatasetSampler(dataset) if is_train and use_weighted_sampler else None
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=1 if args.debug else args.num_workers ,
        shuffle=None if sampler else is_train ,
        sampler=sampler ,
        pin_memory=args.pin_mem,
        drop_last=is_train
    )

