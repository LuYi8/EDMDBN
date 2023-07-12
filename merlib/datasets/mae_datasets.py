from pathlib import Path

import torch.utils.data
# from .utils.video_dataset import VideoFrameDataset
from .utils.me_video_dataset import VideoFrameDataset
from preprocess.utils.base import CASME3UL_info,CASME2_info
from .transforms import get_mae_transform

UL_DATASET=CASME3UL_info
# UL_DATASET=CASME2_info
# A row for each video sample as: (VIDEO_PATH START_FRAME END_FRAME CLASS_ID)


def add_dataset_specific_args(parent_parser):
    parser = parent_parser.add_argument_group("MAE Dataset")
    
    # Dataset parameters
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--input_size', default=224, type=int,
                            help='images input size')
    parser.add_argument('--num_segments', default=10, type=int)
    parser.add_argument('--frames_per_segment', default=1, type=int)

    parser.add_argument('--data_root', default='/home/neu-wang/mby/database/CASME2/dlib_crop_twice2', type=str,
                        help='dataset path')
    parser.add_argument('--annotation_file', default='/home/neu-wang/mby/database/CASME2/unsupervised-annotations.txt', type=str,
                        help='annotation_file')
    return parent_parser

def get_maetrain_dataloader_from_args(args):
    num_frames=args.num_segments*args.frames_per_segment
    dataset = VideoFrameDataset(
        root_path=args.data_root,
        annotationfile_path=args.annotation_file,
        num_segments=args.num_segments,
        frames_per_segment=args.frames_per_segment,
        imagefile_template=UL_DATASET.imagefile_template,
        transform=get_mae_transform(args.input_size,num_frames=num_frames),
        data_mode=4,
        test_mode=False)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
