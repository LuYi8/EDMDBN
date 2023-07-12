import torchvision.transforms as T
from .utils.optical_flow import getOFAndGray,getOFVideo
from .utils.extract_keyframes import FixSequenceInterpolation
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
import torch
from torchvision.transforms import InterpolationMode
from pytorchvideo.transforms import UniformTemporalSubsample
from merlib.preprocess.utils.mtcnn_crop_face import crop_face_list
import typing
from PIL import Image

class TimeInterpolation(nn.Module):
    def __init__(self,num_frames,interpolation_strategy) -> None:
        super().__init__()
        time_transform=nn.Identity()
        if interpolation_strategy == 'UniformTemporalSubsample':
            time_transform=UniformTemporalSubsample(num_frames, temporal_dim=-4) 
        elif interpolation_strategy == 'FixSequenceInterpolation':
            time_transform=FixSequenceInterpolation(num_frames)  # type: ignore
        
        self.time_transform=time_transform
    def forward(self,x):
        """
        x: B, C, H, W
        """
        return self.time_transform(x)

class ImglistToTensor(torch.nn.Module):
    """
    trans_class [ T.ToTensor or T.PILToTensor]

    Converts a list of PIL images in the range [0,255] to a torch.TensorFloat
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1] or [0-255].
    Can be used as first transform for ``VideoFrameDataset``.
    """
    def __init__(self,trans_class=T.ToTensor) -> None:
        # T.PILToTensor
        super().__init__()
        self.trans_func=trans_class()
        pass

    def forward(self,img_list: typing.List[Image.Image]) -> 'torch.Tensor[NUM_IMAGES, CHANNELS, HEIGHT, WIDTH]':
        """

        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.

        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        # T.ToTensor()(img).dtype,T.PILToTensor()(img).dtype 分别对应 (torch.float32, torch.uint8)
        return torch.stack([self.trans_func(pic) for pic in img_list] )

class ConvertBCHWtoCBHW(nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)"""

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)

class CropFaceList:
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        """
        x: PIL.Image list
        return: PIL.Image list
        """
        assert len(x) != 0
        y = crop_face_list(x)
        assert len(y) == len(x) and len(y) > 0, "{}".format(len(y))
        return y

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class OpticalFlow:
    def __init__(self,static=False) -> None:
        self.static_of=static
        pass

    def __call__(self, x):
        """
        Calculate optical flow from video.
        Args:
        video(torch.Tensor): shape is (S, C, H, W)
        return video optical flow: shape is (S, C, H, W)
        """
        y = getOFVideo(x,static=self.static_of)
        return y

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        alpha=4
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

def get_me_transform(input_size, train, num_samples, optical_flow, sample_strategy, scale,registration_method):
    """
    用于有监督时的微表情训练，测试
    """
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    # input_size=(input_size,input_size)
    me_transform = [ImglistToTensor()]
    # size=(256,256)
    if train:
        me_transform.extend([
            T.Resize(size, interpolation=InterpolationMode.BICUBIC),  # type: ignore
            T.CenterCrop(input_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomResizedCrop(input_size, scale=scale,ratio=(0.9,1.1),interpolation=InterpolationMode.BICUBIC),
            # T.RandomRotation(5,interpolation=InterpolationMode.BICUBIC),
        
        ])  # type: ignore
    else:
        me_transform.extend([
            T.Resize(size, interpolation=InterpolationMode.BICUBIC),  # type: ignore
            T.CenterCrop(input_size)# type: ignore
        ])  

    num_samples += 1 if optical_flow else 0
    if sample_strategy == 'UniformTemporalSubsample':
        me_transform.append(UniformTemporalSubsample(
            num_samples, temporal_dim=-4))  # type: ignore
    elif sample_strategy == 'FixSequenceInterpolation':
        me_transform.append(FixSequenceInterpolation(num_samples))  # type: ignore
    else:
        assert False

    if optical_flow:
        me_transform.append(
            OpticalFlow()  # type: ignore
        )
    else:
        me_transform.append(
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)  # type: ignore
        )

    me_transform.append(ConvertBCHWtoCBHW())  # type: ignore
    return T.Compose(me_transform)


def get_mae_transform(input_size, num_frames=10):
    """
    用于自掩码训练的transform
    """
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    mae_transform = T.Compose([
        ImglistToTensor(),
        # T.Resize(size, interpolation=InterpolationMode.BICUBIC),
        T.RandomResizedCrop(size=input_size, scale=(0.7, 1),
                            interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ConvertBCHWtoCBHW(),
        UniformTemporalSubsample(num_samples=num_frames)
    ])
    return mae_transform


def build_transform(is_train, args):
    """
    利用timm的transforms
    """
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        # to maintain same ratio w.r.t. 224 images
        T.Resize(size, interpolation=InterpolationMode.BICUBIC),
    )
    t.append(T.CenterCrop(args.input_size))

    t.append(T.ToTensor())
    t.append(T.Normalize(mean, std))
    return T.Compose(t)
