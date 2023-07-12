import cv2
import numpy as np
import torch
from einops import asnumpy, rearrange

__all__ = ['getOFVideo']


def rgb2gray(func):
    """
    Decorator for converting RGB numpy array to grayscale numpy array.
    If RGB array is input, (H, W, 3) is converted to (H, W, 1).
    If grayscale array is input, (H, W) is converted to (H, W, 1).
    """

    def rgb2gray(*args, **kwargs):
        newargs = []
        for data in args:
            if data.shape[-1] == 3:
                newargs.append(cv2.cvtColor(data, cv2.COLOR_RGB2GRAY))
            else:
                newargs.append(data.squeeze(-1))
        return func(*newargs, **kwargs)

    return rgb2gray


@rgb2gray
def calcOpticalFlow(prev: np.array, curr: np.array):
    """
    Calculate optical flow from two consecutive frames.
    Args:
        prev(np.array): shape is (H, W, 1), type is uint8 np array
        curr(np.array): shape is (H, W, 1), type is uint8 np array
    Return:
        np array of optical flow. Shape is (H, W, 3), type is float32
    """
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    dx = flow[..., 0]
    dy = flow[..., 1]
    # mag = np.sqrt(dx ** 2 + dy ** 2)
    opt = np.stack([dx, dy], axis=-1)
    # opt = np.stack([dx, dy,mag], axis=-1)
    return opt

@rgb2gray
def calcOpticalFlowAndGray(prev: np.array, curr: np.array):
    """
    Calculate optical flow from two consecutive frames.
    Args:
        prev(np.array): shape is (H, W, 1), type is uint8 np array
        curr(np.array): shape is (H, W, 1), type is uint8 np array
    Return:
        np array of optical flow. Shape is (H, W, 3), type is float32
    """
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    dx = flow[..., 0]
    dy = flow[..., 1]
    opt = np.stack([dx, dy,prev], axis=-1)
    return opt

def getOFVideo(video: torch.Tensor,static=False) -> torch.Tensor:
    """
    Calculate optical flow from video.
    Args:
        video(torch.Tensor): shape is (S, C, H, W)，dtype is float32
    """
    OF = []
    video = asnumpy(rearrange(video, "s c h w -> s h w c"))
    assert video.dtype==np.uint8, video.dtype
    prev = video[0]
    for frame in video[1:]:
        OF.append(calcOpticalFlow(prev, frame))
        if not static:
            prev = frame
    OF = torch.from_numpy(rearrange(OF, "s h w c -> s c h w"))
    return OF

def getOFAndGray(video: torch.Tensor,static=True) -> torch.Tensor:
    video = asnumpy(rearrange(video, "s c h w -> s h w c"))
    
    OF = [    ]
    prev = video[0]
    for frame in video[1:]:
        OF.append(calcOpticalFlowAndGray(prev, frame))
        if not static:
            prev = frame
    OF = torch.from_numpy(rearrange(OF, "s h w c -> s c h w"))
    return OF