import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from einops import asnumpy, rearrange
from .tim import tim


def TAS(indexes: Tuple[int], *, N_k=7) -> Tuple[int]:
    def travel_BST_by_layer(bst_arr, N):
        result = []
        bst_arr = deque([list(bst_arr)])
        while bst_arr:
            for i in range(len(bst_arr)):
                if len(result) >= N:
                    return result
                x = bst_arr.popleft()
                mid_idx = len(x) // 2
                mid_el = x[mid_idx]
                result.append(mid_el)
                if x[:mid_idx]:
                    bst_arr.append(x[:mid_idx])
                if x[mid_idx + 1:]:
                    bst_arr.append(x[mid_idx + 1:])
        return result

    final_indexes = [x for x in indexes]
    t_s, t_p, t_e = indexes
    N = t_e - t_s + 1
    assert N_k <= N, "the length of generating keyframes are larger than the length of the frames sequence"
    N_sp = (t_p - t_s + 1) / N * N_k
    N_sp = int(N_sp)
    N_pe = N_k - N_sp
    # print(N_sp,N_pe)

    final_indexes.extend(travel_BST_by_layer(list(range(t_s + 1, t_p)), N_sp))

    final_indexes.extend(travel_BST_by_layer(list(range(t_p + 1, t_e)), N_pe))
    final_indexes.sort()
    return tuple(final_indexes)


class FixSequenceTIM:
    def __init__(self, sequence_length: int) -> None:
        """
        Args:
            sequence_length(int): desired sequence length
        """
        self.sequence_length = sequence_length

    def __call__(self, x: torch.ByteTensor) -> torch.Tensor:
        """
        Args:
            x(torch.ByteTensor): shape is (S, C, H, W)
        Return:
            torch byte tensor of desired length's video. Shape is (S_desired, C, H, W)
        """
        with torch.no_grad():
            if x.shape[0] == self.sequence_length:
                return x
            else:
                x = x.numpy().astype(np.float32)
                x = tim(x, self.sequence_length)
                x = np.where(x <= 255, x, 255)
                x = np.where(x >= 0, x, 0)
                x = x.astype(np.uint8)
                return torch.from_numpy(x)


class FixSequenceInterpolation:
    def __init__(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length

    def __call__(self, x) -> None:
        """
        Arrgs:
            x(torch.ByteTensor): shape is (S, C, H, W)
        Return:
            torch byte tensor of desired length's video. Shape is (S_desired, C, H, W)
        """
        with torch.no_grad():
            s, _, h, w = x.shape
            if s == self.sequence_length:
                return x
            else:
                x = x.float()
                x = rearrange(x, "s c h w -> 1 c s h w")
                x = F.interpolate(
                    x, size=(self.sequence_length, h, w), mode="trilinear", align_corners=False
                )
                x = rearrange(x, "1 c s h w -> s c h w")
                # x = x.byte()
                return x

    def __str__(self):
        return str(self.sequence_length)


if __name__ == '__main__':
    tas_frames = np.random.randint(0, 256, (3, 224, 224, 3), np.uint8)
    tas_indexes = [69, 123, 123]

    len(TAS(tas_indexes, N_k=13))
