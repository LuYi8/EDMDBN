# Copied from:
# https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
# 修改1，对于缺帧的情况，使用相对下标进行图片读取，它是完全兼容之前的
# 修改2，如果由于缺帧导致record.num_frames < (self.num_segments * self.frames_per_segment)，就返回所有帧

import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from typing import List, Union, Tuple, Any
from pathlib import Path
import pandas as pd


class VideoRecord(object):
    """
    Helper class for class VideoFrameDataset. This class
    represents a video sample's metadata.

    Args:
        root_datapath:Path object, the system path to the root folder
                       of the videos.
        row: A list with four or more elements where 
             1) The first element is the path to the video sample's frames excluding the root_datapath prefix 
             2) The second element is the starting frame id of the video
             3) The third element is the  apex frame id of the video
             4) The fourth element is the inclusive ending frame id of the video
             5) The forth element is the label index.
             6) any following elements are labels in the case of multi-label classification
        imagefile_template: The image filename template that video frame files
                    have inside of their video folders as described above.
        sampling_strategy:
            0=>all frames
            1=>apex frame
            2=>onset, apex frames
            3=>onset, apex, offset frames
            4=>frames_per_segment format

    继承VideoRecord,要求imagefile_template出现且只出现一次'{*}',用于之后的正则匹配
    """

    def __init__(self, row: list, root_datapath: Path, imagefile_template: str, sampling_strategy: int):
        self.video_path = root_datapath / row[0]
        self.raw_interval = int(row[1]), int(row[3])
                #这个修改破坏了多标签行为
        self.dataset_name=row[5] if isinstance(imagefile_template,dict) else None
        self.imgs_list = self.prepare_imgs(row, imagefile_template, sampling_strategy)

        self._labels = row[4:5]

        self._row=row

    def prepare_imgs(self, row, imagefile_template, sampling_strategy):
        assert 0 <= sampling_strategy <= 4
        # print(self.dataset_name)
        if isinstance(imagefile_template,dict) and self.dataset_name:
            imagefile_template=imagefile_template[self.dataset_name]
        onset_id, apex_id, offset_id = int(row[1]), int(row[2]), int(row[3])
        if sampling_strategy == 0 or sampling_strategy == 4:
            # all frames
            imgs_list = [img_path for x in range(onset_id, offset_id+1)
                         if (img_path := self.video_path/imagefile_template.format(x)).exists()]
        elif sampling_strategy == 1:
            # only apex
            apex_path: Path = self.video_path / \
                imagefile_template.format(apex_id)
            imgs_list = [apex_path] if img_path.exists() else []
        elif sampling_strategy == 2:
            # onset and apex frames
            onset_path: Path = self.video_path / \
                imagefile_template.format(onset_id)
            apex_path: Path = self.video_path / \
                imagefile_template.format(apex_id)
            imgs_list = [onset_path, apex_path] if apex_path.exists(
            ) and onset_path.exists() else []
        elif sampling_strategy == 3:
            # onset, apex and offset frames
            onset_path: Path = self.video_path / \
                imagefile_template.format(onset_id)
            apex_path: Path = self.video_path / \
                imagefile_template.format(apex_id)
            offset_path: Path = self.video_path / \
                imagefile_template.format(offset_id)
            flag = apex_path.exists() and onset_path.exists() and offset_path.exists()
            if not flag:
                real_exist_imgs=[img_path for x in range(onset_id, offset_id+1)
                        if (img_path := self.video_path/imagefile_template.format(x)).exists()]
                assert len(real_exist_imgs)>=3,real_exist_imgs
                if not offset_path.exists():
                    offset_path: Path = real_exist_imgs[-1]
                if not apex_path.exists():
                    apex_path=real_exist_imgs[len(real_exist_imgs)//2]
                if not onset_path.exists():
                    onset_path: Path = real_exist_imgs[0]
            flag = apex_path.exists() and onset_path.exists() and offset_path.exists()
            imgs_list = [onset_path, apex_path, offset_path] if flag else []

        return imgs_list

    def __repr__(self) -> str:
        return "record path:{},interval:{},label:{}".format(self.path, self.raw_interval, self.label)

    def __len__(self) -> int:
        return self.num_frames

    @property
    def path(self) -> str:
        return self.video_path

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1  # +1 because end frame is inclusive

    @property
    def start_frame(self) -> int:
        return 0

    @property
    def end_frame(self) -> int:
        return len(self.imgs_list)-1

    @property
    def label(self) -> Union[int, List[int]]:
        # just one label_id
        assert len(self._labels) == 1,self._labels
        if len(self._labels) == 1:
            return int(self._labels[0])
        # sample associated with multiple labels
        return [int(label_id) for label_id in self._labels]


class VideoFrameDataset(torch.utils.data.Dataset):
    r"""
    A highly efficient and adaptable dataset class for videos.
    Instead of loading every frame of a video,
    loads x RGB frames of a video (sparse temporal sampling) and evenly
    chooses those frames from start to end of the video, returning
    a list of x PIL images or ``FRAMES x CHANNELS x HEIGHT x WIDTH``
    tensors where FRAMES=x if the ``ImglistToTensor()``
    transform is used.

    More specifically, the frame range [START_FRAME, END_FRAME] is divided into NUM_SEGMENTS
    segments and FRAMES_PER_SEGMENT consecutive frames are taken from each segment.

    Note:
        A demonstration of using this class can be seen
        in ``demo.py``
        https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch

    Note:
        This dataset broadly corresponds to the frame sampling technique
        introduced in ``Temporal Segment Networks`` at ECCV2016
        https://arxiv.org/abs/1608.00859.


    Note:
        This class relies on receiving video data in a structure where
        inside a ``ROOT_DATA`` folder, each video lies in its own folder,
        where each video folder contains the frames of the video as
        individual files with a naming convention such as
        img_001.jpg ... img_059.jpg.
        For enumeration and annotations, this class expects to receive
        the path to a .txt file where each video sample has a row with four
        (or more in the case of multi-label, see README on Github)
        space separated values:
        ``VIDEO_FOLDER_PATH     START_FRAME      END_FRAME      LABEL_INDEX``.
        ``VIDEO_FOLDER_PATH`` is expected to be the path of a video folder
        excluding the ``ROOT_DATA`` prefix. For example, ``ROOT_DATA`` might
        be ``home\data\datasetxyz\videos\``, inside of which a ``VIDEO_FOLDER_PATH``
        might be ``jumping\0052\`` or ``sample1\`` or ``00053\``.

    Args:
        root_path: The root path in which video folders lie.
                   this is ROOT_DATA from the description above.
        annotationfile_path: The .txt annotation file containing
                             one row per video sample as described above.
        num_segments: The number of segments the video should
                      be divided into to sample frames from.
        frames_per_segment: The number of frames that should
                            be loaded per segment. For each segment's
                            frame-range, a random start index or the
                            center is chosen, from which frames_per_segment
                            consecutive frames are loaded.
        imagefile_template: The image filename template that video frame files
                            have inside of their video folders as described above.
        transform: Transform pipeline that receives a list of PIL images/frames.
        sampling_strategy: int, select different data format.
            0=>all frames
            1=>apex frame
            2=>onset, apex frames
            3=>onset, apex, offset frames
            4=>frames_per_segment format
        test_mode: If True, frames are taken from the center of each
                   segment, instead of a random location in each segment.
    """

    def __init__(self,
                 root_path: str,
                 annotationfile_path: str,
                 num_segments: int = 3,
                 frames_per_segment: int = 1,
                 imagefile_template: str = 'img_{:05d}.jpg',
                 transform=None,
                 sampling_strategy: int = 0,
                 test_mode: bool = False,
                 debug: bool =False):
        super(VideoFrameDataset, self).__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.test_mode = test_mode
        self.sampling_strategy = sampling_strategy
        self.debug = debug
        self._parse_annotationfile()
        self._sanity_check_samples()

    def get_labels(self):
        # 用于构造非均衡数据sampler
        return self.labels

    def _load_image(self, record: VideoRecord, idx: int):
        img_path = record.imgs_list[idx]
        img = Image.open(img_path).convert('RGB')
        return img

    def _parse_annotationfile(self):
        # self.video_list = [VideoRecord(x.strip().split(
        # ), self.root_path, self.imagefile_template) for x in open(self.annotationfile_path)]
        df = pd.read_csv(self.annotationfile_path, sep=" ", header=None)
        self.video_list = [VideoRecord(list(x), Path(self.root_path), self.imagefile_template,sampling_strategy=self.sampling_strategy)
                           for x in df.itertuples(index=False, name="MEVideo")]
        self.labels = [x.label for x in self.video_list]

    def _sanity_check_samples(self):
        sanity_check_result = []
        for record in self.video_list:
            if self.sampling_strategy == 0 and len(record) <= 1:
                sanity_check_result.append(
                    f"Dataset Warning: video {record.path} have one frame on disk but in all frames mode!")
            elif self.sampling_strategy in [1, 2, 3] and record.num_frames <= 0:
                sanity_check_result.append(
                    f"Dataset Warning: video {record.path} seems to have onset, apex, or offset frames lost on disk!\n {record._row}")
            elif self.sampling_strategy == 4 and record.num_frames < (self.num_segments * self.frames_per_segment):
                sanity_check_result.append(f"Dataset Warning: video {record.path} has {record.num_frames} frames, but need (num_segments={self.num_segments})*(frames_per_segment={self.frames_per_segment})"
                 )
            elif self.sampling_strategy == 0:
                assert self.num_segments > 1, f"Dataset Warning: video {record.path} have one frame on disk but in all frames mode!"
            elif self.sampling_strategy ==5:
                assert self.num_segments > 1, f"Dataset Warning: video {record.path} have one frame on disk but in all frames mode!"
        
        if len(sanity_check_result)>0 and  self.debug:
            log_path=Path("sanity_check_result.log")
            pd.Series(sanity_check_result).to_csv(log_path)
            raise Exception("data fail to pass sanity check, please look at  {}".format(log_path.resolve()))
        else:
            print("data sanity check successfully!")
    def _get_start_indices(self, record: VideoRecord) -> 'np.ndarray[int]':
        """
        For each segment, choose a start index from where frames
        are to be loaded from.

        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            List of indices of where the frames of each
            segment are to be loaded from.
        """
        # choose start indices that are perfectly evenly spread across the video frames.
        if self.test_mode:
            distance_between_indices = (
                record.num_frames - self.frames_per_segment + 1) / float(self.num_segments)

            start_indices = np.array([int(distance_between_indices / 2.0 + distance_between_indices * x)
                                      for x in range(self.num_segments)])
        # randomly sample start indices that are approximately evenly spread across the video frames.
        else:
            max_valid_start_index = (
                record.num_frames - self.frames_per_segment + 1) // self.num_segments

            start_indices = np.multiply(list(range(self.num_segments)), max_valid_start_index) + \
                np.random.randint(max_valid_start_index,
                                  size=self.num_segments)

        return start_indices

    def _get_video_segments(self, record: VideoRecord):
        """
        Loads the frames of a video at the corresponding indices.

        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            a list of PIL images loaded at the corresponding indices.
        """
        if record.num_frames<=self.frames_per_segment *self.num_segments:
            return [self._load_image(record, x) for x in range(
                record.start_frame, record.start_frame+len(record))]

        frame_start_indices: 'np.ndarray[int]' = self._get_start_indices(
            record)

        frame_start_indices = frame_start_indices + record.start_frame
        images = list()

        # from each start_index, load self.frames_per_segment
        # consecutive frames
        for start_index in frame_start_indices:
            frame_index = int(start_index)

            # load self.frames_per_segment consecutive frames
            for _ in range(self.frames_per_segment):
                image = self._load_image(record, frame_index)
                images.append(image)

                if frame_index < record.end_frame:
                    frame_index += 1
        return images

    def __getitem__(self, idx: int) -> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple['torch.Tensor[num_frames, channels, height, width]',
              Union[int, List[int]]],
        Tuple[Any, Union[int, List[int]]],
    ]:
        """
        For video with id idx, loads self.NUM_SEGMENTS * self.FRAMES_PER_SEGMENT
        frames from evenly chosen locations across the video.

        Args:
            idx: Video sample index.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
            if the transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """
        record: VideoRecord = self.video_list[idx]

        return self._get_sample(record)

    def _get_sample(self, record: VideoRecord) -> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple['torch.Tensor[num_frames, channels, height, width]',
              Union[int, List[int]]],
        Tuple[Any, Union[int, List[int]]]
    ]:
        """
        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 
            1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1], if the transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """
        if self.sampling_strategy == 4 :
            images = self._get_video_segments(record)
        else:
            images = [self._load_image(record, x) for x in range(
                record.start_frame, record.start_frame+len(record))]
        try:
            if self.transform == 'default':
                self.transform=ImglistToTensor()

            if self.transform is not None:
                images = self.transform(images)
            # assert isinstance(images, torch.Tensor), type(images)
        except:
            print("{} can't be transformed successfully! len(images) {}".format(
                record, len(images)))
        assert len(images)>0,record
        return images, record.label

    def __len__(self):
        return len(self.video_list)


class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``VideoFrameDataset``.
    """
    @staticmethod
    def forward(img_list: List[Image.Image]) -> 'torch.Tensor[NUM_IMAGES, CHANNELS, HEIGHT, WIDTH]':
        """

        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.

        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        _ = [transforms.functional.to_tensor(pic) for pic in img_list]
        return torch.stack(_)

class ImglistToNumpy(torch.nn.Module):
    @staticmethod
    def forward(img_list: List[Image.Image]):
        """

        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.

        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        return [np.array(pic) for pic in img_list]
