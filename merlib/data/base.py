########################
# 此文件中定义了数据的元信息，用于锁定数据集相关信息。例如：
#   name = 'casme2' 对应的数据集名称，将被多次使用
#   unique=True 是否使用唯一路径，使用唯一路径而非原路径的好处有两点：
#       1. 避免数据集中两个微表情存在交叉的情况，不方便进行离线处理。
#       2. 方便debug时，快速确定有问题的视频路径。
#   imagefile_template = 'img{}.jpg' 数据集中图片的命名格式
#   raw_data_path='/home/neu-wang/public/CASME2/CASME2-RAW/' 原数据集路径
#       1. 当unique=False时，从原数据集路径提取数据，此时，数据路径一般是唯一的，且是被预处理好的。
#       2. 当unique=True时，从源数据集提取数据并进行剪裁等处理，进而构造唯一路径
#   savepath= "/home/neu-wang/mby/database/CASME2" 离线处理时，处理后数据默认保存路径、anotations文件默认保存路径
#   use_crop=True 是否进行剪裁
#   use_align=False 是否进行对齐

########################

import inspect
import sys
from dataclasses import dataclass, is_dataclass
from pathlib import Path, PurePath

DATASET_CHOICES = []
ME_DATASET = {}


@dataclass
class SMICHS_info:
    """SMIC-HS数据集源信息
    """
    name = 'smic-hs'
    unique = False
    imagefile_template = "reg_image{:06d}.bmp"
    doc_path = "/home/neu-wang/mby/database/SMIC-HS/SMIC-HS-E_annotation_orig.xlsx"
    save_path = "/home/neu-wang/mby/database/SMIC-HS/"
    raw_data_path = '/home/neu-wang/public/SMIC/SMIC_all_cropped/HS/'

    def __post_init__(self):
        self.doc_path = Path(self.doc_path)
        self.save_path = Path(self.save_path)
        self.raw_data_path = Path(self.raw_data_path)


@dataclass
class SMICHSE_info:
    """SMIC-HS-E数据集源信息
    """
    name = 'smic-hs-e'
    fps = 100
    unique = True
    imagefile_template = "image{:06d}.jpg"
    doc_path = "/home/neu-wang/public/SMIC/SMIC-E_raw_image/HS_long/SMIC-HS-E_annotation_2019.xlsx"
    save_path = "/home/neu-wang/mby/database/SMIC-E-HS/"
    raw_data_path = '/home/neu-wang/public/SMIC/SMIC-E_raw_image/HS_long/SMIC-HS-E'

    def __post_init__(self):
        self.doc_path = Path(self.doc_path)
        self.save_path = Path(self.save_path)
        self.raw_data_path = Path(self.raw_data_path)
    @staticmethod
    def get_labels(num_classes=3):
        assert num_classes in [3]
        if num_classes == 3:
            return ThreeEmotion.emotions

@dataclass
class CASME2_info:
    """ CASME2数据集元信息
    """
    name = 'casme2'
    unique = True
    imagefile_template = 'img{}.jpg'
    fps = 200
    doc_path = "/home/neu-wang/public/CASME2/CASME2-coding-20190701.xlsx"
    raw_data_path = '/home/neu-wang/public/CASME2/CASME2-RAW/'
    save_path = "/home/neu-wang/mby/database/CASME2"

    def __post_init__(self):
        self.doc_path = Path(self.doc_path)
        self.save_path = Path(self.save_path)
        self.raw_data_path = Path(self.raw_data_path)
    @staticmethod
    def get_labels(num_classes=3):
        assert num_classes in [3,5]
        if num_classes == 3:
            return ThreeEmotion.emotions
        elif num_classes == 5:
            return ['happiness', 'surprise', 'disgust', 'repression', 'others']
        else:
            return []

@dataclass
class SAMM_info:
    """SAMM微表情数据集元信息
    注意原本图片命名格式为{subject_id}_{frame_id}.jpg,为了方便读取，先将所有图片命名改为
    {frame_id}.jpg，如"009_3577.jpg=>3577.jpg",因此最终的imagefile_template = '{}.jpg'
    """
    name = 'samm'
    unique = True
    fps = 200
    imagefile_template = '{}.jpg'
    doc_path = "/home/neu-wang/public/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx"
    raw_data_path = '/home/neu-wang/public/SAMM/SAMM_clip/'
    save_path = "/home/neu-wang/mby/database/SAMM/"

    def __post_init__(self):
        self.doc_path = Path(self.doc_path)
        self.save_path = Path(self.save_path)
        self.raw_data_path = Path(self.raw_data_path)

    @staticmethod
    def get_labels(num_classes=3):
        assert num_classes in [3,5]
        if num_classes == 3:
            return ThreeEmotion.emotions
        elif num_classes == 5:
            return ['happiness', 'surprise', 'contempt', 'anger', 'other']
        else:
            return []

@dataclass
class MMEW_info:
    name = 'mmew'
    unique = True
    fps = 90
    imagefile_template = '{}.jpg'
    doc_path = "/home/neu-wang/mby/database/MMEW/MMEW_Micro_Exp_v2.xlsx"
    raw_data_path = '/home/neu-wang/mby/database/MMEW/Micro_Expression'
    save_path = "/home/neu-wang/mby/database/MMEW"

    def __post_init__(self):
        self.doc_path = Path(self.doc_path)
        self.save_path = Path(self.save_path)
        self.raw_data_path = Path(self.raw_data_path)

    @staticmethod
    def get_labels(num_classes=3):
        assert num_classes in [3,4]
        if num_classes == 3:
            return ThreeEmotion.emotions
        elif num_classes == 4:
            return ['happiness', 'surprise', 'disgust',  'others']
        else:
            return []
        
@dataclass
class CASME3PartA_info:
    name = 'casme3-partA'
    raw_data_path = '/data/CASME3/part_A/'
    doc_path = "/home/neu-wang/mby/database/CASME3-PartA/CAS-ME-3-supply-update_2023_3_3/annotation/partA/cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx"
    save_path = "/home/neu-wang/mby/database/CASME3-PartA/"
    imagefile_template = '{}.jpg'
    unique = True

    def __post_init__(self):
        self.doc_path = Path(self.doc_path)
        self.save_path = Path(self.save_path)
        self.raw_data_path = Path(self.raw_data_path)


@dataclass
class CASME3UL_info:
    name = 'casme3-ul'
    raw_data_path = '/home/neu-wang/public/CASME3/part_A/'
    doc_path = "/home/neu-wang/mby/database/CASME3-UL/CAS(ME)3_part_A_v2.xls"
    save_path = "/home/neu-wang/mby/database/CASME3-UL/"
    imagefile_template = '{}.jpg'

    def __post_init__(self):
        self.doc_path = Path(self.doc_path)
        self.save_path = Path(self.save_path)
        self.raw_data_path = Path(self.raw_data_path)


@dataclass
class FPS200CD2:
    """
    casme2-samm-mmew 联合数据库
    """
    name = 'fps200cd2'
    doc_path = {
        CASME2_info.name: CASME2_info().doc_path,
        SAMM_info.name: SAMM_info().doc_path,
    }
    raw_data_path = None
    save_path = "/home/neu-wang/mby/database/FPS200CD2"
    imagefile_template = {
        CASME2_info.name: CASME2_info().imagefile_template,
        SAMM_info.name: SAMM_info().imagefile_template,
    }
    unique = None

    def __post_init__(self):
        self.save_path = Path(self.save_path)
        self.raw_data_path = None


@dataclass
class CSMCD:
    """
    casme2-samm-mmew 联合数据库
    """
    name = 'csmcd'
    doc_path = {
        CASME2_info.name: CASME2_info().doc_path,
        MMEW_info.name: MMEW_info().doc_path,
        SAMM_info.name: SAMM_info().doc_path,
    }
    raw_data_path = None
    save_path = "/home/neu-wang/mby/database/CSMCD"
    imagefile_template = {
        CASME2_info.name: CASME2_info().imagefile_template,
        MMEW_info.name: MMEW_info().imagefile_template,
        SAMM_info.name: SAMM_info().imagefile_template,
    }
    unique = None

    def __post_init__(self):
        self.save_path = Path(self.save_path)
        self.raw_data_path = None


@dataclass
class FPS200CD:
    """
    casme2-samm-mmew 联合数据库，然而后面mmew帧率并非200，这个命名成为历史，新的命名是CSMCD
    """
    name = 'fps200cd'
    doc_path = {
        CASME2_info.name: CASME2_info().doc_path,
        MMEW_info.name: MMEW_info().doc_path,
        SAMM_info.name: SAMM_info().doc_path,
    }
    raw_data_path = None
    save_path = "/home/neu-wang/mby/database/FPS200CD"
    imagefile_template = {
        CASME2_info.name: CASME2_info().imagefile_template,
        MMEW_info.name: MMEW_info().imagefile_template,
        SAMM_info.name: SAMM_info().imagefile_template,
    }
    unique = None

    def __post_init__(self):
        self.save_path = Path(self.save_path)
        self.raw_data_path = None


@dataclass
class MEGC2019CD:
    """
    casme2-smic_hs-samm 联合数据库
    """
    name = 'megc2019cd'
    doc_path = {
        CASME2_info.name: CASME2_info().doc_path,
        SMICHS_info.name: SMICHS_info().doc_path,
        SAMM_info.name: SAMM_info().doc_path,
    }
    raw_data_path = None
    save_path = "/home/neu-wang/mby/database/MEGC2019CD"
    imagefile_template = {
        CASME2_info.name: CASME2_info().imagefile_template,
        SMICHS_info.name: SMICHS_info().imagefile_template,
        SAMM_info.name: SAMM_info().imagefile_template,
    }
    unique = None

    def __post_init__(self):
        self.save_path = Path(self.save_path)
        self.raw_data_path = None


class FiveEmotion:
    emotions = {
        CASME2_info.name: CASME2_info.get_labels(5),
        SAMM_info.name:SAMM_info.get_labels(5),
        MMEW_info.name:MMEW_info.get_labels(4)
    }

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        assert dataset_name in FiveEmotion.emotions

        pass

    def transform(self, x: str):
        other_nickname = ['other', 'others']
        x = x.strip().lower()
        if x in FiveEmotion.emotions[self.dataset_name]:
            return x
        else:
            return None

    def str2id(self, x: str):
        try:
            x = x.strip().lower()
        except AttributeError as e:
            print(x)
            raise e
        return FiveEmotion.emotions[self.dataset_name].index(x)

    def id2str(self, x: int):
        x = int(x)
        return FiveEmotion.emotions[self.dataset_name][x]


class ThreeEmotion:
    emotions = ['surprise', 'positive', 'negative']

    def __init__(self):
        pass

    @staticmethod
    def transform(x: str):
        surprise_nickname = ['surprise']
        positive_nickname = ['happiness', 'happy', 'positive']
        other_nickname = ['other', 'others']
        x = x.strip().lower()
        if x in surprise_nickname:
            return ThreeEmotion.emotions[0]
        elif x in positive_nickname:
            return ThreeEmotion.emotions[1]
        elif x in other_nickname:
            return None
        else:
            return ThreeEmotion.emotions[2]

    @staticmethod
    def str2id(x: str):
        x = x.strip().lower()
        return ThreeEmotion.emotions.index(x)

    @staticmethod
    def id2str(x: int):
        x = int(x)
        return ThreeEmotion.emotions[x]


for name, class_ in inspect.getmembers(sys.modules[__name__], is_dataclass):
    x = class_()
    ME_DATASET[x.name] = x
    DATASET_CHOICES.append(x.name)
