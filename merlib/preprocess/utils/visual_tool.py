from PIL import Image, ImageDraw
from typing import Tuple,Union
import dlib
import numpy as np

from merlib.data import DLIB_FACE_LANDMARK_PATH
 

def rect_to_tuple(rect):
    """
    rect:dlib.rectangle: left, top, right, bottom
    """
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom

def dlib_landmarks2tuple(landmarks):
    """
    landmarks:dlib.full_object_detection
    """
    result=[]
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        result.append((x,y))
    return tuple(result)

def visualize_bbox_landmarks(img:Image.Image, bbox:Union[Tuple[float,float,float,float],None],landmarks:Union[Tuple[Tuple[float, float], ...],None],*,names=None,show_vis=True):
    """
    bbox:一个包含四个元素的元组 (x0, y0, x1, y1)，其中 (x0, y0) 是矩形左上角的坐标，(x1, y1) 是矩形右下角的坐标。
    landmarks:二维 Numpy 数组，其形状为 (68, 2)，表示 68 个特征点的横纵坐标。每个特征点都是一个 (x, y) 坐标对，其中 x 表示横坐标，y 表示纵坐标。
        如果landmarks存在，则进行特征点绘制。
    """
    # 加载图像
    image = img.convert("RGB")
    # 创建绘制对象
    draw = ImageDraw.Draw(image)
    # 定义字体大小
    font_size = 60
    # from PIL import ImageFont
    # 加载默认字体并调整大小
    # font = ImageFont.load_default().font_variant(size=font_size)
    # 绘制矩形框
    if bbox is not None:
        draw.rectangle(bbox, outline="green")
    if landmarks is not None:
        if names is None:
            names=[None]*len(landmarks)
        for (x, y),name in zip(landmarks,names):
            draw.ellipse((x-2, y-2, x+2, y+2), fill='red')
            if name:
                draw.text((x, y-15), str(name), fill='blue',font_size=font_size)
    # 显示图像
    if show_vis:
        image.show()
    return image

def visualize_dlib_bbox(img:Image.Image,show_landmarks=True,show_vis=True,*,show_bbox=True,landmarks_indexes=None,names=None):
    detector = dlib.get_frontal_face_detector()
    rects = detector(np.array(img), 1)
    if show_bbox:
        bbox=rect_to_tuple(rects[0])
    else:
        bbox=None
    if show_landmarks:
        predictor = dlib.shape_predictor(DLIB_FACE_LANDMARK_PATH)
        landmarks = predictor(np.array(img), rects[0])
        landmarks=dlib_landmarks2tuple(landmarks)
        landmarks=np.array(landmarks)
        if landmarks_indexes:
            landmarks=landmarks[landmarks_indexes]
        
        return visualize_bbox_landmarks(img,bbox,tuple(landmarks),show_vis=show_vis,names=names)

    return visualize_bbox_landmarks(img,bbox,None,show_vis=show_vis,names=names)
# def visualize_landmarks

from pathlib import Path
import typing
from collections import defaultdict
import random
def sample_precessed_data(data_root:typing.Union[Path,str],template="*.jpg"):
    data_root=Path(data_root)
    all_imgs=data_root.rglob(template)

    def check_hidden_files(file_path):
        for part in file_path.parts:
            if part.startswith('.'):
                return True
        return False
    all_imgs=list(filter(lambda x: not check_hidden_files(x), all_imgs))
    all_imgs=sorted(all_imgs)
    img_dir_dict=defaultdict(list)
    for x in all_imgs:
        img_dir_dict[x.parent].append(x)
    
    res=[]
    for x in img_dir_dict:
        imgs=img_dir_dict[x]
        rand_img_path=random.choice(imgs)
        flag=rand_img_path.relative_to(data_root).parent
        img=Image.open(rand_img_path)
        res.append((img,flag))

    return [x[0] for x in res],[x[1] for x in res]
