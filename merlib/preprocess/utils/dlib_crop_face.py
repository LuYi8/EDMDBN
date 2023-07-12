import dlib
from imutils import face_utils
import cv2 as cv
import numpy as np
from PIL import Image
from merlib.data import DLIB_FACE_LANDMARK_PATH

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(DLIB_FACE_LANDMARK_PATH))

def get_landmarks(img):
    face_rects = detector(img, 0)
    for index, face in enumerate(face_rects):
        shape = predictor(img, face_rects[index])
        shape = face_utils.shape_to_np(shape)
        # for (x, y) in shape:
        #     cv.circle(img, (x, y), 1, (0, 0, 255), -1)
        return shape

def xs_ys(img_RGB):
    img_gray = cv.cvtColor(img_RGB, cv.COLOR_RGB2GRAY)
    xs = []
    ys = []
    rects = detector(img_gray, 0)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img_RGB,rects[i]).parts()])

        for idx, point in enumerate(landmarks):
            xs.append(point[0, 0])
            ys.append(point[0, 1])
    return xs, ys


def crop_once(img_RGB):
    xs, ys = xs_ys(img_RGB)
    if len (xs)!= 0:
        ymin = min(ys)-(ys[36]-ys[18])//2
        # ymin = min(ys)-(ys[37]-ys[19])
        ymax = max(ys)
        xmin = min(xs)
        xmax = max(xs)
        img_crop_RGB = img_RGB[max(ymin,0):min(ymax,img_RGB.shape[0]-1), 
                               max(xmin,0):min(xmax,img_RGB.shape[1]-1)]
    else:
        ymin = ymax = xmin = xmax = 0
        img_crop_RGB = img_RGB
    return img_crop_RGB, ymin, ymax, xmin, xmax


def max_more(img_crop_RGB, ymin_temp, ymax):
    img_crop2, ymin2, ymax2, xmin2, xmax2 = crop_once(img_crop_RGB)
    if ymax2 != 0:
        ymax = min(ymax, ymax2 + ymin_temp)
    return img_crop2, ymax, ymin2


def crop_times(img_RGB, times=2):
    img_crop_RGB, ymin, ymax, xmin, xmax = crop_once(img_RGB)

    ymin_temp = ymin
    for i in range(times-1):
        img_crop_RGB, ymax, ymin_new = max_more(img_crop_RGB, ymin_temp, ymax)
        ymin_temp = ymin_temp + max(ymin_new,0)

    return ymin, ymax, xmin, xmax

def dlib_crop_twice(rgb_img:Image.Image,times=2,*,img_path=None):
    """
    输入PIL.Image rgb图像, 如果检测成功，返回一个box,对应着（left, top, right, low)
    否则返回None
    """
    rgb_img = np.array(rgb_img)
    ymin, ymax, xmin, xmax =crop_times(rgb_img,times=times)
    if ymin==0 and ymax==0 and xmin==0 and xmax==0:
        return None
    return (xmin,ymin,xmax,ymax)

import openface
align = openface.AlignDlib(DLIB_FACE_LANDMARK_PATH)
def dlib_crop_more_ROI(rgb_img:Image.Image,*,img_path=None):
    """
    输入PIL.Image rgb图像, 如果检测成功，返回一个box,对应着（left, top, right, low)
    否则返回None
    """
    np_img=np.array(rgb_img)
    # landmarks=get_landmarks(np_img)
    
    face_bb=align.getLargestFaceBoundingBox(np_img)
    assert face_bb is not None,img_path
    landmarks=align.findLandmarks(np_img,face_bb)
    landmarks=np.array(landmarks)
    assert landmarks is not None,img_path
    face_right=np.max(landmarks[:,0])
    face_left=np.min(landmarks[:,0])
    face_top=np.min(landmarks[:,1])
    face_bottom=np.max(landmarks[:,1])
    w,h=rgb_img.size
    face_left=max(0,face_left)
    face_top=max(0,face_top)
    face_right=min(rgb_img.size[0],face_right)
    face_bottom=min(h,face_bottom)

    face_ROI_left=landmarks[17][0]
    face_ROI_top=max(face_top-10,0)
    face_ROI_right=landmarks[26][0]
    face_ROI_bottom=(landmarks[5][1]+landmarks[11][1])//2
    A=(face_ROI_left,face_ROI_top)
    B=(face_ROI_right,face_ROI_bottom)
    # rgb_img.crop([*A,*B])

    return [*A,*B]

def dlib_crop_face_v3(rgb_img:Image.Image,*,img_path=None,crop_twice=False):
    """
    输入PIL.Image rgb图像, 如果检测成功，返回一个box,对应着（left, top, right, low)
    否则返回None
    相比dlib_crop_more,将保护人脸眉毛、下巴等器官，相比dlib_crop_twice，将去除更多的非ROI区域。
    """
    np_img=np.array(rgb_img)
    
    face_bb=align.getLargestFaceBoundingBox(np_img)
    assert face_bb is not None,img_path
    landmarks=align.findLandmarks(np_img,face_bb)
    landmarks=np.array(landmarks)
    assert landmarks is not None,img_path

    # 获取人脸外侧最大范围边界
    face_right=np.max(landmarks[:,0])
    face_left=np.min(landmarks[:,0])
    face_top=np.min(landmarks[:,1])
    face_bottom=np.max(landmarks[:,1])

    #保证不超出图形范围
    w,h=rgb_img.size
    face_left=max(0,face_left)
    face_top=max(0,face_top)
    face_right=min(w,face_right)
    face_bottom=min(h,face_bottom)

    # 获取感兴趣区域
    
    # 根据左眼和左眉毛上特征点获取感兴趣人脸左边界,保留眉毛和眼睛
    face_ROI_left=min(landmarks[17:22,0].tolist()+landmarks[36:42,0].tolist())
    # 根据右眼和右眉毛上特征点获取感兴趣人脸右边界
    face_ROI_right=max(landmarks[22:27,0].tolist()+landmarks[42:48,0].tolist())
    # 根据下巴的最下方三个坐标，确定下边界，保留下巴
    face_ROI_bottom=sum(landmarks[7:10,1].tolist())/len(landmarks[7:10,1])

    # 计算顶点
    # 1.根据眉毛坐标和27下标点，保留眉毛和部分额头
    # face_ROI_top= face_top - np.abs(landmarks[27,1]-np.mean(landmarks[17:27,1]))/2
    # 2.计算左外眼点和右外眼点的距离a，及在y坐标上的均值，向上扩招0.3a距离作为上顶点，参考Deep3DCANN
    # 假设左眼和右眼的坐标如下：
    left_eye ,right_eye= landmarks[36], landmarks[45]

    # 计算两眼之间的距离
    a = np.linalg.norm(left_eye - right_eye)

    # 计算两眼的y坐标均值
    mean_y = np.mean([left_eye[1], right_eye[1]])

    # 计算上顶点的y坐标
    # 发现对应samm 0.3距离有些人可能仅仅到眉毛上方，0.5略多余，因此选择0.45，基本兼顾所有人在眉毛上方有额外肌肤留白
    face_ROI_top = mean_y - 0.45 * a

    #防止超出图形范围
    face_ROI_left=max(face_ROI_left,face_left)
    face_ROI_right=min(face_ROI_right,face_right)
    face_ROI_bottom=min(face_ROI_bottom,face_bottom)
    face_ROI_top=max(face_ROI_top,0) # face_top只能达到眉毛上边界，然而这里我们保留了一部分额头信息

    A=(face_ROI_left,face_ROI_top)
    B=(face_ROI_right,face_ROI_bottom)

    if not crop_twice:
        return [*A,*B]

    # 为了解决有些人脸在剪裁时，下巴没有准确识别，因此，在上次剪裁的基础上再次剪裁
    ymin, ymax, xmin, xmax=face_ROI_top,face_ROI_bottom,face_ROI_left,face_ROI_right
    img_crop_RGB=rgb_img.crop([*A,*B])
    ymin_temp = ymin
    
    img_crop_RGB, ymax, ymin_new= max_more(np.array(img_crop_RGB), ymin_temp, ymax)
    return [*A,face_ROI_right,ymax]

