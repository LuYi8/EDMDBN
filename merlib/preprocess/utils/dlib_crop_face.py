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

def dlib_crop_twice(rgb_img:Image.Image,times=2):
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
    rgb_img.crop([*A,*B])

    return [*A,*B]