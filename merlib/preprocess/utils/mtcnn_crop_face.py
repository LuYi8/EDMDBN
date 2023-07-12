
# refer https://github.com/timesler/facenet-pytorch
# https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb



from facenet_pytorch import MTCNN
import torch
import os
from PIL import Image, ImageDraw

def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """ Draw bounding boxes and facial landmarks. """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([(b[0], b[1]), (b[2], b[3])], 
            outline='red')
        
    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([(p[i] - 1.0, p[i + 5] - 1.0),
                          (p[i] + 1.0, p[i + 5] + 1.0)],
                          outline='blue')
    return img_copy

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('Running on device: {}'.format(device))
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
)
mtcnn=mtcnn.eval()

def crop_face_by_mtcnn(img):
    """
    如果检测成功返回一个标记人脸框的元组(left,top,right,low),否则返回None
    """
    boxes, probs = mtcnn.detect(img) # bixes 是一个长度不小于1的数组或是None
    if boxes is None: return None
    for i, box in enumerate(boxes):
        return box

def crop_face_list(images):
    box=None
    cropped_imgs=[]
    size=None
    for img in images:
        if box is None :
            box=crop_face_by_mtcnn(img)
        crop_img=img.crop(box)
        if size is None:
            size=crop_img.size
        cropped_imgs.append( crop_img.resize(size) )
    return cropped_imgs