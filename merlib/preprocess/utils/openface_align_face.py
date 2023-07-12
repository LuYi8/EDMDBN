import numpy as np
from merlib.others import openface
from typing import Union,Tuple,List
from merlib.data import DLIB_FACE_LANDMARK_PATH


def align_face_in_frames(frames: List[np.ndarray],detected_frame_id:Union[int,None]=0, landmarkIndices=None,size=224):
    """
    frames: List[np.ndarray[Tuple[int, int, int], np.uint8]]
    """
    predictor_path=DLIB_FACE_LANDMARK_PATH

    # 初始化 OpenFace 人脸对齐器
    align = openface.AlignDlib(predictor_path)

    # 初始化landmarkIndices
    if landmarkIndices is None:
        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE
    
    # 初始化关键点
    landmarks = None
    face_bb=None
    aligned_frames = []

    # 根据指定帧获得，默认是第一帧landmarks和face_bb
    if detected_frame_id is not None:
        detected_frame=frames[detected_frame_id]
        face_bb=align.getLargestFaceBoundingBox(detected_frame)
        landmarks=align.findLandmarks(detected_frame,face_bb)
        
    for i, frame in enumerate(frames):
        # 没有指定detected_frame_id，进行逐帧配准
        if detected_frame_id is None:
            # 检测人脸并定位关键点
            face_bb=align.getLargestFaceBoundingBox(frame)
            landmarks=align.findLandmarks(frame,face_bb)

        # 使用 OpenFace 进行人脸对齐
        aligned_face = align.align(
            size, frame, face_bb, landmarks, landmarkIndices=landmarkIndices)

        # 将对齐后的人脸图像保存到帧序列中
        aligned_frames.append(aligned_face)

    return aligned_frames

from PIL import Image

def get_simple_square_face(frames:List[np.ndarray],detected_frame_id: Union[int,None]):
    if detected_frame_id is None:
        detected_frame_id =0
    assert isinstance(detected_frame_id,int)
    align = openface.AlignDlib(DLIB_FACE_LANDMARK_PATH)
    bb=align.getLargestFaceBoundingBox(frames[detected_frame_id]) # left,top,right,bottom
    assert bb is not None
    width,height=bb.width(),bb.height()
    left,top,right,bottom = bb.left(),bb.top(),bb.right(),bb.bottom()
    y1,x1,y2,x2=None,None,None,None
    assert  height>=width
    if  x1 is None and height>=width:
        y1=top
        y2=bottom
        diff=(height-width)/2
        x1=left-diff
        x2=right+diff
    return [x[ int(y1):int(y2),int(x1):int(x2)] for x in frames]    




def align_face_by_openface(frames:List[Image.Image],detected_frame_id: Union[int,None]=0,resize=224,
                           landmarkIndices:List[int]=openface.AlignDlib.OUTER_EYES_AND_NOSE,**kwargs)->List[Image.Image]:
                           
    """
    detected_frame_id:int|None, default 0
    landmarkIndices:list, default OUTER_EYES_AND_NOSE = [36, 45, 33]
    首先，指定对哪一帧进行关键点检测，如果没有指定，表示逐帧配准，默认第一帧；指定哪三个关键点确定仿射变换矩阵，如果没有指定，默认outer eyes and nose
    然后，根据获得的仿射变换矩阵，对整个序列进行配准。
    """
    _=[np.array(x) for x in frames]
    aligned_frames=align_face_in_frames(_,detected_frame_id,landmarkIndices,size=resize)
    aligned_frames=[Image.fromarray(x) for x in aligned_frames]
    return aligned_frames

def test_align_face_in_frames():
    import numpy as np
    from PIL import Image
    from pathlib import Path
    from skimage import io,color,transform

    from moviepy.editor import ImageSequenceClip
    
    face_frames_dir="/home/neu-wang/mby/database/CASME2/dlib_crop_twice2/sub01/EP02_01f" # img46-86.jpg
    start_end=(46,86)
    format_str="img{}.jpg"
    fps=200
    # face_frames_dir="/home/neu-wang/mby/database/CASME3-PartA/demo_data/partA_spNO.184_k_color" # 149-161.jpg
    # start_end=(149,161)
    # format_str="{}.jpg"
    # fps=30

    image_paths=[Path(face_frames_dir)/format_str.format(x)  for x in range(*start_end)]
    
    # 将路径排序，保证读取顺序一致
    frames=io.imread_collection( [x.resolve().__str__() for x in image_paths])
    frames=[x for x in frames]
    # frames=[np.array( Image.fromarray(x).resize((224,224))) for x in frames]
    video_clip = ImageSequenceClip(frames, fps=fps)

    # Write video to file
    video_clip.write_videofile(Path('~/download/origin_frames.mp4').expanduser().__str__(), fps=fps)
    # 调用函数进行人脸对齐
    aligned_frames = align_face_in_frames(frames)

    # Create video clip from aligned frames
    video_clip = ImageSequenceClip(aligned_frames, fps=fps)
    # Write video to file
    video_clip.write_videofile(Path('~/download/aligned_frames.mp4').expanduser().__str__(), fps=fps)
    # 检查返回结果是否符合要求
    assert len(aligned_frames) == len(frames)
    for aligned_frame in aligned_frames:
        assert isinstance(aligned_frame, np.ndarray)
        assert aligned_frame.shape == (224, 224, 3)
        assert aligned_frame.dtype == np.uint8

    pass
    # 检查对齐后的图像是否与原图有差异
    for i in range(len(frames)):
        # 获取对齐前后的图像
        frame = frames[i]
        aligned_frame = aligned_frames[i]
        
        # 使用 OpenCV 计算两幅图像的结构相似度指标（SSIM）
        from skimage.metrics import structural_similarity as compare_ssim
        score, diff = compare_ssim( transform.resize( color.rgb2gray(frame),(224,224)) ,
                                    color.rgb2gray(aligned_frame) ,
                                    full=True)

        # 检查 SSIM 是否大于阈值
        assert score > 0.95

