#######################
# 
# 文件依赖于3方面，数据集dataclass类、处理后的doc dataframe、剪裁函数。
# 处理出错时信息被写入当前文件下crop_log.txt下，没有报错说明处理成功
#######################
import argparse
import typing
from merlib.preprocess.utils.openface_align_face import align_face_by_openface

from pathlib import Path,PurePath
from tqdm import tqdm
from PIL import Image
import multiprocessing 
from functools import partial
import multiprocessing, logging
import sys

# sys.path.append('/home/neu-wang/mby/mer/reference_lib/')
from merlib.data.base import ME_DATASET, DATASET_CHOICES
from merlib.data.clean_doc import build_cleaned_doc

#全局变量
# 定义可用的剪裁方法
ALIGN_FUNC_DICT = {'openface_align':align_face_by_openface }
PARAM_DICT={'openface_align': ['detected_frame_id','resize']}
LOG_FILE_PATH='./temp/align_log.txt'

if not Path(LOG_FILE_PATH).parent.exists():
    Path(LOG_FILE_PATH).parent.mkdir()

# 设置logger，error级别以上信息可以输入到控制台，warning级别以上信息输入到文件日志中
logger = multiprocessing.log_to_stderr()
logger.handlers[0].setLevel(logging.WARNING)

ch = logging.FileHandler(LOG_FILE_PATH,mode='w')
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def align_one_sample(align_face_func:typing.Callable,
                     img_paths:typing.Sequence[Path],save_img_paths:typing.Sequence[Path],
                     **kwargs):
    """给定对齐方法，通过原序列路径img_paths对其进行对齐，然后保存到指定的save_img_paths。
    openface_align func:
        detected_frame_id:int|None, default 0
        landmarkIndices:list, default OUTER_EYES_AND_NOSE = [36, 45, 33]
        首先，指定对哪一帧进行关键点检测，如果没有指定，表示根据第一帧配准；指定哪三个关键点确定仿射变换矩阵，如果没有指定，默认outer eyes and nose
        然后，根据获得的仿射变换矩阵，对整个序列进行配准。
    """
    # 每次处理一个序列
    imgs=[Image.open(img_path).convert('RGB') for img_path in img_paths ]

    aligned_imgs=align_face_func(imgs,**kwargs)

    assert len(img_paths)==len(save_img_paths)==len(aligned_imgs)
    for i,(img_path,save_img_path,aligned_img) in enumerate(zip(img_paths,save_img_paths,aligned_imgs)):
        try:
            save_img_path.parent.mkdir(parents=True,exist_ok=True)
            aligned_img.save(save_img_path) # 默认覆盖已存在的文件      
        except :
            logger.error("something is wrong in {}!".format(img_path.resolve()))
        
          
def main(args):
    dataset_name=args.dataset_name

    # 获取dataset_name数据集对应的元信息
    data_info=ME_DATASET[dataset_name]
    imagefile_template = data_info.imagefile_template
    raw_data_dir: Path = data_info.raw_data_path


    # 获取经过处理的样本文件
    df = build_cleaned_doc(dataset_name, data_info.doc_path)

    aligned_dir_name= args.align_function
    
    # debug模式下，对齐的数据保存在当前目录的/temp/dataset_name下，否则使用数据集元信息中save_path
    save_path: Path = (Path('./temp')/dataset_name) if args.debug else Path( data_info.save_path) 
    
    # 如果额外指定了save_path,则优先保存在此目录
    if args.save_path:
        save_path = Path(args.save_path)
    
    save_path=save_path/aligned_dir_name
    
    if save_path and not save_path.exists():
        print("{} doesn't exist, it will be created!".format(save_path.resolve()))
        save_path.mkdir(parents=True)
    
    # 多进程+qtdm进度显示处理剪裁任务
    with multiprocessing.Pool(args.workers) as pool:
        res=[]
        n=len(df)
        pbar = tqdm(total=n)
        pbar.set_description('Processing {} by {}'.format(args.dataset_name,args.align_function))
        update = lambda res:  pbar.update()

        # 根据相应的args.align_function进行人脸对齐
        align_func=partial(align_one_sample,ALIGN_FUNC_DICT[args.align_function])
        
        for x in df.itertuples(index=False):
            # x是一个视频序列
            start_index,end_index=x.start_frame,x.end_frame
            raw_video_path=x.path

            # 因为一个video可能存在多个samples，video_path不足以区分sample，导致图片剪裁时可能有重叠部分, sample_unique_path正是用来给sample提供唯一的路径
            sample_unique_path=x.unique_path # raw_video_path/"{}_{}".format(start_index,end_index)

            img_paths=[img_path for i in range(start_index,end_index+1) if (img_path:=raw_data_dir/raw_video_path /imagefile_template.format(i)).exists()  ]
            save_img_paths=[save_path/sample_unique_path/xx.name for xx in img_paths  ]
            assert len(img_paths) >0, "no images in {}".format(raw_video_path)
            kwargs=dict(zip(PARAM_DICT[args.align_function],[ getattr(args,x) for x in PARAM_DICT[args.align_function]]))
        
            _=pool.apply_async(align_func,
                args=(img_paths,save_img_paths),kwds=kwargs,callback=update,error_callback=lambda xx: print(xx))
            res.append(_)
        [x.get() for x in res]
    
def get_args():
   
    parser = argparse.ArgumentParser(
        'align face for the specified dataset',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'dataset_name', choices=DATASET_CHOICES,help="根据数据集名，指定需要使用的dataclass类，获取部分元信息")
    parser.add_argument('--no-debug', action='store_false',dest='debug', help='根据debug模式确定将处理的数据保存在当下的temp目录还是指定目录')
    parser.add_argument('--debug',default=True, action='store_true', help='根据debug模式确定将处理的数据保存在当下的temp目录还是指定目录')

    parser.add_argument('--save_path', default=None,help="指定要保存数据的目录，优先级高，一般情况不用指定")
    
    parser.add_argument('-af','--align_function', choices=ALIGN_FUNC_DICT.keys(), default='openface_align')

    parser.add_argument('--detected_frame_id',default=0,choices=[0,'Apex',None])
    parser.add_argument('--resize',default=224,type=int)

    parser.add_argument('--workers',default=1,type=int)

    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
