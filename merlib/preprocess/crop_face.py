#######################
# 
# 文件依赖于3方面，数据集dataclass类、处理后的doc dataframe、剪裁函数。
# 处理出错时信息被写入当前文件下crop_log.txt下，没有报错说明处理成功
#######################
import argparse
import typing
from utils.dlib_crop_face import dlib_crop_twice,dlib_crop_more_ROI,dlib_crop_face_v3
from utils.mtcnn_crop_face import crop_face_by_mtcnn
from pathlib import Path,PurePath
from tqdm import tqdm
from PIL import Image
import multiprocessing 
from functools import partial
import multiprocessing, logging
from merlib.data.base import ME_DATASET, DATASET_CHOICES
from merlib.data.clean_doc import build_cleaned_doc
from merlib.data import TEMP_DIR_PATH

Path(TEMP_DIR_PATH).mkdir(exist_ok=True)
# 定义可用的剪裁方法
CROP_FUNC_DICT = {'dlib_crop_twice':dlib_crop_twice, 'mtcnn_crop':crop_face_by_mtcnn,
                  'dlib_crop_more_ROI':dlib_crop_more_ROI,'dlib_crop_face_v3':dlib_crop_face_v3}
PARAM_DICT={'dlib_crop_face_v3': ['crop_twice']}

LOG_FILE_PATH=Path(TEMP_DIR_PATH)/'crop_log.txt'

# 设置logger，error级别以上信息可以输入到控制台，warning级别以上信息输入到文件日志中
logger = multiprocessing.log_to_stderr()
logger.handlers[0].setLevel(logging.WARNING)

ch = logging.FileHandler(LOG_FILE_PATH,mode='w')
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def crop_one_sample(crop_face_func:typing.Callable,img_paths:typing.Sequence[Path],save_img_paths:typing.Sequence[Path],**kwargs):
    """给定剪裁方法，通过原序列路径img_paths对其进行剪裁，保存到指定的save_img_paths。
    对第一张图片进行人脸检测，如果失败则此序列不会被处理；
    对随后的图片进行按得到box剪裁，如果剪裁失败、保存失败，log此张图片路径，不影响序列中其它图片处理。
    """
    box=None # The crop rectangle, as a (left, upper, right, lower)-tuple
    
    for i,(img_path,save_img_path) in enumerate(zip(img_paths,save_img_paths)):
        rgb_img = Image.open(img_path).convert('RGB')
        if i==0:
            box=crop_face_func(rgb_img,img_path=img_path,**kwargs)
            # 假定剪裁识别泛化None,需要了解crop_face_func剪裁失败的返回值
            if box is None:
                logger.error("face can't be found in {}!".format(img_path.resolve()))
                return
        try:
            cropped_img = rgb_img.crop(box)
            save_img_path.parent.mkdir(parents=True,exist_ok=True)
            cropped_img.save(save_img_path) # 默认覆盖已存在的文件      
        except :
            logger.error("something is wrong in {}!".format(img_path.resolve()))
        
       
          
def main(args):
    dataset_name=args.dataset_name

    data_info=ME_DATASET[dataset_name]
    imagefile_template = data_info.imagefile_template

    df = build_cleaned_doc(dataset_name, data_info.doc_path)
    raw_data_dir: Path = Path(args.align_data_path) if args.align_data_path  else  data_info.raw_data_path
    
    cropped_dir_name='aligned_{}'.format(args.crop_function) if args.align_data_path else args.crop_function

    # debug模式下，剪裁的数据保存在当前目录的/temp/dataset_name下，否则使用数据集元信息中save_path
    save_path: Path = (Path('./temp')/dataset_name) if args.debug else Path( data_info.save_path) 
    
    # 如果额外指定了save_path,则优先保存在此目录
    if args.save_path:
        save_path = Path(args.save_path)

    save_path=save_path/cropped_dir_name

    if save_path and not save_path.exists():
        print("{} doesn't exist, it will be created!".format(save_path.resolve()))
        save_path.mkdir(parents=True)
    
    # 多进程+qtdm进度显示处理剪裁任务
    with multiprocessing.Pool(args.workers) as pool:
        res=[]
        n=len(df)
        pbar = tqdm(total=n)
        pbar.set_description('Processing {} by {}'.format(args.dataset_name,args.crop_function))
        update = lambda res:  pbar.update()

        crop_func=partial(crop_one_sample,CROP_FUNC_DICT[args.crop_function])
        
        for x in df.itertuples(index=False):
            #x是一个视频序列
            start_index,end_index=x.start_frame,x.end_frame
            sample_unique_path=x.unique_path
            if args.use_given_faces_data:
                video_path= x.path
            else:
                video_path=x.unique_path if args.align_data_path else x.path


            img_paths=[img_path for i in range(start_index,end_index+1) if (img_path:=raw_data_dir/video_path /imagefile_template.format(i)).exists()  ]
            # 因为一个video可能存在多个samples，video_path不足以区分sample，导致图片剪裁时可能有重叠部分, sample_unique_path正是用来给sample提供唯一的路径
            save_img_paths=[save_path/sample_unique_path/xx.name for xx in img_paths  ]
            assert len(img_paths) >0, "no images in {}".format(video_path)
            kwargs=dict(zip(PARAM_DICT[args.crop_function],[ getattr(args,x) for x in PARAM_DICT[args.crop_function]]))

            _=pool.apply_async(crop_func,
                args=(img_paths,save_img_paths),kwds=kwargs ,callback=update,error_callback=lambda xx: print(xx))
            res.append(_)
        [x.get() for x in res]
    
def get_args():
   
    parser = argparse.ArgumentParser(
        'crop face  for the specified dataset',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'dataset_name', choices=DATASET_CHOICES,help="指定需要使用的dataclass类")
    parser.add_argument('-adp','--align_data_path',type=str,default=None,help='指定对对齐的人脸数据集进行剪裁，而非默认的原数据')
    parser.add_argument('--no-debug', action='store_false',dest='debug', help='根据debug模式确定将处理的数据保存在当下的temp目录还是指定目录')
    parser.add_argument('--debug',default=True, action='store_true', help='根据debug模式确定将处理的数据保存在当下的temp目录还是指定目录')
    parser.add_argument('--save_path', default=None,help="指定要保存数据的目录，优先级高，一般情况不用指定")
    parser.add_argument('-cf',
                        '--crop_function', choices=CROP_FUNC_DICT.keys(), default='dlib_crop_twice')
    parser.add_argument('--use_given_faces_data',default=False,action='store_true',help='如果使用数据集携带的、预处理好的人脸数据，则路径是非unique_path')

    parser.add_argument('--workers',default=1,type=int)

    parser.add_argument('--crop_twice',default=False, action='store_true', help='选择是否剪裁人脸两次，精确下边界')

    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
