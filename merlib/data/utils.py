from pathlib import Path
from tqdm import tqdm

import itertools
from collections import defaultdict
import pandas as pd

from  typing import List,Union
import typing
from PIL import Image

import matplotlib.pyplot as plt
import math

def check_onset_apex_offset_order(clipids,onsets,apexs,offsets):
    """
    return error_clip_ids
    """

    return_flag=True
    error_clip_ids=[]
    for clip_id, start,apex,end in zip( clipids,onsets,apexs,offsets):
        if  start<= apex<=end:
            pass
        else:
            error_clip_ids.append(clip_id)
            return_flag=False
            print(  "{} apex-{} is not in [{},{}]".format(clip_id,apex,start,end))
    if return_flag:
        print("Every thing is ok!")
    return return_flag,clipids[clipids.isin(error_clip_ids)]
def check_sample_lacks(video_paths,clipids,onsets,apexs,offsets,imagefile_template ):
    def list_to_str(nums):
        res = ""
        i = 0
        while i < len(nums):
            start_num = nums[i]
            end_num = start_num
            while i < len(nums) - 1 and nums[i + 1] == end_num + 1:
                end_num = nums[i + 1]
                i += 1
            if start_num != end_num:
                res += str(start_num) + "-" + str(end_num)
            else:
                res += str(start_num)
            if i < len(nums) - 1:
                res += ","
            i += 1
        return res

    df=pd.DataFrame({
        'video_path':[],
        'clip_id':[],
        'onset':[],
        'apex':[],
        'offset':[],
        'duration':[],
        'real_duration':[],
        'lack_rate':[],
        'lack_img_id':[]
    })

    for video_path,clip_id,start,apex,end in zip(video_paths,clipids,onsets,apexs,offsets):
        row={
        'video_path':video_path,
        'clip_id':clip_id,
        'onset': True,
        'apex':True,
        'offset':True,
        'duration':end-start+1,
        'real_duration':None,
        'lack_rate':None,
        'lack_img_id':[]
    }
        assert start<=end,clip_id
        # 检测onset,apex,offset帧是否存在
        img_path:Path=video_path/ imagefile_template.format(start)
        if not img_path.exists():
            row['onset']=False
        img_path:Path=video_path/ imagefile_template.format(apex)
        if not img_path.exists():
            row['apex']=False
        img_path:Path=video_path/ imagefile_template.format(end)
        if not img_path.exists():
            row['offset']=False
        counter=itertools.count()
        for z in range(start,end+1):
            img_path:Path=video_path/ imagefile_template.format(z)
            if  img_path.exists():
                next(counter)
                row['lack_img_id'].append(z)
                
        row['real_duration']=next(counter)
        row['lack_rate']=1- row['real_duration']/row['duration']
        row['lack_img_id']=list_to_str(row['lack_img_id'])

    lack_df=df.copy()
    lack_df=lack_df[lack_df['real_duration']<lack_df['duration']].reset_index(drop=True)
    res_df=lack_df.set_index(['video_path','clip_id'])

    if len(lack_df)==0:
        print('everything is ok')
    else:
        total_imgs=sum(df['duration'])
        lack_imgs=sum(lack_df['duration']-lack_df['real_duration'])
        print("总图片{}，丢失图片 {}，丢失率{:.2%}，集中在 {}个视频，{}个样本中。".format(total_imgs,lack_imgs,lack_imgs/total_imgs,len(lack_df.index.levels[0]),len(lack_df.index.levels[1])))
    return res_df


def random_read_img_in_seq(standard_df:pd.DataFrame,dataset_info):
    import random

    # 指定随机种子为10
    random.seed(10)

    raw_data_dir=dataset_info.raw_data_path
    doc=standard_df
    imagefile_template=dataset_info.imagefile_template
    res=[]
    sample_df = doc.groupby('group').apply(lambda x: x.sample(n=1, random_state=42))
    doc=sample_df

    names=[]
    for video_path,start,end in zip(doc['path'], doc['start_frame'],doc['end_frame']):
        while True:
            i=random.choice(range(start,end+1))
            img_path:Path=raw_data_dir/video_path/ imagefile_template.format(i)
            if img_path.exists():break
        res.append(img_path)
        names.append('{}'.format(video_path))
    return res,names

def get_images_info(imgs:List[Image.Image]):
    # 输出图像尺寸和色彩模式
    query_size=set()
    query_mode=set()
    query_format=set()
    for q in imgs:
        query_size.add(q.size)
        query_mode.add(q.mode)
        query_format.add(q.format)
    print('Image size:{}种，{}'.format(len(query_size),query_size))
    print('Image mode:{}种，{}'.format(len(query_mode),query_mode)) # L灰度，单通道；RGB,彩色，三通道
    print('Image format:{}种，{}'.format(len(query_format),query_format)) # L灰度，单通道；RGB,彩色，三通道

def random_read_img_in_seq(standard_df:pd.DataFrame,dataset_info,data_dir:typing.Optional[Path]=None):
    import random

    # 指定随机种子为10
    random.seed(10)
    
    raw_data_dir=dataset_info.raw_data_path if data_dir is None else data_dir
    doc=standard_df
    imagefile_template=dataset_info.imagefile_template
    res=[]
    sample_df = doc.groupby('group').apply(lambda x: x.sample(n=1, random_state=42))
    doc=sample_df

    names=[]
    for video_path,start,end in zip(doc['path'], doc['start_frame'],doc['end_frame']):
        i=random.choice(range(start,end))
        img_path:Path=raw_data_dir/video_path/ imagefile_template.format(i)
        res.append(img_path)
        names.append('{}'.format(video_path))
    return res,names

# 将多张图片展示在一个网格中，每行尽量占满一行
def plot_images(images, cols=5, cmap='gray', titles=None):
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    fig.tight_layout(pad=0.5)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img_height, img_width = images[i].size
            new_width = fig.get_figwidth() * fig.dpi / cols
            new_height = (new_width / img_width) * img_height
            ax.imshow(images[i], cmap=cmap)
            if titles is not None:
                ax.set_title(titles[i])
            ax.axis('off')
            ax.set_aspect('equal')
            # ax.set_ylim(2*new_height, 0)  # 图片默认y轴方向向下，需要翻转

def show_duration_len(standard_df,fps):
    standard_df=standard_df.copy()
    standard_df['Duration']=(standard_df['end_frame']-standard_df['start_frame']+1)
    standard_df['FrontDuration']=(standard_df['apex_frame']-standard_df['start_frame']+1)
    standard_df['BackDuration']=(standard_df['end_frame']-standard_df['apex_frame']+1)
    fig, ax = plt.subplots(2,3, figsize=(20, 12))
    ax=[*ax[0],*ax[1]]
    standard_df['Duration'].plot.hist(ax=ax[0])
    standard_df['FrontDuration'].plot.hist(ax=ax[1])
    standard_df['BackDuration'].plot.hist(ax=ax[2])
    (standard_df['Duration']/fps).plot.kde(ax=ax[3])
    (standard_df['FrontDuration']/fps).plot.kde(ax=ax[4])
    (standard_df['BackDuration']/fps).plot.kde(ax=ax[5])
    titles=['Duration','FrontDuration','BackDuration']
    for i in range(3):
        ax[i].set_title(titles[i])
        ax[i].set_xlabel('Frame Length')
        ax[i].set_ylabel('Count')
    for i in range(3,6):
        ax[i].set_title(titles[i%3])
        ax[i].set_xlabel('Time')
        ax[i].set_ylabel('Frequency')
    fig.suptitle('Frame Length/Time Distribution')
    plt.show()

def show_label_distribution(standard_df):
    fig, ax = plt.subplots(1,2, figsize=(12, 4))

    label_counts=standard_df['emotion_label'].value_counts()
    labels=label_counts.index
    label_count=label_counts.values
    label_percent=[label_count[i]/sum(label_count) for i in range(len(labels)) ]
    label_counts.plot.bar(ax=ax[0])
    label_counts.plot.pie(ax=ax[1], autopct='%1.1f%%',title='')
    # 设置标题和轴标签
    # 在柱子上方添加标签
    for i in range(len(labels)):
        ax[0].text(i, label_count[i], '{}'.format(label_count[i]) , ha='center', va='bottom')
    # 显示图形
    ax[0].set_xlabel('Label')
    ax[0].set_ylabel('Count')
    ax[1].set_ylabel('')
    fig.suptitle('Emotion Label Distribution(total:{})'.format(sum(label_count)))
    plt.show()