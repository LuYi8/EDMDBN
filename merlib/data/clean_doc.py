###################
# 此文件只处理标注错误，而不处理数据错误，比如数据帧缺失、数据剪裁、对齐遇到的错误。
###################
from .base import DATASET_CHOICES,ME_DATASET,SMICHSE_info, CASME2_info, MMEW_info, SAMM_info, SMICHS_info, CASME3UL_info,MEGC2019CD,FPS200CD,FPS200CD2
import pandas as pd
import numpy as np
from pathlib import Path, PurePath
from typing import Union

def make_dataframe(path , start_frame, apex_frame, end_frame, emotion_label, group,**kwargs):
    """
    path:pd.Series[PurePath] 对应video path
    unique_path: pd.Series[PurePath] 对应clip path
    , start_frame:pd.Series[int], apex_frame:pd.Series[int],
      end_frame:pd.Series[int], emotion_label:pd.Series[str], group:pd.Series[str],**kwargs):
    """
    if 'unique_path' not in kwargs:
        unique_path= path /( start_frame.map(str) +'_'+  end_frame.map(str))
    return pd.DataFrame({
        'path': path,
        'unique_path':unique_path,
        'start_frame': start_frame,
        'apex_frame': apex_frame,
        'end_frame': end_frame,
        'emotion_label': emotion_label,
        'group': group,
        **kwargs
    })

def process_CASME2(doc_path: Path) -> pd.DataFrame:
    """
    处理CASME2标注文件中有问题的数据: 一个sample数据帧不全,另一个是一个未标注的Apex帧
    :param doc_path:
    :return:
    """
    doc = pd.read_excel(doc_path)

    doc = doc.loc[:, ~doc.columns.str.contains("^Unnamed")]
    # 问题1
    # 方案1 Cropped数据中数据帧不全, 可选删除此clip sub16/EP01_08, repression
    # 方案2 通过RAW进行检测数据集
    # doc = doc[~((doc.Subject == 16) & (doc.Filename == 'EP01_08'))]

    # 问题2
    # 方案1：删除未标注Apex帧的数据
    # doc = doc[doc.ApexFrame != "/"]
    # 方案2：丢失的ApexFrame取中间帧，充分利用数据
    doc.loc[doc.ApexFrame == "/", 'ApexFrame'] = (doc.loc[doc.ApexFrame == "/", 'OnsetFrame'] + doc.loc[
        doc.ApexFrame == "/", 'OffsetFrame'])//2
    
    # 跟随主流，删除 标签为 'sadness','fear'的样本  
    doc=doc[~doc['Estimated Emotion'].isin(['sadness','fear'])]


    doc['ApexFrame'] = doc['ApexFrame'].astype(np.int64)


    doc = doc.reset_index(drop=True)

    video_paths = PurePath() / doc['Subject'].map(lambda x: 'sub{:02d}'.format(x)) / doc['Filename']

    return make_dataframe(
        path=video_paths,
        start_frame=doc['OnsetFrame'],
        apex_frame=doc['ApexFrame'],
        end_frame=doc['OffsetFrame'],
        emotion_label=doc['Estimated Emotion'],
        group=doc['Subject'].map(str)
     )

def process_MMEW(doc_path:Path)->pd.DataFrame:
    doc = pd.read_excel(doc_path)
    video_paths = PurePath() / doc['Estimated Emotion'] / doc['Filename']
    return make_dataframe(
        path=video_paths,
        start_frame=doc['OnsetFrame'],
        apex_frame=doc['ApexFrame'],
        end_frame=doc['OffsetFrame'],
        emotion_label=doc['Estimated Emotion'],
        group=doc['Subject']
     )

def process_CASME3(doc_path: Path) -> pd.DataFrame:
    """
    处理采取官方给出的解决方案：https://github.com/jingtingEmmaLi/CAS-ME-3
    对应文档cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx
    ,根据官方给出的错误处理文件casme3_partA_error_list_ZhouJu.xlsx: 
    15各样本人脸不全,存在手部遮挡，
    1个样本起点帧和Apex帧不存在；
    经过处理 860->816

    :param doc_path:
    :return:
    """

    doc = pd.read_excel(doc_path)

    doc = doc[doc['shade'] != 1]
    doc = doc[(doc['Offset']-doc['Onset'] + 1) > 1]
    doc = doc[(doc['Offset']-doc['Onset']+1) < 100]
    doc = doc[~((doc['Subject'] == 'spNO.215') & (
        doc['Filename'].str.lower() == 'l'))]
    doc = doc[~((doc['Subject'] == 'spNO.184') & (
        doc['Filename'].str.lower() == 'k'))]
    # .isin(['spNO.215/l', 'spNO.184/k'])] # 人脸比较正常，但dlib剪裁失败
    doc = doc.reset_index(drop=True)

    # /spNO.185/g/depth/1948.png
    video_paths = PurePath() / \
        doc['Subject'] / doc['Filename'].str.lower() / 'color'
 
    return make_dataframe(
        path=video_paths,
        start_frame=doc['Onset'],
        apex_frame=doc['Apex'],
        end_frame=doc['Offset'],
        emotion_label=doc['emotion'],
        group=doc['Subject']
     )


def process_SMICHS(doc_path: Path) -> pd.DataFrame:
    """
    目录s18_sur_03丢失，此外少数目录中部分图片丢失，共丢失32张
    对应SMIC_all_cropped数据文件夹，以下子目录存在缺失：
    s18_sur_03内容完全丢失
    s3_po_08丢失2张
    s3_sur_01、s3_sur_05、s3_ne_16各丢失1张
    """
    doc = pd.read_excel(doc_path)
    doc = doc[~doc['Filename'].isin(['s18_sur_03'])]
    doc = doc.reset_index(drop=True)

    video_paths = PurePath() / \
        doc['Subject'].map(lambda x: 's{}'.format(x)) / \
        'micro' / doc['Emotion'] / doc['Filename']
    doc['ApexF']=(doc['OnsetF']+doc['OffsetF'])//2
    return make_dataframe(
        path=video_paths,
        start_frame=doc['OnsetF'],
        apex_frame=doc['ApexF'],
        end_frame=doc['OffsetF'],
        emotion_label=doc['Emotion'],
        group=doc['Subject']
     )

def process_SMICHSE(doc_path: Path) -> pd.DataFrame:
    smic_hs_e_dataframe = pd.read_excel(doc_path)
    one_in_row_doc_dataframe:pd.DataFrame=pd.DataFrame({
    'Subject':[], 'Filename':[], 'OnsetF':[],'OffsetF':[],
    'TotalMF':[],'Emotion':[],
    'FirstF':[],'LastF':[],'TotalVL':[] # 该微表情所在的长视频的起点、终点帧，及总帧数
})

    for index, row in smic_hs_e_dataframe.iterrows():
        new_rows=[]
        if not pd.isna(row['TotalMF1']):
            new_row1 = {
            'Subject':row['Subject'], 'Filename':row['Filename'], 'OnsetF':row['OnsetF'],'OffsetF':row['OffsetF'],
            'TotalMF':row['TotalMF1'],'Emotion':row['Emotion'],
            'FirstF':row['FirstF'],'LastF':row['LastF'],'TotalVL':row['TotalVL']
            }
            new_rows.append(new_row1)

        if not pd.isna(row['TotalMF2']):
            new_row2 = {
            'Subject':row['Subject'], 'Filename':row['Filename'], 'OnsetF':row['Onset2F'],'OffsetF':row['Offset2F'],
            'TotalMF':row['TotalMF2'],'Emotion':row['Emotion'],
            'FirstF':row['FirstF'],'LastF':row['LastF'],'TotalVL':row['TotalVL']
            }
            new_rows.append(new_row2)

        if not pd.isna(row['TotalMF3']):
            new_row3 = {
            'Subject':row['Subject'], 'Filename':row['Filename'], 'OnsetF':row['Onset3F'],'OffsetF':row['Offset3F'],
            'TotalMF':row['TotalMF3'],'Emotion':row['Emotion'],
            'FirstF':row['FirstF'],'LastF':row['LastF'],'TotalVL':row['TotalVL']
            }
            new_rows.append(new_row3)
        one_in_row_doc_dataframe = pd.concat([one_in_row_doc_dataframe, pd.DataFrame(new_rows)], axis=0, ignore_index=True)

    one_in_row_doc_dataframe['Subject'] =one_in_row_doc_dataframe['Subject'].astype(int).astype(str)
    one_in_row_doc_dataframe[['Filename','Emotion']]=one_in_row_doc_dataframe[['Filename','Emotion']].astype(str)
    one_in_row_doc_dataframe[['OnsetF', 'OffsetF','TotalMF',
                            'FirstF','LastF','TotalVL']] = one_in_row_doc_dataframe[['OnsetF', 'OffsetF','TotalMF','FirstF','LastF','TotalVL']].astype(int)

    path= one_in_row_doc_dataframe['Subject'].apply(lambda x: 's{:02d}'.format(int(x)) ).apply(lambda x: PurePath(x) )
    path=path  / one_in_row_doc_dataframe['Filename']
    apex_frame=(one_in_row_doc_dataframe['OnsetF']+one_in_row_doc_dataframe['OffsetF'])//2
    standard_df=make_dataframe(path=path,
                            start_frame=one_in_row_doc_dataframe['OnsetF'],
                            apex_frame=apex_frame,
                            end_frame=one_in_row_doc_dataframe['OffsetF'],
                                emotion_label=one_in_row_doc_dataframe['Emotion'],
                                group=one_in_row_doc_dataframe['Subject']
                            )
    return standard_df

def process_SAMM(doc_path: Path) -> pd.DataFrame:
    """
    处理SAMM微表情数据集存在标注问题，主要是apex帧标注问题
    """
    doc = pd.read_excel(doc_path, header=13)

    doc=doc.drop(['Notes','Inducement Code','Micro'], axis=1)

    # apex帧不在区间内,取中间帧,共159条数据，没有舍去数据
    error_apex_filename = ["028_4_1", "032_3_1"]
    selected_rows=doc['Filename'].isin(error_apex_filename)
    doc.loc[selected_rows, 'Apex Frame'] = (doc.loc[selected_rows, 'Onset Frame'] + doc.loc[selected_rows, 'Offset Frame']) // 2

    video_paths = PurePath() / doc['Subject'].map(lambda x: '{:03d}'.format(x)) / doc['Filename']
    
    return make_dataframe(
        path=video_paths,
        start_frame=doc['Onset Frame'],
        apex_frame=doc['Apex Frame'],
        end_frame=doc['Offset Frame'],
        emotion_label=doc['Estimated Emotion'],
        group=doc['Subject'].astype(str)
        )


def process_CASME3UL(doc_path: Path) -> pd.DataFrame:
    """
    对应文档 CAS(ME)3_part_A_v2.xls
    处理CASME3标注文件中有问题的数据: 其一，一些人脸不全,存在手部遮挡;其二，一些微表情intervals. 
    :param doc_path:
    :return:
    """
    # error_video_paths=['spNO.143/h',
    # 'spNO.147/i', 'spNO.147/k', 'spNO.147/l', 'spNO.161/a', 'spNO.161/h', 'spNO.169/g', 'spNO.175/m',
    # 'spNO.179/g', 'spNO.184/a', 'spNO.184/b',
    # 'spNO.145/g','spNO.184/k','spNO.2/e', 'spNO.200/b', 'spNO.201/f','spNO.203/l', 'spNO.203/i',
    # 'spNO.214/b', 'spNO.214/k','spNO.215/l']

    doc = pd.read_excel(doc_path)

    # 处理1人脸剪裁出错
    error_video_paths = ["spNO.13/m","spNO.145/g","spNO.2/e","spNO.214/b",
"spNO.143/h",
"spNO.147/k",
"spNO.147/l",
"spNO.147/l",
"spNO.147/l",
"spNO.161/a",
"spNO.161/h",
"spNO.169/g",
"spNO.175/m",
"spNO.179/g",
"spNO.184/a",
"spNO.184/a",
"spNO.184/a",
"spNO.184/b",
"spNO.184/k",
"spNO.184/k",
"spNO.184/k",
"spNO.200/b",
"spNO.201/f",
"spNO.201/f",
"spNO.203/i",
"spNO.203/l",
"spNO.214/k",
"spNO.215/l"]
    temp_video_paths =  PurePath() /doc['Subject'] /doc['Filename'].str.lower()
    temp_video_paths:pd.Series = temp_video_paths.map(str)
    doc = doc[~temp_video_paths.isin(error_video_paths)]
    # 处理2 帧序列过短
    doc = doc[(doc['Offset']-doc['Onset']) >= 3]
    doc = doc.reset_index(drop=True)

    # /spNO.185/g/depth/1948.png
    video_paths = PurePath() / doc['Subject'] / doc['Filename'].str.lower() / 'color'

    return make_dataframe(
        path=video_paths,
        start_frame=doc['Onset'],
        apex_frame=doc['Apex'],
        end_frame=doc['Offset'],
        emotion_label=doc['Objective class'],
        group=doc['Subject'],
        expression_type=doc['Expression type']
     )


def process_MECG2019(doc_path:dict,data_root_path:dict):
    doc_path_dict=doc_path
    data_root_dict=data_root_path
    df_list=[]
    for name,doc_path in doc_path_dict.items():
        df=build_cleaned_doc(name, doc_path)
        df['dataset_name']=name
        if ME_DATASET[name].unique:
            df['path']= data_root_dict[name]/df['unique_path']
        else:
            df['path']= data_root_dict[name]/df['path']
        
        df['group']= name+df['group'].map(str)
        df_list.append(df)
    df=pd.concat(df_list,axis=0,ignore_index=True)
    return df

from typing import Union


def build_cleaned_doc(name: str, doc_path:Union[Path,dict],data_root_path:dict=None) -> pd.DataFrame:
    """
    @return a DataFrame obj with these attributes:
        'path': video_paths,
        'start_frame': start_indexes,
        'apex_frame': apex_indexes,
        'end_frame': end_indexes,
        'emotion_label': emotion_labels,
        'groups': groups
    """

    assert name in DATASET_CHOICES, "{} not in given {}".format(
        name, DATASET_CHOICES)
    if name == CASME2_info.name:
        return process_CASME2(doc_path)
    # if name == CASME3_info.name:
        # return process_CASME3(doc_path)
    if name == CASME3UL_info.name:
        return process_CASME3UL(doc_path)
    if name == SMICHS_info.name:
        return process_SMICHS(doc_path)
    if name ==SMICHSE_info.name:
        return process_SMICHSE(doc_path)
    if name == SAMM_info.name:
        return process_SAMM(doc_path)
    if name == MMEW_info.name:
        return process_MMEW(doc_path)
    if name == MEGC2019CD.name or name == FPS200CD.name or name==FPS200CD2.name:
        return process_MECG2019(doc_path,data_root_path)
    
    return pd.DataFrame()

