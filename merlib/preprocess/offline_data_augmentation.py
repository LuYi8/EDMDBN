"""
经过casme2上初步测试，此项技术在微表情分类不起作用
"""
from pathlib import Path
from typing import Union
anno='/home/neu-wang/mby/database/CASME2/uqi_3classes-5folds-annotations'
def generate_offline_aug_train_anno(train_annotation_file_path:Union[str,Path]):
    import pandas as pd
    orign_df=pd.read_csv(train_annotation_file_path,sep=' ',header=None)
    derived_df1=orign_df.copy()
    derived_df1[3]=derived_df1[2]
    # derived_df1
    derived_df2=orign_df.copy()
    derived_df2[1]=derived_df2[2]
    # derived_df2
    aug_df=pd.concat([orign_df,derived_df1,derived_df2],ignore_index=True)
    # aug_df
    name=Path(train_annotation_file_path).name.split('_')
    name.insert(1,'offline_aug')
    name='_'.join(name)
    aug_df.to_csv(Path(train_annotation_file_path).parent/name,header=False,index=False)