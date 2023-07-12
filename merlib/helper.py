import re
from pathlib import Path
import pandas as pd
from prettytable import PrettyTable
import typing
from typing import Union,Iterable,List

def extract_metrics(pattern_str:str,path_list:Iterable[Union[str,Path]],save_file_path:Path):
    """

    """
    pattern=re.compile(pattern_str)
    path_list:List[Path]=list([Path(x) for x in path_list ] )
    if path_list[0].is_file():
        path_list=sorted(path_list,key= lambda x :x.stat().st_ctime)
    record_list=[]
    for x in path_list:
        res_dict=pattern.search(x.name).groupdict()
        record_list.append(res_dict)
    chdir_path=path_list[0].parent
    res=pd.DataFrame(record_list)
    res=res.applymap(float)
    last_line=str(res.mean()).replace('\n','; ')
    if save_file_path is not None:
        pd.Series([*[x.name for x in path_list],last_line]).to_csv(save_file_path,header=[str(chdir_path)])
    else: 
        print(last_line)
    return last_line

def print_dict_as_table(args_dict:typing.Dict):
    ptable = PrettyTable()
    # ptable._max_width={'value':45}
    _dict = args_dict.copy()
    _dict = dict(filter(lambda x:x[1] is not None ,_dict.items()))
    _dict = dict(sorted(_dict.items(), key=lambda x: x[0]))
    ptable.add_column('name', list(_dict.keys()))
    ptable.add_column('value', list(_dict.values()))
    ptable.add_column('type', [type(x) for x in _dict.values()])
    print(ptable.get_string())
    print('未出现的参数值为None\n')


def send_email(addr_from, authorization_code, addr_to, smtp_server, head_from, head_to, head_subject, message):
    """Python smtp服务发送邮件.

    Examples:
    --------
        1. 从163邮箱发邮件到qq邮箱.

        >>> send_email(
        ...    addr_from='qq418055608@163.com', #发送方邮箱
        ...    authorization_code='CVHKURKYNMSITPOG', #发送方邮箱授权码，不是密码！
        ...    addr_to='418055608@qq.com', #接收方邮箱
        ...    smtp_server='smtp.163.com',
        ...    head_from='我的程序',
        ...    head_to='本人',
        ...    head_subject='程序结束提醒',
        ...    message='程序已经跑完啦,快去查看')   
        邮件已发送
        True
        2. 从校园邮箱发送到qq邮箱(服务器无需联网)
        >>> send_email(
        ...     addr_from='2101768@stu.neu.edu.cn', #发送方邮箱
        ...     authorization_code="Ma990123@", #校园邮箱密码
        ...     addr_to='418055608@qq.com', #接收方邮箱
        ...     smtp_server='smtp.stu.neu.edu.cn', #NEUsmtp服务器地址
        ...     head_from='我的程序',
        ...     head_to='本人',
        ...     head_subject='测试',
        ...     message='测试')
        邮件已发送
        True
    """
    # smtplib 用于邮件的发信动作
    import smtplib
    # email 用于构建邮件内容
    from email.mime.text import MIMEText
    #构建邮件头
    from email.header import Header
    import  time

    time_for_email = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    # 发信方的信息：发信邮箱，QQ 邮箱授权码
    from_addr = addr_from
    password = authorization_code
 
    # 收信方邮箱
    to_addr = addr_to
 
    # 发信服务器
    smtp_server = smtp_server
 
    # 邮箱正文内容，第一个参数为内容，第二个参数为格式(plain 为纯文本)，第三个参数为编码
    msg = MIMEText('在{},{}'.format(time_for_email,message), 'plain', 'utf-8')
    msg['From'] = Header(head_from)
    msg['To'] = Header(head_to)
    msg['Subject'] = Header(head_subject)
    
    try:
        # server = smtplib.SMTP_SSL(smtp_server,465,timeout=3)
        server = smtplib.SMTP(smtp_server,25,timeout=3)
        # server = smtplib.SMTP_SSL(smtp_server,994,timeout=3)
        # server.connect(smtp_server, 465)
    
        server.login(from_addr, password)
    
        server.sendmail(from_addr, to_addr, msg.as_string())
        # 关闭服务器
        server.quit()
        print('邮件已发送')
        return True
    except TimeoutError:
        print('邮件超时未发送，可能是网络原因')
    
    return False

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
def plot_confusion(y_true:list, y_pred:list, labels:Union[list,None],*,save_path: typing.Optional[Path]=None ):
    """
    绘制混淆矩阵图像

    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param labels: 类别标签
    
    true_labels = [0, 1, 2, 0, 1, 2, 0, 2, 2]
    pred_labels = [0, 1, 2, 1, 2, 0, 2, 1, 1]
    classes = ['cat', 'dog', 'bird']
    plot_confusion(true_labels, pred_labels, classes,'test.png')
    """
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    if labels is None:
        labels=list(map(str,range(len(set(y_pred)))))
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred,normalize=None)
    # 绘制混淆矩阵图像
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=labels)
    disp.plot(ax=ax[0])

    cm = confusion_matrix(y_true, y_pred,normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=labels)
    disp.plot(ax=ax[1])
    
    fig.suptitle("Confusion Matrix(total:{})".format(len(y_true)))
    # fig.subplots_adjust(wspace=0.4)
    if save_path:
        fig.savefig(save_path)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def plot_clustering(feature_vectors:List[np.ndarray], num_classes:int,*,save_path:typing.Optional[Path]=None):
    """
    Example usage
    feature_vectors =np.random(100,202) # List of feature vectors
    labels=['surprise', 'positive', 'negative']

    perform_clustering(feature_vectors, len(labels))
    """
    # Convert feature_vectors to numpy array
    feature_vectors = np.array(feature_vectors)

    # Perform dimensionality reduction using PCA
    pca_reduced_features =  PCA(n_components=2,random_state=33).fit_transform(feature_vectors)
    tsne_reduced_features = TSNE(n_components=2,random_state=33,init='pca',learning_rate='auto').fit_transform(feature_vectors)

    # Create subplots and get the axes
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    # Perform K-means clustering
    names=['pca-kmean','tsne-kmean']
    label_names={}
    for col,reduced_features in enumerate([pca_reduced_features,tsne_reduced_features]):
        kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(reduced_features)

        # Save clustering results
        cluster_labels = kmeans.labels_
        # print(cluster_labels)
        # Define colors for plotting
        colors = plt.cm.get_cmap('tab10', num_classes).colors  # Generate colors based on a colormap
        # colors = ['r','g','b']
        for i in range(kmeans.labels_.max() + 1):
            ax[col].scatter(reduced_features[kmeans.labels_ == i, 0], reduced_features[kmeans.labels_ == i, 1],
                    color=[colors[i]], s=5)
            
            ax[col].set_title(names[col])
            label_names[i]=len(reduced_features[kmeans.labels_ == i])
        # Generate legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label="{},len {}".format(i,label_names[i]),
                                        markerfacecolor=colors[i], markersize=8) for i in range(num_classes)]
        ax[col].legend(handles=legend_elements, loc='upper right')
    fig.suptitle('clustering results')
    plt.show()
    # Save the figure
    if save_path:
        fig.savefig(save_path)




if __name__=='__main__':
    import unittest
    class TheseMethods(unittest.TestCase):

        def test_send_email(self):
            # 参考 https://mail.neu.edu.cn/coremail/help/mobile_zh_CN.jsp
            res=send_email(
                addr_from='2101768@stu.neu.edu.cn', #发送方邮箱
                authorization_code="Ma990123@",
                addr_to='418055608@qq.com', #接收方邮箱
                smtp_server='smtp.stu.neu.edu.cn',
                head_from='我的程序',
                head_to='本人',
                head_subject='测试',
                message='测试')
            self.assertTrue(res)
        
        def test_extract_metrics(self):
            re_path = 'fold(?P<fold>\d+)-epoch=(?P<epoch>\d+)-acc=(?P<acc>[.0-9]{6})-uf1=(?P<uf1>[.0-9]{6})-uar=(?P<uar>[.0-9]{6})'
            path_list=['/home/neu-wang/mby/mer/dual_flows/lightning_logs/samm/frames-swin_tiny-loso/version_3/checkpoints/fold0-epoch=0-acc=1.0000-uf1=1.0000-uar=1.0000.ckpt']
            res=extract_metrics(re_path,path_list,None)
            self.assertIsNotNone(res)
    
    
    unittest.main()



