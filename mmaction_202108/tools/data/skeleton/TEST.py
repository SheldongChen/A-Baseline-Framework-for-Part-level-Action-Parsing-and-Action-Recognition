#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import glob
list_train = sorted(glob.glob('/export/home/data/PartHuman/train_videos'+'/*'*2))


# In[ ]:


dir_train =  sorted(glob.glob('/export/home/data/PartHuman/train_videos'+'/*'*1))


# In[ ]:


dict_train = dict()
for i,item in enumerate(dir_train):
    dict_train[item.split('/')[-1]] = i


# In[ ]:


dict_train


# In[ ]:


from tqdm import tqdm as tqdm
for video in tqdm(list_train):
    label = str(dict_train[video.split('/')[-2]])
    pkl_name = './train_pkls/'+video.split('/')[-1].replace('.mp4','.pkl')
    cmd = ' '.join(['python','ntu_pose_extraction.py',video,pkl_name,'--label',label])
    #print(cmd)
    os.system(cmd)


# In[ ]:





# In[ ]:


# import pickle
# with open("ZX12MQsqnI4_000094_000104.pkl", 'rb') as fo:     # 读取pkl文件数据
#     dict_data = pickle.load(fo, encoding='bytes')


# In[ ]:


#dict_data


# In[ ]:


list_test = sorted(glob.glob('/export/home/data/PartHuman/test_videos'+'/*'*2))
for video in tqdm(list_test):
    label = str(dict_train[video.split('/')[-2]])
    pkl_name = './test_pkls/'+video.split('/')[-1].replace('.mp4','.pkl')
    cmd = ' '.join(['python','ntu_pose_extraction.py',video,pkl_name,'--label',label])
    #print(cmd)
    os.system(cmd)


# In[ ]:





# In[ ]:




