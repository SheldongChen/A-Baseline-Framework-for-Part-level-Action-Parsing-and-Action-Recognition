import glob
f_dir = '/export/home/data/PartHuman/test_videos_frames/'
list_frame = sorted(glob.glob(f_dir + '/*' * 3),reverse=True)
from tqdm import tqdm
import os
for i in tqdm(list_frame):
    if ".jpg"  in i:
        j = i[:-9]+str(int(i[-9:-4])+1).zfill(5)+i[-4:]
        cmd = ' '.join(['mv',i,j])
        #print(cmd)
        os.system(cmd)
