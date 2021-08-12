import os

import cv2

data_dir =  '/home/meprint/sunanlin_folder/data_copy'

with open('data_copy.txt','w') as f:
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir,folder)
        for img_name in os.listdir(folder_path):
            if "(1)" not in img_name:
                img_path = os.path.join(folder_path,img_name)
                img = cv2.imread(img_path)
                h,w,_ = img.shape
                if h==0 or w==0:
                    break
                bbox = str(0)+','+str(0)+','+str(w)+','+str(h)
                f.write(img_path+' '+bbox+','+folder+'\n')
f.close()