import torch
from torch.utils import data 
import cv2
import os
from augmment import *


class dataloader(data.Dataset):
    def __init__(self,augment,txt_path,img_dir):
        super().__init__()
        self.augment = augment 
        self.txt_path = txt_path
        self.img_dir = img_dir

        with open(txt_path,'r',encoding='utf-8') as f:
            lines = f.readlines()
        self.lines = lines 


    def __len__(self):
        return len(self.lines)

    def __getitem__(self,index):
    
        img_info = self.lines[index].split(' ')
        img_name = img_info[0]
        image = cv2.imread(os.path.join(self.img_dir,img_name))
        bbxs = []
        i = 0
        labels = []
        while 1:
            bbx = (img_info[i*5+1:(i+1)*5+1]) 
            x0 = float(bbx[0])
            y0 = float(bbx[1])
            x1 = float(bbx[2])
            y1 = float(bbx[3])
            label = int(bbx[4])
            labels.append(label)
            bbxs.append([x0,y0,x1,y1,label])
            if (i+1)*5+1 >= len(img_info)-1:break
            i+=1
        return image,bbxs,labels


if __name__ == "__main__":
    txt_path = './get_data/voc2012_trainval.txt' 
    img_dir  = '/media/gwf/D1/Dataset/VOC2012/JPEGImages'

    for i in dataloader(1,txt_path,img_dir):
        print(i)
        break







