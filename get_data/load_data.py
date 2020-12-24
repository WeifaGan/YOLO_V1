import torch
from torch.utils import data 
import os
import sys
from get_data.augment import *


class dataset(data.Dataset):
    def __init__(self,txt_path,img_dir,transform,to_shape,snum,bnum,cnum):
        super().__init__()
        self.txt_path = txt_path
        self.img_dir = img_dir
        self.to_shape = to_shape
        self.transform = transform 
        self.snum = snum
        self.bnum = bnum
        self.cnum = cnum
        self.bbxs = []
        self.labels = []
        self.frames = []

 
        with open(txt_path,'r',encoding='utf-8') as f:
            self.lines = f.readlines()

        for line in self.lines: 
            line = line.split(' ')
            label = []
            bx = []
            i = 0
            while 1:
                bbx = (line[i*5+1:(i+1)*5+1]) 
                x0 = float(bbx[0])
                y0 = float(bbx[1])
                x1 = float(bbx[2])
                y1 = float(bbx[3])
                clss = int(bbx[4].replace('\n',''))
                label.append(clss)
                bx.append([x0,y0,x1,y1])

                if (i+1)*5+1 >= len(line)-1:break
                i+=1
            self.labels.append(torch.tensor(label))
            self.bbxs.append(torch.tensor(bx))
            self.frames.append(line[0])


    def __len__(self):
        return len(self.lines)

    def __getitem__(self,index):
        frame = self.frames[index]
        image = cv2.imread(os.path.join(self.img_dir,frame))

        bbxs = self.bbxs[index]
        labels = self.labels[index]
        image,bbxs,labels = self.augment(image,bbxs,labels)

        h,w,_ = image.shape 
        assert (h,w,3)==(self.to_shape[0],self.to_shape[1],3)

        bbxs /= torch.tensor([w,h,w,h]).expand_as(bbxs)
        target = self.make_target_label(bbxs,labels)
        image = self.transform(image)
        return image,target

    def make_target_label(self,bbxs,labels):

        bbx_t = []
        for i in bbxs:
            c_x,c_y = (i[0]+i[2])/2,(i[1]+i[3])/2
            width,height = i[2]-i[0],i[3]-i[1]
            bbx_t.append([c_x,c_y,width,height])

        ele_num =  self.bnum*5+self.cnum
        if not bbx_t:return np.zeros(self.snum,self.snum,ele_num)
        
        
        bbx_t = np.array(bbx_t)
        c_x = bbx_t[:,0].reshape(-1,1)
        c_y = bbx_t[:,1].reshape(-1,1)
        w = bbx_t[:,2].reshape(-1,1)
        h = bbx_t[:,3].reshape(-1,1)

        grid_id_x = c_x//(1/self.snum) 
        grid_id_y = c_y//(1/self.snum) 

        offset_x = c_x-1/self.snum/2-grid_id_x/self.snum
        offset_y = c_y-1/self.snum/2-grid_id_y/self.snum

        bbx_num = len(bbx_t)
        np_target = np.zeros((self.snum, self.snum, ele_num),dtype=np.float32)
        np_clss = np.zeros((bbx_num,self.cnum))

        for i in range(bbx_num):
            np_clss[i,int(labels[i])] = 1

        conf = np.ones_like(c_x)
        tmp = np.concatenate([offset_x,offset_y,w,h,conf],axis=1)
        tmp = np.repeat(tmp,self.bnum,axis=0).reshape(bbx_num,-1)
        tmp = np.concatenate([tmp,np_clss],axis=1)
        for i in range(bbx_num):
            np_target[int(grid_id_x[i]),int(grid_id_y[i])]=tmp[i]

        return np_target 


    def augment(self,img,bbxs,labels):
        
        # cv2.imshow('img_ori',img)
        # cv2.waitKey(-1)
        image,bbxs = random_flip(img,bbxs)
        image = RandomSaturation(img)
        image = RandomBrightness(img)
        image,bbxs,labels = randomShift(img,bbxs,labels)
        # cv2.imshow('img',image)
        # cv2.waitKey(-1)
        image,bbxs = padding_resize(image,bbxs,self.to_shape)


        return image,torch.tensor(bbxs),torch.tensor(labels)


if __name__ == "__main__":
    txt_path = './get_data/voc2012_trainval.txt' 
    img_dir  = '/media/gwf/D1/Dataset/VOC2012/JPEGImages'

    for i in dataloader(txt_path,img_dir,(448,448),7,2,20,True):
        print(i)







