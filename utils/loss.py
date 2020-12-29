import torch
import torch.nn as nn
from torch.nn import functional
import time
from utils.IOU import count_iou
import numpy as np
class YOLOLoss(nn.Module):
    def __init__(self,batch_size,bnum,snum,clssnum,coord,noobj):
        super().__init__()
        self.batch_size = batch_size
        self.coord = coord
        self.noobj = noobj
        self.bnum = bnum
        self.snum = snum
        self.clss_num = clssnum


    def forward(self,predict,target):

        ele_num = self.bnum*5+self.clss_num

        predict = predict.view(predict.size(0),-1,ele_num)
        target = target.view(target.size(0),-1,ele_num)
        coord_mask = target[:,:,5]>0
        noobj_mask = target[:,:,5]==0

        coord_mask = coord_mask.unsqueeze(-1).expand_as(target)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target)


        coord_target = target[coord_mask].view(-1,ele_num)
        coord_predict = predict[coord_mask].view(-1,ele_num)
        noobj_target = target[noobj_mask].view(-1,ele_num)
        noobj_predict = predict[noobj_mask].view(-1,ele_num)


        # seperate class and bounding box
        clss_target = coord_target[:,self.bnum*5:]
        clss_predict = coord_predict[:,self.bnum*5:]
        bbx_target = coord_target[:,:self.bnum*5].contiguous().view(-1,5) #view要求连续存储
        bbx_predict = coord_predict[:,:self.bnum*5].contiguous().view(-1,5)

        # count noobj_loss for confidence
        noobj_target = noobj_target[:,:self.bnum*5].contiguous().view(-1,5)
        noobj_predict = noobj_predict[:,:self.bnum*5].contiguous().view(-1,5)
        noobj_target_conf = noobj_target[:,4]
        noobj_predict_conf = noobj_predict[:,4]
        noobj_conf_loss = functional.mse_loss(noobj_predict_conf,noobj_target_conf,size_average=False)

        # choose the best bounding box to count between the two 
        best_bx_mask = torch.zeros(bbx_predict.size())
        no_best_bx_mask = torch.zeros(bbx_predict.size())
        for i in range(0,bbx_predict.size(0),self.bnum):
            bbx1 = bbx_predict[i:i+self.bnum]
            bbx2 = bbx_target[i:i+self.bnum]
            iou = count_iou(bbx1,bbx2)
            max_iou,max_index = iou.max(0) 
            best_bx_mask[i+max_index]=1
            # nobest_bx_mask[i+max_index]=1
        best_bx_mask = best_bx_mask.bool()
        bbx_target = bbx_target[best_bx_mask].view(-1,5)
        bbx_predict = bbx_predict[best_bx_mask].view(-1,5)


        obj_confidence_loss = functional.mse_loss(bbx_predict[:,4],bbx_target[:,4],size_average=False)
        obj_center_loss = functional.mse_loss(bbx_predict[:,:2],bbx_target[:,:2],size_average=False) 
        obj_wh_loss = functional.mse_loss(bbx_predict[:,2:4],bbx_target[:,2:4],size_average=False)
        obj_class_loss = functional.mse_loss(clss_predict,clss_target,size_average=False)
        total_loss = self.coord*(obj_center_loss +obj_wh_loss)+obj_confidence_loss+self.noobj*noobj_conf_loss+obj_class_loss
        # TODO: sqrt?
        print(obj_confidence_loss.item(),self.coord*(obj_center_loss.item()+obj_wh_loss.item()),obj_class_loss.item(),self.noobj*noobj_conf_loss.item())
        return total_loss


        
        

        # print(b)
