import numpy as np 
import torch
def count_iou(bbx1,bbx2):
    """
    Compute the intersection over union of two boxes.
    params
        bbx1(tensor,numpy):[x_center,y_center,w,h],size:[N,4]
        bbx2(tensor,numpy):[x_center,y_center,w,h],size:[N,4]
    return
        IOU result(tensor)
    """
    #TODO:tensor check
    #transform the format:[xmin,ymin,xmax,ymax]
    
    bb1_xmin = bbx1[:,0]-bbx1[:,2]/2
    bb1_ymin = bbx1[:,1]-bbx1[:,3]/2
    bb1_xmax = bbx1[:,0]+bbx1[:,2]/2
    bb1_ymax = bbx1[:,1]+bbx1[:,3]/2
    
    bb2_xmin = bbx2[:,0]-bbx2[:,2]/2
    bb2_ymin = bbx2[:,1]-bbx2[:,3]/2
    bb2_xmax = bbx2[:,0]+bbx2[:,2]/2
    bb2_ymax = bbx2[:,1]+bbx2[:,3]/2

    iou_xmin = torch.max(bb1_xmin.float(),bb2_xmin.float())
    iou_ymin = torch.max(bb1_ymin.float(),bb2_ymin.float())
    iou_xmax = torch.min(bb1_xmax.float(),bb2_xmax.float())
    iou_ymax = torch.min(bb1_ymax.float(),bb2_ymax.float())

    w = torch.max(torch.zeros(iou_xmax.size(),dtype=torch.float).to('cuda:0'),(iou_xmax-iou_xmin))
    h = torch.max(torch.zeros(iou_xmax.size(),dtype=torch.float).to('cuda:0'),(iou_ymax-iou_ymin))
    iou_area = w*h 
    iou = iou_area/((bbx1[:,2]*bbx1[:,3]+bbx2[:,2]*bbx2[:,3])-iou_area)

    return iou

if __name__ == "__main__":
    res = count_iou(torch.tensor([[15,15,10,10],[15,15,10,10]]),torch.tensor([[20,20,10,10],[20,20,10,10]]))
    print(res)



 