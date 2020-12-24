import sys
import torch
import torch.nn as nn
import argparse
from get_data.load_data import dataset
import torchvision.transforms as transforms
from model import yolov1
import torch.optim as optim
from utils.loss import YOLOLoss

parser = argparse.ArgumentParser(description='yolov1 training')
parser.add_argument('--lr',default=0.01,type=float,help='learning rate')
parser.add_argument('--batch_size',default=16,type=int,help='batch size')
parser.add_argument('--epochs',default=100,type=int,help='training epoch')
parser.add_argument('--grid_num',default=7,type=int,help='the num of grid')
parser.add_argument('--bbx_num',default=2,type=int,help='the num of bbx each grid')
parser.add_argument('--class_num',default=20,type=int,help='the num of class')
parser.add_argument('--img_dir',type=str,help='the path of image dir')
parser.add_argument('--txt_path',type=str,help='the path of txt file which contain image name and labels')
parser.add_argument('--input_size',default=(448,448),type=tuple,help='the input size of model')


args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])


trainset = dataset(args.txt_path,args.img_dir,transform_train,args.input_size,args.grid_num,
            args.bbx_num,args.class_num)

trianloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

net = yolov1(args.bbx_num,args.grid_num,args.batch_size).to(device)

optimizer = optim.SGD(net.parameters(),lr=args.lr,momentum=0.95, weight_decay=5e-4) 
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[30,50,80], gamma=0.5)

criterion = YOLOLoss(args.batch_size,args.bbx_num,args.grid_num,args.class_num,5,0.5)

for ep in range(args.epochs):
    for idx,(image,targets) in enumerate(trianloader):
        image,targets = image.to(device),targets.to(device)
        out = net(image)
        loss = criterion(out,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss:',loss.item())
        if idx %5==0:
            torch.save(net.state_dict(),'./checkpoint/model.pth')
            print('loss:',loss.item())


    lr_scheduler.step()








