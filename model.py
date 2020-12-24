import torch 
import torch.nn as nn

class yolov1(nn.Module):
    def __init__(self,bnum,snum,batch):
        super().__init__()
        self.bnum = bnum
        self.snum = snum
        self.batch_size = batch
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,padding=3,stride=2),
        nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,padding=1),
        nn.LeakyReLU(inplace=True))

        self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels=192,out_channels=128,kernel_size=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
        nn.LeakyReLU(inplace=True))


        self.conv4 = nn.Sequential(
        nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
        nn.LeakyReLU(inplace=True),

        nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
        nn.LeakyReLU(inplace=True),

        nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
        nn.LeakyReLU(inplace=True),
        
        nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
        nn.LeakyReLU(inplace=True),
        
        nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1),
        nn.LeakyReLU(inplace=True))

        self.conv5 = nn.Sequential(
        nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1),
        nn.LeakyReLU(inplace=True),     
        nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1),
        nn.LeakyReLU(inplace=True),     
        nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1),
        nn.LeakyReLU(inplace=True),     
        nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1,stride=2),
        nn.LeakyReLU(inplace=True)

        )

        self.conv6 = nn.Sequential(
        nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1),
        nn.LeakyReLU(inplace=True)
        )


        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(7*7*1024,4096)
        self.fc2 = nn.Linear(4096,7*7*30)


    def forward(self,x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        out = self.maxpool(out)
        out = self.conv4(out)
        out = self.maxpool(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = nn.LeakyReLU(inplace=True)(out)
        out = self.fc2(out)
        out = out.reshape(self.batch_size,self.snum,self.snum,-1)
        return out