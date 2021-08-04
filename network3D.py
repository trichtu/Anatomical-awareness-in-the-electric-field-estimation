import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv3D_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv3D_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class conv3D_11_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv3D_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=1,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=1,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x



class up_conv3D(nn.Module):
    def __init__(self, ch_in, ch_out, scale_factor):
        super(up_conv3D, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='nearest'),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
		    nn.BatchNorm3d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class U_Net3D(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(U_Net3D, self).__init__()
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1))
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv3D_block(ch_in=img_ch, ch_out=8)
        self.Conv2 = conv3D_block(ch_in=8, ch_out=16)
        self.Conv3 = conv3D_block(ch_in=16, ch_out=32)
        self.Conv4 = conv3D_block(ch_in=32, ch_out=64)
        self.Conv5 = conv3D_block(ch_in=64, ch_out=128)

        self.Up5 = up_conv3D(ch_in=128, ch_out=64, scale_factor=2)
        self.Up_conv5 = conv3D_block(ch_in=128, ch_out=64)

        self.Up4 = up_conv3D(ch_in=64, ch_out=32, scale_factor=2)
        self.Up_conv4 = conv3D_block(ch_in=64, ch_out=32)
        
        self.Up3 = up_conv3D(ch_in=32, ch_out=16, scale_factor=2)
        self.Up_conv3 = conv3D_block(ch_in=32, ch_out=16)
        
        self.Up2 = up_conv3D(ch_in=16, ch_out=8, scale_factor=(2,2,1))
        self.Up_conv2 = conv3D_block(ch_in=16, ch_out=8)

        self.Conv_1x1 = nn.Conv3d(8,output_ch, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool2(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        print(x4.shape, d5.shape)
        d5 = torch.cat((x4,d5), dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)

        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)

        d3 = torch.cat((x2, d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



class Attention_block3D(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm3d(1)
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        psi = self.relu(psi)

        return x*psi



class U_Net3D_Att(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(U_Net3D_Att, self).__init__()
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1))
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv3D_block(ch_in=img_ch, ch_out=8)
        self.Conv2 = conv3D_block(ch_in=8, ch_out=16)
        self.Conv3 = conv3D_block(ch_in=16, ch_out=32)
        self.Conv4 = conv3D_block(ch_in=32, ch_out=64)
        self.Conv5 = conv3D_block(ch_in=64, ch_out=128)

        self.Up5 = up_conv3D(ch_in=128, ch_out=64, scale_factor=2)
        self.Att5 = Attention_block3D(F_g=64,F_l=64,F_int=32)
        self.Up_conv5 = conv3D_block(ch_in=128, ch_out=64)

        self.Up4 = up_conv3D(ch_in=64, ch_out=32, scale_factor=2)
        self.Att4 = Attention_block3D(F_g=32,F_l=32,F_int=16)
        self.Up_conv4 = conv3D_block(ch_in=64, ch_out=32)
        
        self.Up3 = up_conv3D(ch_in=32, ch_out=16, scale_factor=2)
        self.Att3 = Attention_block3D(F_g=16,F_l=16,F_int=8)
        self.Up_conv3 = conv3D_block(ch_in=32, ch_out=16)
        
        self.Up2 = up_conv3D(ch_in=16, ch_out=8, scale_factor=(2,2,1))
        self.Att2 = Attention_block3D(F_g=8,F_l=8,F_int=4)
        self.Up_conv2 = conv3D_block(ch_in=16, ch_out=8)

        self.Conv_1x1 = nn.Conv3d(8,output_ch, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool2(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)


        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5), dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2, d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

