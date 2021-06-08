import torch
import torch.nn as nn
import math

class IlluminationInvariant():
    def __call__(self, x):
        eps = 1e-7
        r = x[0,:,:]
        g = x[1,:,:]
        b = x[2,:,:]
        c_1 = torch.atan((r + eps)/(torch.max(g,b) +eps)).unsqueeze_(0)
        c_2 = torch.atan((g + eps)/(torch.max(r,b) +eps)).unsqueeze_(0)
        c_3 = torch.atan((b + eps)/(torch.max(g,r) +eps)).unsqueeze_(0)
        res = torch.cat([c_1,c_2,c_3], dim=0)
        return res

    def __repr__(self):
        return self.__class__.__name__ + '()'


class LBP(object):
    '''create p neighbouring positions in r distance to center pixel
       threshholding neighboring pixel with center pixel
       if neighbour >= center pixel => 1 else 0
       create decimal number from binary result'''
    def __init__(self, device, radius=1, points=8):
        self.radius = radius
        self.points = points
        
        #FILTER TO CONVERT TO GRAYSCALE
        #self.intensity = Intensity()
        
        #FILTER TO COMPARE WITH NEIGHBOURS
        self.size = self.radius*2+1
        self.neighbor_positions = self.positions()
        self.lbp_conv1 = nn.Conv2d(in_channels = 1, out_channels = self.points, kernel_size=self.size, padding =self.radius, bias = False)
        self.lbp_conv1.weight.data.fill_(0.0)
        for i, (w, h) in enumerate(self.neighbor_positions):
            self.lbp_conv1.weight.data[i,0,self.size//2,self.size//2]  = -1
            self.lbp_conv1.weight.data[i,0,w,h]= 1
        #print(self.lbp_conv1.weight.data)
        self.lbp_conv1.to(device)

        #FILTER TO CONVERT BINARY NUMBER(0,1 in channels) TO DECIMAL
        self.lbp_conv2 = nn.Conv2d(in_channels = self.points, out_channels = 1, kernel_size=1,bias = False)
        for i,i_chan in enumerate(range(self.points)):
            self.lbp_conv2.weight.data[0,i,0,0] = 2**i
        #print(self.lbp_conv2.weight.data)
        self.lbp_conv2.to(device)
        self.max_value = float(2**(points+1)-1) #max decimal number, defined by number of points

    def positions(self):
        mid  = self.radius
        positions =[]
        for i in range(self.points):
            #calculate angle and according position for every point
            alpha = 2*math.pi / self.points *i 
            x = int(round(mid + self.radius *math.cos(alpha)))
            y = int(round(mid + self.radius *math.sin(alpha)))
            positions.append((x,y))
        #print(positions)
        return positions

    def apply_to_channel(self,x):
        #convert to grayscale
        #x = self.intensity(x)
        #compare the the neighboring pixels to that of the central pixel
        x = self.lbp_conv1(x) 
        
        #convert to binary: 0 if lessequall
        x[x >= 0] = 1
        x[x < 0] = 0
        #print(x)
        
        #convert to decimal
        x = self.lbp_conv2(x)

        x= x/self.max_value
        return x

    def __call__(self,x):
        #print(x.shape)
        r = x[0,:,:].unsqueeze_(0).unsqueeze_(0)
        g = x[1,:,:].unsqueeze_(0).unsqueeze_(0)
        b = x[2,:,:].unsqueeze_(0).unsqueeze_(0)
        #print("r", r.shape)
        with torch.no_grad():
            c_1 = self.apply_to_channel(r).squeeze_(0)
            #print("c1", c_1.shape)
            #sys.exit()
            c_2 = self.apply_to_channel(g).squeeze_(0)
            c_3 = self.apply_to_channel(b).squeeze_(0)
        res = torch.cat([c_1,c_2,c_3], dim=0)
        #print(res.shape)
        #sys.exit()
        return res

    def __str__(self):
        return "LBP(radius_%i_points_%i)"%(self.radius, self.points)