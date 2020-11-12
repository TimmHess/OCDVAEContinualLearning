import torch

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