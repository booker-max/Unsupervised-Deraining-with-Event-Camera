import torch
import torch.nn.functional as F
from einops import rearrange

class FALoss(torch.nn.Module):
    def __init__(self,subscale=0.0625):
        super(FALoss,self).__init__()
        self.subscale=int(1/subscale)

    def forward(self,feature1,feature2):
        feature1=torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2=torch.nn.AvgPool2d(self.subscale)(feature2)
        
        m_batchsize, C, height, width = feature1.size()
        feature1 = feature1.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        # L2norm=torch.norm(feature1,2,1,keepdim=True).repeat(1,C,1)   #[N,1,W*H]
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)  #haven't implemented in torch 0.4.1, so i use repeat instead
        # feature1=torch.div(feature1,L2norm)
        mat1 = torch.bmm(feature1.permute(0,2,1),feature1) #[N,W*H,W*H]

        m_batchsize, C, height, width = feature2.size()
        feature2 = feature2.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        # L2norm=torch.norm(feature2,2,1,keepdim=True).repeat(1,C,1)
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)
        # feature2=torch.div(feature2,L2norm)
        mat2 = torch.bmm(feature2.permute(0,2,1),feature2) #[N,W*H,W*H]

        L1norm=torch.norm(mat2-mat1,1)

        return L1norm/((height*width)**2) 

#### Channel similarity
class FCALoss(torch.nn.Module):
    def __init__(self,subscale=0.0625):
        super(FCALoss,self).__init__()
        self.subscale=int(1/subscale)

    def forward(self,feature1,feature2):
        feature1=torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2=torch.nn.AvgPool2d(self.subscale)(feature2)
        
        m_batchsize, C, height, width = feature1.size()
        feature1 = feature1.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        # L2norm=torch.norm(feature1,2,1,keepdim=True).repeat(1,C,1)   #[N,1,W*H]
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)  #haven't implemented in torch 0.4.1, so i use repeat instead
        # feature1=torch.div(feature1,L2norm)
        mat1 = torch.bmm(feature1,feature1.permute(0,2,1)) #[N,C,C]

        m_batchsize, C, height, width = feature2.size()
        feature2 = feature2.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        # L2norm=torch.norm(feature2,2,1,keepdim=True).repeat(1,C,1)
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)
        # feature2=torch.div(feature2,L2norm)
        mat2 = torch.bmm(feature2,feature2.permute(0,2,1)) #[N,C,C]

        L1norm=torch.norm(mat2-mat1,1)

        return L1norm/(C**2)


#### 直接计算 两个特征之间的similarity，positive之间的similarity最大, negative 之间的similarity最小
class FALoss_max(torch.nn.Module):

    def __init__(self, alpha=0.25, subscale=0.0625):
        super(FALoss_max,self).__init__()

        self.subscale=int(1/subscale)
        self.alpha = alpha

    def forward(self,feature1,feature2):
        feature1=torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2=torch.nn.AvgPool2d(self.subscale)(feature2)
        
        m_batchsize, C, height, width = feature1.size()
        feature1 = feature1.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        feature2 = feature2.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        
        ###ccam
        feature1 = F.normalize(feature1, dim=1)
        feature2 = F.normalize(feature2, dim=1)

        # L2norm=torch.norm(feature1,2,1,keepdim=True).repeat(1,C,1)   #[N,1,W*H]
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)  #haven't implemented in torch 0.4.1, so i use repeat instead
        # feature1=torch.div(feature1,L2norm)
        mat1 = torch.bmm(feature1.permute(0,2,1),feature2) #[N,W*H,W*H], 表示N个(W*H) * (W*H)个相似矩阵

        mat1 = torch.clamp(mat1, min=0.0005, max=0.9995)

        loss = -torch.log(mat1)
        loss[loss < 0] = 0
        _, indices = mat1.sort(descending=True, dim=1) #indices.shape = [2, 16, 16]
        _, rank = indices.sort(dim=1)
        rank = rank - 1
        rank_weights = torch.exp(-rank.float() * self.alpha)
        loss = loss * rank_weights

        return torch.mean(loss)

class FALoss_min(torch.nn.Module):

    def __init__(self,subscale=0.0625):
        super(FALoss_min,self).__init__()
        self.subscale=int(1/subscale)

    def forward(self,feature1,feature2):
        feature1=torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2=torch.nn.AvgPool2d(self.subscale)(feature2)
        
        m_batchsize, C, height, width = feature1.size()
        feature1 = feature1.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        feature2 = feature2.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        
        ###ccam
        feature1 = F.normalize(feature1, dim=1)
        feature2 = F.normalize(feature2, dim=1)

        # L2norm=torch.norm(feature1,2,1,keepdim=True).repeat(1,C,1)   #[N,1,W*H]
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)  #haven't implemented in torch 0.4.1, so i use repeat instead
        # feature1=torch.div(feature1,L2norm)
        mat1 = torch.bmm(feature1.permute(0,2,1),feature2) #[N,W*H,W*H]
        mat1 = torch.clamp(mat1, min=0.0005, max=0.9995)
        
        loss = -torch.log(1 - mat1)

        return torch.mean(loss)

##Attention FALoss_min
class AFALoss_min(torch.nn.Module):

    def __init__(self, alpha=0.25, subscale=0.0625):
        super(AFALoss_min,self).__init__()
        self.subscale=int(1/subscale)
        self.alpha = alpha

    def forward(self,feature1,feature2):
        feature1=torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2=torch.nn.AvgPool2d(self.subscale)(feature2)
        
        m_batchsize, C, height, width = feature1.size()
        feature1 = feature1.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        feature2 = feature2.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        
        ###ccam
        feature1 = F.normalize(feature1, dim=1)
        feature2 = F.normalize(feature2, dim=1)

        # L2norm=torch.norm(feature1,2,1,keepdim=True).repeat(1,C,1)   #[N,1,W*H]
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)  #haven't implemented in torch 0.4.1, so i use repeat instead
        # feature1=torch.div(feature1,L2norm)
        mat1 = torch.bmm(feature1.permute(0,2,1),feature2) #[N,W*H,W*H]
        mat1 = torch.clamp(mat1, min=0.0005, max=0.9995)
        
        loss = -torch.log(1 - mat1)

        loss[loss < 0] = 0
        _, indices = mat1.sort(descending=True, dim=1) #indices.shape = [2, 16, 16]
        _, rank = indices.sort(dim=1)
        rank = rank - 1
        rank_weights = torch.exp(-rank.float() * self.alpha)
        loss = loss * rank_weights

        return torch.mean(loss)

#### FIALoss是指Instance,即在中间过程.view(batchsize, -1), 这样是为了像CCAM那样构造Instace_level的loss
class FIALoss_max(torch.nn.Module):

    def __init__(self, alpha=0.25, subscale=0.0625):
        super(FIALoss_max,self).__init__()

        self.subscale=int(1/subscale)
        self.alpha = alpha

    def forward(self,feature1,feature2):
        feature1=torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2=torch.nn.AvgPool2d(self.subscale)(feature2)
        
        m_batchsize, C, height, width = feature1.size()
        feature1 = feature1.view(m_batchsize, -1)  #[N,C*W*H]
        feature2 = feature2.view(m_batchsize, -1)  #[N,C*W*H]
        
        ###ccam
        feature1 = F.normalize(feature1, dim=1)
        feature2 = F.normalize(feature2, dim=1)
        
        # L2norm=torch.norm(feature1,2,1,keepdim=True).repeat(1,C,1)   #[N,1,W*H]
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)  #haven't implemented in torch 0.4.1, so i use repeat instead
        # feature1=torch.div(feature1,L2norm)
        mat1 = torch.matmul(feature1,feature2.T) #[N,N]

        mat1 = torch.clamp(mat1, min=0.0005, max=0.9995)

        loss = -torch.log(mat1)
        loss[loss < 0] = 0
        _, indices = mat1.sort(descending=True, dim=1) #indices.shape = [2, 16, 16]
        _, rank = indices.sort(dim=1)
        rank = rank - 1
        rank_weights = torch.exp(-rank.float() * self.alpha)
        loss = loss * rank_weights

        return torch.mean(loss)

class FIALoss_min(torch.nn.Module):

    def __init__(self,subscale=0.0625):
        super(FIALoss_min,self).__init__()
        self.subscale=int(1/subscale)

    def forward(self,feature1,feature2):
        feature1=torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2=torch.nn.AvgPool2d(self.subscale)(feature2)
        
        m_batchsize, C, height, width = feature1.size()
        feature1 = feature1.view(m_batchsize, -1)  #[N,C*W*H]
        feature2 = feature2.view(m_batchsize, -1)  #[N,C*W*H]
        
        ###ccam
        feature1 = F.normalize(feature1, dim=1)
        feature2 = F.normalize(feature2, dim=1)

        # L2norm=torch.norm(feature1,2,1,keepdim=True).repeat(1,C,1)   #[N,1,W*H]
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)  #haven't implemented in torch 0.4.1, so i use repeat instead
        # feature1=torch.div(feature1,L2norm)
        mat1 = torch.matmul(feature1,feature2.T) #[N,N]
        mat1 = torch.clamp(mat1, min=0.0005, max=0.9995)
        
        loss = -torch.log(1 - mat1)

        return torch.mean(loss)

#### FIGALosss是指Instance,即在中间过程.view(batchsize, -1), 这样是为了像CCAM那样构造Instace_level的loss
class FIGALoss_max(torch.nn.Module):

    def __init__(self, alpha=0.25, subscale=0.0625):
        super(FIGALoss_max,self).__init__()

        self.subscale=int(1/subscale)
        self.alpha = alpha

    def forward(self,feature1,feature2):
        feature1=torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2=torch.nn.AvgPool2d(self.subscale)(feature2)
        
        m_batchsize, C, height, width = feature1.size()

        feature1 = rearrange(feature1, 'b c h w -> (b h w) c') #[N*W*H,C]
        feature2 = rearrange(feature2, 'b c h w -> (b h w) c') #[N*W*H,C]

        # print(f"the shape of feature1 is {feature1.shape}")
        # print(f"the shape of feature2 is {feature2.shape}")
        # import pdb
        # pdb.set_trace()
        
        ###ccam
        feature1 = F.normalize(feature1, dim=1)
        feature2 = F.normalize(feature2, dim=1)
        
        # L2norm=torch.norm(feature1,2,1,keepdim=True).repeat(1,C,1)   #[N,1,W*H]
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)  #haven't implemented in torch 0.4.1, so i use repeat instead
        # feature1=torch.div(feature1,L2norm)
        mat1 = torch.matmul(feature1,feature2.T) #[N*W*H,N*W*H]

        mat1 = torch.clamp(mat1, min=0.0005, max=0.9995)

        loss = -torch.log(mat1)
        loss[loss < 0] = 0
        _, indices = mat1.sort(descending=True, dim=1) #indices.shape = [2, 16, 16]
        _, rank = indices.sort(dim=1)
        rank = rank - 1
        rank_weights = torch.exp(-rank.float() * self.alpha)
        loss = loss * rank_weights

        return torch.mean(loss)

class FIGALoss_min(torch.nn.Module):

    def __init__(self,subscale=0.0625):
        super(FIGALoss_min,self).__init__()
        self.subscale=int(1/subscale)

    def forward(self,feature1,feature2):
        feature1=torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2=torch.nn.AvgPool2d(self.subscale)(feature2)
        
        m_batchsize, C, height, width = feature1.size()
        feature1 = rearrange(feature1, 'b c h w -> (b h w) c') #[N*W*H,C]
        feature2 = rearrange(feature2, 'b c h w -> (b h w) c') #[N*W*H,C]
        
        ###ccam
        feature1 = F.normalize(feature1, dim=1)
        feature2 = F.normalize(feature2, dim=1)

        # L2norm=torch.norm(feature1,2,1,keepdim=True).repeat(1,C,1)   #[N,1,W*H]
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)  #haven't implemented in torch 0.4.1, so i use repeat instead
        # feature1=torch.div(feature1,L2norm)
        mat1 = torch.matmul(feature1,feature2.T) #[N*W*H,N*W*H]
        mat1 = torch.clamp(mat1, min=0.0005, max=0.9995)
        
        loss = -torch.log(1 - mat1)

        return torch.mean(loss)


#### 计算max时也不用attn
class FALoss_max_NA(torch.nn.Module):

    def __init__(self, alpha=0.25, subscale=0.0625):
        super(FALoss_max_NA,self).__init__()

        self.subscale=int(1/subscale)
        self.alpha = alpha

    def forward(self,feature1,feature2):
        feature1=torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2=torch.nn.AvgPool2d(self.subscale)(feature2)
        
        m_batchsize, C, height, width = feature1.size()
        feature1 = feature1.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        feature2 = feature2.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        
        ###ccam
        feature1 = F.normalize(feature1, dim=1)
        feature2 = F.normalize(feature2, dim=1)

        # L2norm=torch.norm(feature1,2,1,keepdim=True).repeat(1,C,1)   #[N,1,W*H]
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)  #haven't implemented in torch 0.4.1, so i use repeat instead
        # feature1=torch.div(feature1,L2norm)
        mat1 = torch.bmm(feature1.permute(0,2,1),feature2) #[N,W*H,W*H]

        mat1 = torch.clamp(mat1, min=0.0005, max=0.9995)

        loss = -torch.log(mat1)
        loss[loss < 0] = 0

        return torch.mean(loss)


if __name__ == "__main__":

    loss = FALoss_max().cuda()

    feat_1 = torch.randn(2,64,64,64).cuda()
    feat_2 = torch.randn(2,64,64,64).cuda()

    out = loss(feat_1, feat_2)

    print("")