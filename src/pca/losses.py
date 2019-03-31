import torch

class SMPLLoss(torch.nn.Module):

    def __init__(self, pca):
        super(SMPLLoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.pca = pca.transpose(0,1)
        self.pca_weight = 0.0

    def decay_pca_weight(self, epoch):
        start = 0.05
        mse_weight = start + (1.0-start)**(epoch+1)
        self.pca_weight = 1.0 - mse_weight

    def forward(self, pred, target):
        mse_lss = self.mse(pred, target)
        res_lss = torch.mm(pred, self.pca) - torch.mm(target, self.pca)
        res_lss = torch.mean(torch.mean(res_lss**2, dim=1))
        return (1.0-self.pca_weight) * mse_lss + self.pca_weight*res_lss
