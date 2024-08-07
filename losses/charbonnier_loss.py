import torch
import torch.nn.functional as F

class CharbonnierLoss(torch.nn.Module):
    """L1 Charbonnierloss. from paper <<Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks>>"""

    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6  # 论文超参数

    def forward(self, X, Y):
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")  # Add this line
        if X.size() != Y.size():
            Y = F.interpolate(Y, size=X.size()[2:], mode='bilinear', align_corners=False)
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)  # 论文里面没有取均值，这里应该是为了数值小
        return loss
