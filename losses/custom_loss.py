import torch
import torch.nn.functional as F

class CharbonnierLoss(torch.nn.Module):
    """L1 Charbonnier loss from the paper <<Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks>>"""

    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6  # Small constant to avoid division by zero

    def forward(self, X, Y):
        if X.size() != Y.size():
            X = F.interpolate(X, size=Y.size()[2:], mode='bilinear', align_corners=False)
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)  # Mean over all dimensions
        return loss


class CustomLoss(torch.nn.Module):
    """Pixel + FFT L1 Charbonnier loss."""

    def __init__(self):
        super(CustomLoss, self).__init__()
        self.charbonnierloss = CharbonnierLoss()
        self.l1loss = torch.nn.L1Loss()

    def forward(self, X, Y):
        # Ensure the same size for pixel loss calculation
        if X.size() != Y.size():
            X = F.interpolate(X, size=Y.size()[2:], mode='bilinear', align_corners=False)

        # Compute pixel loss
        pixel_loss = self.charbonnierloss(X, Y)
        
        # Compute FFT loss
        X_fft = torch.fft.fft2(X, dim=(-2, -1))
        Y_fft = torch.fft.fft2(Y, dim=(-2, -1))
        
        # Compute magnitudes of FFT results
        X_fft_magnitude = torch.abs(X_fft)
        Y_fft_magnitude = torch.abs(Y_fft)
        
        # If sizes of X_fft and Y_fft are different, interpolate them
        if X_fft_magnitude.size() != Y_fft_magnitude.size():
            X_fft_magnitude = F.interpolate(X_fft_magnitude.unsqueeze(0), size=Y_fft_magnitude.size()[2:], mode='bilinear', align_corners=False).squeeze(0)
            Y_fft_magnitude = F.interpolate(Y_fft_magnitude.unsqueeze(0), size=X_fft_magnitude.size()[2:], mode='bilinear', align_corners=False).squeeze(0)
        
        fft_loss = self.charbonnierloss(X_fft_magnitude, Y_fft_magnitude)
        return 0.9 * pixel_loss + 0.1 * fft_loss


if __name__ == '__main__':
    input1 = torch.randn(2, 3, 256, 256)
    input2 = torch.randn(2, 3, 256, 256)
    custom_loss = CustomLoss()
    loss = custom_loss(input1, input2)
    print("loss:", loss)
