from torchvision.models import resnet18
import torch
from torch import nn
from torchvision.models import densenet
# from grad_cam import GradCAM


class MyResnet(nn.Module):
    def __init__(self,num_classes=2, pretrained=True):
        super(MyResnet, self).__init__()
        self.feature = nn.Sequential(*list(resnet18(pretrained=pretrained).children())[:9])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    net = MyResnet()
    # print(net)
    # for name, module in net.named_modules():
    #     print(name)
    # x = torch.rand([1,3,512,512])
    # y = net(x)
    # cam = GradCAM(model=net,cam_layer='feature.5.1.bn2')
    # output,label = cam(x[0], label=1)
    # print(output.shape)
    #print(y.shape)


