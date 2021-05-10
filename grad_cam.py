from torch.nn import functional as F
from torchvision.models import resnet18
import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from utils.visualization import Concat3CImage


class GradCAM(object):
    def __init__(self, model, cam_layer):
        self.model = model
        self.gradient = None
        self.feature = None
        self.model.eval()
        self.handles = []
        self.__init_hook(cam_layer)

    def __init_hook(self, cam_layer):
        for name, module in self.model.named_modules():
            if name == cam_layer:
                self.handles.append(module.register_backward_hook(self.get_grad_hook))
                self.handles.append(module.register_forward_hook(self.get_feature_hook))

        assert len(self.handles) == 2

    def __call__(self, inputs, label=None):
        """

        :param inputs: torch.tensor [C, H, W]
        :param label: torch.tensor or int, only accept scalar.
        :return: torch.tensor [1, H, W]
        """
        self.model.zero_grad()
        y = self.model(inputs)
        if label is None:
            label = np.argmax(y.cpu().data.numpy())
        target = y[0][label]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()
        weight = np.mean(gradient,axis=(1,2))
        feature = self.feature[0].cpu().data.numpy()
        # print(feature)

        cam = feature*weight[:, np.newaxis, np.newaxis]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)    # Relu
        # normalization
        Max = np.max(cam)
        Min = np.min(cam)
        cam = (cam - Min) / (Max - Min)
        cam = cv2.resize(cam, (512,512))
        return cam

    def get_grad_hook(self, module, input_grad, output_grad):
        # assert len(output_grad) == 1 and self.gradient is None
        self.gradient = output_grad[0]

    def get_feature_hook(self, module, input, output):
        self.feature = output

    def remove_handlers(self):
        for handle in self.handles:
            handle.remove()


def gen_heatmap(mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask),cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)/255
    cam = heatmap/np.max(heatmap)
    cam = np.uint8(cam*255)
    return cam


def cam_display(cam=None, visual_data=None):
    assert visual_data is not None
    assert cam is not None
    for i, (x, y, image_path) in tqdm(enumerate(visual_data)):
        if y == 1:
            output = cam(x, label=1)
            img = np.uint8(output * 255)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(360, 720))
            src = cv2.imread(image_path[0])
            src = cv2.resize(src, dsize=(360, 720))
            heatmap = gen_heatmap(output)

            heatmap = cv2.resize(heatmap, dsize=(360, 720))
            # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            image_list = [src, img, heatmap]
            image = Concat3CImage(image_list, mode='Row', offset=5, fill_color=(0, 0, 255))
            cv2.imwrite('checkpoints/cam_output/{}.png'.format(i), image)
    cam.remove_handlers()


if __name__ == "__main__":
    grad_cam = GradCAM(model=resnet18(pretrained=False), cam_layer="layer4.1.bn2")
    print(resnet18(pretrained=False))
    cam = grad_cam(torch.randn([3,512,512]), 1)
    print(cam.shape)
    # for name, module in resnet18().named_modules():
    # print(name)
    #print(cam)

