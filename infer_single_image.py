import argparse
from model.resnet18 import MyResnet
import torch
from tqdm import tqdm
from utils.seg_metrics import Seg_metrics
import numpy as np
from PIL import Image
from data_api.get_dataset_mean_std import get_mean_std
import os
import shutil
import cv2
from glob import glob
from torchvision import transforms as T

base_path = os.getcwd()
test_parser = argparse.ArgumentParser(description="testing---resnet18 on groove dataset!")
test_parser.add_argument("--data_root", '-dr', default="/media/root/软件/wqr/data/groove")
test_parser.add_argument("--phase", "-p", default="test", choices=["train", "test", "val"],
                         help="the mode of running")
test_parser.add_argument("--device", "-d", default='cuda:0' if torch.cuda.is_available() else 'cpu')
test_parser.add_argument("--batch_size", "-bs", default=1, type=int)
test_parser.add_argument("--continue_train", "-ct", default=False, type=bool,
                         help="continue train using trained model")
test_parser.add_argument("--state_path", "-sp", default="checkpoints/network_state/acc100.0_model.pth",
                         help="the path of trained model")
opt = test_parser.parse_args()
print(opt.device)
# 获取数据集的均值、方差
Norm, [mean, std] = get_mean_std(mean_std_path=os.path.join(base_path, "data_api"))


# model
model = MyResnet().to(opt.device)
model.eval()
checkpoints = torch.load(os.path.join(base_path, opt.state_path))
# model.load_state_dict(checkpoints["model"])
model.load_state_dict(checkpoints)


metrics = Seg_metrics(num_classes=2)

# 清空文件夹
shutil.rmtree(base_path + '/checkpoints/test_single_result')
os.mkdir(base_path + '/checkpoints/test_single_result')

# 遍历的文件夹
image_paths = glob("/media/root/软件/wqr/data/test_single/*.bmp")
trans = T.Compose([T.Resize([720, 320]),
                   T.ToTensor()])
trans_pil = T.ToPILImage()

# 遍历文件夾對圖片進行分类，不用数据loader
for i, image_path in enumerate(tqdm(image_paths)):
    img = cv2.imread(image_path)
    image_array = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_pil = Image.fromarray(image_array)
    # 数据处理
    img_tensor = trans(img_pil)
    img_tensor = Norm(img_tensor)
    img_tensor = img_tensor.unsqueeze(dim=0).to(opt.device)
    pre = model(img_tensor)
    # 模型输出的数据转换
    pre_y = torch.argmax(pre, dim=1)
    y = pre_y.cpu()

    if not os.path.exists(base_path + '/checkpoints/test_single_result/P/'):
        os.makedirs(base_path + '/checkpoints/test_single_result/P/')
    if not os.path.exists(base_path + '/checkpoints/test_single_result/N/'):
        os.makedirs(base_path + '/checkpoints/test_single_result/N/')

    # 根据预测分类
    if y == 0:
        cv2.imwrite(base_path + '/checkpoints/test_single_result/P/' + image_path.split("/")[-1], img)
    else:
        cv2.imwrite(base_path + '/checkpoints/test_single_result/N/' + image_path.split("/")[-1], img)







