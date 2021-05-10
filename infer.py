import argparse
from torch.utils.data import DataLoader
from model.resnet18 import MyResnet
import torch
from torchvision import transforms as tran
import cv2
from tqdm import tqdm
from utils.seg_metrics import Seg_metrics
import numpy as np
from data_api.data_image_label import Data_image_label
from data_api.get_dataset_mean_std import get_mean_std
import os
import shutil
from glob import glob
from grad_cam import *

# base_path = os.path.dirname(os.getcwd())
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
                         help="the p500ath of trained model")

opt = test_parser.parse_args()
print(opt.device)
# 获取数据集的均值、方差
Norm, [mean, std] = get_mean_std(mean_std_path=os.path.join(base_path, "data_api"))
test_data = Data_image_label(root=opt.data_root, updata_txt=False,
                             train_test_ratio=0.8, phase="test", Norm=Norm)
test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)

# model
model = MyResnet()
# model = MyResnet()

# device_ids = [0]
# model = torch.nn.DataParallel(model, device_ids=device_ids)
# model = model.cuda(device=device_ids[0])

model.eval()
checkpoints = torch.load(os.path.join(base_path, opt.state_path))
# model.load_state_dict(checkpoints["model"])
model.load_state_dict(checkpoints)

trans = tran.ToPILImage()
metrics = Seg_metrics(num_classes=2)
result_acc = []
result_iou = []

c_name = ["TP", "FP", "FN", "TN"]
# 清空文件夹
shutil.rmtree(base_path + '/checkpoints/test_result')
os.mkdir(base_path + '/checkpoints/test_result')

# cam可视化
cam = GradCAM(model=model, cam_layer="feature.7.1.bn2")
output_path = 'checkpoints/cam_output'
if not os.path.exists(output_path):
    os.makedirs(output_path)
cam_display(cam=cam, visual_data=test_loader)
model.to(opt.device)

for i, (x, y, image_path) in enumerate(tqdm(test_loader)):
    pre = model(x.to(opt.device))
    pre_y = torch.argmax(pre, dim=1)
    metrics.add_batch(pre_y.cpu(), y.cpu())
    acc = metrics.pixelAccuracy()
    # 根据混淆矩阵对分类结果进行保存，对应TP、 FP、 FN、 TN
    confusionMatrix = metrics.confusionMatrix
    metrics.reset()
    confusionMatrix = confusionMatrix.reshape(1, -1)
    image_save_path = base_path + '/checkpoints/test_result/{}'.format(c_name[np.argmax(confusionMatrix, axis=1)[0]])
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    image_save_path = os.path.join(image_save_path, image_path[0].split("/")[-1])
    image = cv2.imread(image_path[0])
    cv2.imwrite(image_save_path, image)
    result_acc.append(acc)

result = 0
for acc in result_acc:
    result += acc
print("acc:{}".format(round(result * 100 / len(result_acc), 2)))


TP = len(glob(base_path + '/checkpoints/test_result/TP/*'))
TN = len(glob(base_path + '/checkpoints/test_result/TN/*'))
FP = len(glob(base_path + '/checkpoints/test_result/FP/*'))
FN = len(glob(base_path + '/checkpoints/test_result/FN/*'))
# 真阳性率,漏检率
TPR = round(FN/(TP+FN), 2)
# 假阳性率，误检率
FPR = round(FP/(FP+TN), 2)
print("漏检率：{}, 误检率: {}".format(TPR, FPR))



