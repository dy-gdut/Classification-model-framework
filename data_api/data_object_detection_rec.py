from torch.utils.data import Dataset
from xml.dom.minidom import parse
import numpy as np
from glob import glob
import os
import cv2
import torch
from PIL import Image
from utils.mytransform import *


def xml2label(xml_path):
    dom_tree = parse(xml_path)
    root_node = dom_tree.documentElement
    indexs = ["xmin", "ymin", "xmax", "ymax"]
    index_nodes = [root_node.getElementsByTagName(index) for index in indexs]
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    params = [xmin, ymin, xmax, ymax]
    for cnt, index_node in enumerate(index_nodes):
        for param in index_node:
            params[cnt].append(param.childNodes[0].data)

    h_node = root_node.getElementsByTagName("height")
    height = next(iter(h_node)).childNodes[0].data
    w_node = root_node.getElementsByTagName("width")
    width = next(iter(w_node)).childNodes[0].data
    loader = np.zeros(shape=[int(height), int(width)])
    for x1, y1, x2, y2 in zip(xmin, ymin, xmax, ymax):
        loader[int(y1):int(y2), int(x1):int(x2)] = 1
    label_path = xml_path.replace(".xml", "_label.bmp").replace("\\", "/")

    cv2.imwrite(label_path, loader * 255)
    # cv2.imshow("a", loader)
    # cv2.imwrite("1.bmp", loader*255)
    # cv2.waitKey()
    # exit()
    return label_path


class Data_api_rec(Dataset):
    def __init__(self, root, gen_label=True, updata_txt=True, train_test_ratio=0.8, phase="test",
                 transform=None, Norm=None, use_relative_path=False):
        """

        :param root: 加载数据集的路径（数据集的格式为 image、label，其中label是存放缺陷图片的xml标注）
        :param gen_label: 是否根据xml生成对应的label图
        :param updata_txt: 是否更新txt文件
        :param train_test_ratio: 默认值为0.8,其中80%作为训练集和验证集，20%作为测试集
        :param phase: 指定训练集的模式，in ["train", "test", "val"]
        :param transform: 数据增强部分，可以指定
        :param Norm: 数据归一化，先根据数据分布计算mean、std，外部指定
        :param use_relative_path: 在txt中保存的路径是否使用相对路径，要配合updata_txt使用
        """
        super(Data_api_rec, self).__init__()
        assert phase in ["train", "test", "val"]
        self.root = root
        self.phase = phase
        self.use_relative_path = use_relative_path
        if gen_label:
            self.images_path = glob(os.path.join(self.root, "image", '*.bmp'))
            # image会被替换为label，所以其他文件夹名字不要出现image，否则会出现错误路径
            self.images_xml = [image_path.replace(".bmp", ".xml").replace("image", "label")
                               for image_path in self.images_path]

            self.images_xml = [image_xml for image_xml in self.images_xml if os.path.exists(image_xml)]
            self.labels_path = [xml2label(image_xml) for image_xml in self.images_xml]

        if not os.path.exists(os.path.join(self.root, "train.txt")) or updata_txt:
            self.images_path = self.make_txt(train_test_ratio=train_test_ratio)
        self.images_path = self.get_image_list()

        self.Norm = Norm
        if not transform:
            if self.phase == "train":
                self.trans = GroupCompose([GroupResize([128, 768]),
                                           GroupRandomHorizontalFlip(p=0.5),
                                           GroupRandomVerticalFlip(p=0.5),
                                           GroupRandomHorizontalMove(),
                                           GroupToTensor()])
            else:
                self.trans = GroupCompose([GroupResize([128, 768]),
                                           GroupToTensor()])
        else:
            self.trans = transform

    def make_txt(self, train_test_ratio=0.5):
        images_path = glob(os.path.join(self.root, "image", '*.bmp'))
        if self.use_relative_path:
            dataset_name = self.root.split("/")[-1]
            images_path = [dataset_name + image_path.split(dataset_name)[1] for image_path in images_path]
        random.shuffle(images_path)
        seg_point = int(len(images_path) * train_test_ratio)
        train_list = images_path[0:int(0.8*seg_point)]
        test_list = images_path[seg_point:len(images_path)]
        val_list = images_path[int(0.8*seg_point):int(seg_point)]
        with open(os.path.join(self.root, "train.txt"), mode='w') as f:
            for i in range(len(train_list)):
                f.write(train_list[i] + "\n")
        with open(os.path.join(self.root, "test.txt"), mode='w') as f:
            for i in range(len(test_list)):
                f.write(test_list[i] + "\n")
        with open(os.path.join(self.root, "val.txt"), mode='w') as f:
            for i in range(len(val_list)):
                f.write(val_list[i] + "\n")
        return images_path

    def get_image_list(self):
        with open(os.path.join(self.root, self.phase + ".txt"), mode='r') as f:
            images = f.readlines()
        images = [img.strip() for img in images]
        return images

    def __getitem__(self, item):
        image_path = self.images_path[item % len(self.images_path)]
        image = cv2.imread(image_path)

        label_path = (image_path.split(".bmp")[0] + "_label.bmp").replace("image", "label")

        image_array = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not os.path.exists(label_path):
            label_array = np.zeros([image_array.shape[0], image_array.shape[1]])
            image_label = torch.tensor([0])
        else:
            label_array = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            image_label = torch.tensor([1])
        image_tensor, label_tensor = self.trans([Image.fromarray(image_array), Image.fromarray(label_array)])

        if self.Norm:
            image_tensor = self.Norm(image_tensor)
        return image_tensor, label_tensor, image_path, image_label

    def __len__(self):
        return len(self.images_path)


def main():
    root = r"/media/root/文档/wqr/my_spixel_fcn/data_preprocessing/up_facet"
    dataset = Data_api_rec(root=root)
    from data_api.get_dataset_mean_std import get_mean_std
    a, b, _ = dataset[0]
    mean, std = get_mean_std(dataset, ratio=0.1)
    # print(mean)
    # print(std)
    # exit()
    # print(a.shape)
    # print(b.shape)
    from torchvision import transforms
    tran = transforms.ToPILImage()
    imga = tran(a)
    imgb = tran(b)
    imgaa = cv2.cvtColor(np.array(imga), cv2.COLOR_RGB2BGR)

    cv2.imshow("a", np.uint8(imgaa))
    cv2.imshow("b", np.uint8(imgb))
    cv2.waitKey()


if __name__ == '__main__':
    main()

