from torch.utils.data import DataLoader
import argparse
from model.resnet18 import MyResnet
from data_api.data_image_label import Data_image_label
from torch import nn
import sys
from utils.visualization import *
from utils.seg_metrics import Seg_metrics
import matplotlib.pyplot as plt
from datetime import datetime
import os
from data_api.get_dataset_mean_std import get_mean_std

np.random.seed(0)
torch.manual_seed(0)

# 参数初始化 *********
# base_path = os.path.dirname(os.getcwd())
base_path = os.getcwd()
acc_all = []
loss_all = []
loss_mean = []
best_acc_epoch = 0

# 创建参数管理器 ******
train_parser = argparse.ArgumentParser(description="training---resnet18 on groove dataset!")
train_parser.add_argument("--data_root", '-dr', default="/media/root/软件/wqr/data/groove")
train_parser.add_argument("--phase", "-p", default="train", choices=["train", "test", "val"],
                          help="the mode of running")
train_parser.add_argument("--batch_size", "-bs", default=8, type=int)
train_parser.add_argument("--device", "-d", default='cuda:0' if torch.cuda.is_available() else 'cpu')
train_parser.add_argument("--epochs", "-epo", default=50, type=int)
train_parser.add_argument("--learning_rate", "-lr", default=0.0001, type=float)
train_parser.add_argument("--epoch_interval", "-ei", default=10, type=int)
train_parser.add_argument("--val_epoch", "-ve", default=1, type=int)
train_parser.add_argument("--continue_train", "-ct", action="store_true", help="if continue train using trained model")
train_parser.add_argument("--state_path", "-sp", default="checkpoints/network_state/network_epo149.pth",
                          help="the path of trained model")
opt = train_parser.parse_args()
print(opt.device)
# 预先保存的训练集均值方差，用作数据的归一化，加快收敛
Norm, [mean, std] = get_mean_std(mean_std_path=os.path.join(base_path, "data_api"))

train_data = Data_image_label(root=opt.data_root,  updata_txt=False,
                              train_test_ratio=0.8, phase="train", use_relative_path=False, Norm=Norm)

val_data = Data_image_label(root=opt.data_root,  updata_txt=False, train_test_ratio=0.8,
                            phase="val", Norm=Norm)

train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=True)


# model
model = MyResnet().to(opt.device)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=0.0005)

# 声明所有可用设备  多GPU
# device_ids = [0, 1]
# device_ids = [0]
# model = torch.nn.DataParallel(model, device_ids=device_ids)
# model = model.cuda(device=device_ids[0])
# loss
criterion = nn.CrossEntropyLoss()
# continue train
start_epoch = 0

if opt.continue_train:
    checkpoint = torch.load(opt.state_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(optimizer.state_dict())
    start_epoch = checkpoint["epoch"]


def main():
    # tensorboard 可视化
    TIMESTAMP = "{0:%Y-%m-%dII%H-%M-%S/}".format(datetime.now())
    log_dir = base_path + '/checkpoints/vis_log/' + TIMESTAMP
    print("The log save in {}".format(log_dir))
    Vis = VisualBoard(log_dir)
    best_acc = 0
    global loss_all
    global loss_mean
    global model
    for epoch in range(start_epoch, opt.epochs):
        model.train()
        for cnt, (x, y, image_label) in enumerate(train_loader):
            x = x.to(opt.device)
            y = y.to(opt.device)

            pre = model(x)
            loss = criterion(pre, y.long())

            # 记录loss
            loss_all.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write('\r epoch:{}-batch:{}-loss:{}'.format(epoch, cnt, loss))
            sys.stdout.flush()

        # 计算每一轮的loss
        b_loss = sum(loss_all)/len(loss_all)
        loss_mean.append(b_loss)
        loss_all = []

        # 可视化loss曲线
        Vis.visual_data_curve(name="loss", data=b_loss, data_index=epoch)

        if epoch % opt.epoch_interval == opt.epoch_interval - 1:
            network_state = {'model': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'epoch': epoch}
            torch.save(network_state, base_path + '/checkpoints/network_state/network_epo{}.pth'.format(epoch))
            print('\n save model.pth successfully!')
        # 验证模式下，关闭梯度回传以及冻结BN层，降低占用内存空间
        with torch.no_grad():
            if epoch % opt.val_epoch == opt.val_epoch - 1:
                model.eval()
                # 验证阶段，每一次返回最优acc，并保存最优acc的模型参数，同时在tensorboard上可视化recall、acc曲线
                best_acc = validate(best_acc, epoch, Vis=Vis)
                # 可视化训练集的训练效果
                acc_metrics = Seg_metrics(num_classes=2)
                for cnt, (x, y, image_label) in enumerate(train_loader):
                    pre = model(x.to(opt.device))
                    pre_y = torch.argmax(pre, dim=1)
                    acc_metrics.add_batch(y.cpu(), pre_y.cpu())
                train_acc = acc_metrics.pixelAccuracy()
                train_recall = acc_metrics.classRecall()
                print("训练集精度为：{},召回率为：{}".format(round(train_acc*100, 2), round(train_recall*100, 2)))
    Vis.visual_close()


# 保存每一次验证的loss、acc最终绘制曲线图
def dis_plt():
    global base_path
    global loss_mean
    plt.figure()
    plt.plot(np.arange(0, len(acc_all), 1)*opt.val_epoch, acc_all, marker='*', color='b', label='acc')
    plt.legend()
    # 横坐标名称
    plt.xlabel('epoch')
    # 纵坐标名称
    plt.ylabel('val_acc')
    plt.savefig(base_path + '/checkpoints/visual_result/acc_all.png')

    plt.cla()
    plt.plot(np.arange(0, len(loss_mean), 1), loss_mean, marker='*', color='b', label='loss')
    plt.legend()
    # 横坐标名称
    plt.xlabel('epoch')
    # 纵坐标名称
    plt.ylabel('loss')
    plt.savefig(base_path + '/checkpoints/visual_result/loss_all.png')


def validate(best_acc, epoch, Vis=None):
    acc_metrics = Seg_metrics(num_classes=2)
    global best_acc_epoch
    global base_path
    global model
    model.eval()
    for cnt, (x, y, image_label) in enumerate(val_loader):
        pre = model(x.to(opt.device))
        pre_y = torch.argmax(pre, dim=1)
        acc_metrics.add_batch(y.cpu(), pre_y.cpu())

    acc = acc_metrics.pixelAccuracy()
    recall = acc_metrics.classRecall()

    cur_acc = round(acc * 100, 2)
    acc_all.append(cur_acc)

    if cur_acc > best_acc:
        best_acc = cur_acc
        best_acc_epoch = epoch
        torch.save(model.state_dict(), 'checkpoints/network_state/acc{}_model.pth'.format(best_acc))
        print('save best_acc_model.pth successfully in the {} epoch!'.format(epoch))

    text_note_acc = "The best_acc gens in the {}_epoch,the best acc is {}". \
        format(best_acc_epoch, best_acc)
    text_note_recall = "the recall is {}".format(round(recall, 2))

    # 最优acc、iou保存路径提示
    Vis.writer.add_text(tag="note", text_string=text_note_acc + "||" + text_note_recall,
                        global_step=epoch)
    Vis.visual_data_curve(name="acc", data=cur_acc, data_index=epoch)
    Vis.visual_data_curve(name="recall", data=recall, data_index=epoch)
    print("\n epoch:{}-acc:{}--recall:{}".format(epoch, cur_acc, recall))
    return best_acc


if __name__ == '__main__':
    main()
    dis_plt()

