import sys, os
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import model
import importlib
import torch
import torch.optim as optim
import yaml
import ctools
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
import argparse
# 使用 Tensorboard 可视化
from torch.utils.tensorboard import SummaryWriter
import time

def main(config):
    # ===============================> Setup <================================

    # 导入读取数据文件：reader/read_data.py
    dataloader = importlib.import_module("reader." + config.reader)
    device = config.device
    torch.cuda.set_device(device)
    cudnn.benchmark = True
    # 数据集存储路径
    data = config.data
    # 训练结果保存路径
    save = config.save
    # 训练过程参数
    params = config.params

    
    print("===> Read data <===")
    # folder 为数组，[p00.label, p01.label, ..., p14.label]
    data, folder = ctools.readLabelFolder(
                        data, 
                        [config.person], 
                        reverse=True
                    )
    # 如 p00.label
    savename = folder[config.person]
    # 创建 DataLoader
    dataset = dataloader.loader(
                    data,
                    params.batch_size, 
                    shuffle=True, 
                    num_workers=6
                )


    print("===> Model building <===")
    # 初始化网络模型
    net = model.Model()
    # 切换到训练模型
    net.train()
    # 将模型迁移到 cuda
    net.to(device)

    
    # 加载预训练模型
    print("===> Loading Pre-trained model <===")
    pretrain = config.pretrain
    if pretrain.enable and pretrain.device:
        net.load_state_dict(
                torch.load(
                    pretrain.path, 
                    map_location={f"cuda:{pretrain.device}": f"cuda:{device}"}
                )
            )
    elif pretrain.enable and not pretrain.device:
            net.load_state_dict(
                torch.load(pretrain.path)
            )


    print("===> optimizer building <===")
    optimizer = optim.Adam(
                    net.parameters(),
                    lr=params.lr, 
                    betas=(0.9, 0.95)
                )

    scheduler = optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=params.decay_step, 
                    gamma=params.decay
                )

    if params.warmup:
        scheduler = GradualWarmupScheduler(
                        optimizer,
                        multiplier=1,
                        total_epoch=params.warmup,
                        after_scheduler=scheduler
                    )

    # 模型保存位置
    savepath = os.path.join(save.metapath, save.folder, f"checkpoint/{savename}")
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Tensorboard 使用步骤 2，将模型写入 tensorboard 中
    face_img = torch.zeros((1, 3, 224, 224), device=device)
    face_input = edict({'face': face_img, 'name': torch.zeros((1), device=device)})
    tb_writer.add_graph(net, face_input)

    # =======================================> Training < ==========================
    print("===> Training <===")
    # 数据集长度
    length = len(dataset)
    # 计算训练总时长
    total = length * params.epoch
    timer = ctools.TimeCounter(total)


    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()


    # 训练结果写入日志
    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        outfile.write(ctools.DictDumps(config) + '\n')

        for epoch in range(1, params.epoch+1):
            for i, (data, anno) in enumerate(dataset):

                # ------------------forward--------------------
                # 将输入人脸图像迁移到 cuda
                data["face"] = data["face"].to(device)
                # 真实标签
                anno = anno.to(device)
                # 根据输入计算预测值后，计算真实值与预测值的损失
                loss = net.loss(data, anno)

                # -----------------backward--------------------
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 计算剩余时间
                rest = timer.step()/3600

                # Tensorboard 使用步骤 3，将 batch 训练的 loss 写入到 tensorboard 中
                tb_writer.add_scalar('train/train_batch_loss', loss, i + (epoch - 1) * length)

                # -----------------loger----------------------
                if i % 20 == 0:
                    log = f"[{epoch}/{params.epoch}]: " + \
                          f"[{i}/{length}] " +\
                          f"loss:{loss}" +\
                          f"lr:{ctools.GetLR(optimizer)} " +\
                          f"rest time:{rest:.2f}h"

                    print(log)
                    outfile.write(log + "\n")
                    sys.stdout.flush()
                    outfile.flush()

            scheduler.step()

            if epoch % save.step == 0:
                torch.save(
                        net.state_dict(), 
                        os.path.join(savepath, f"Iter_{epoch}_{save.model_name}.pt")
                    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')
    # 指定训练使用的配置文件
    parser.add_argument('-s', '--train', type=str,
                        help='The source config for training.')
    # 指定训练使用第几人的数据
    parser.add_argument('-p', '--person', type=int,
                        help='The tested person.')
    args = parser.parse_args()

    # 加载训练使用的配置文件。config 类型为 EasyDict
    config = edict(yaml.load(open(args.train), Loader=yaml.FullLoader))

    config = config.train
    config.person = args.person
    
    print("=====================>> (Begin) Training params << =======================")
    # 可视化打印训练集配置文件
    print(ctools.DictDumps(config))
    print("=====================>> (End) Traning params << =======================")

    # Tensorboard 使用步骤 1，实例化 SummaryWriter 对象
    cur_time = time.time()
    tb_writer = SummaryWriter(log_dir=f'runs/{cur_time}')

    main(config)

