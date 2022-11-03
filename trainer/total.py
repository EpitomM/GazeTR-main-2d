import sys,os
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
from torch.utils.tensorboard import SummaryWriter
import time
import torch.nn as nn

def main(config):

    #  ===================>> Setup <<=================================

    # 配置文件参数设置
    dataloader = importlib.import_module("reader." + config.reader)
    device = config.device
    # torch.cuda.set_device(device)
    cudnn.benchmark = True
    # 数据集路径
    data = config.data
    # 结果保存路径
    save = config.save
    # 训练参数
    params = config.params


    print("===> Read data <===")
    # 读取数据集文件
    if data.isFolder:
        data, _ = ctools.readfolder(data)
    # 创建 DataLoader
    dataset = dataloader.loader(
                    data,
                    params.batch_size, 
                    shuffle=True, 
                    num_workers=8
                )


    print("===> Model building <===")
    # 初始化网络模型
    net = model.Model()
    # 用多个 GPU 加速训练
    # net = nn.DataParallel(net)
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
                    map_location={f"cuda:{pretrain.device}": f"cuda:{config.device}"}
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
                    betas=(0.9, 0.999)
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
    savepath = os.path.join(save.metapath, save.folder, f"checkpoint")
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Tensorboard 使用步骤 2，将模型写入 tensorboard 中
    face_img = torch.zeros((1, 3, 224, 224), device=device)
    face_input = edict({'face': face_img, 'name': torch.zeros((1), device=device)})
    tb_writer.add_graph(net, face_input)

    # =====================================>> Training << ====================================
    print("===> Training <===")
    # 数据集长度
    length = len(dataset)
    total = length * params.epoch
    # 计算训练总时长
    timer = ctools.TimeCounter(total)


    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()


    # 训练结果写入日志
    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        outfile.write(ctools.DictDumps(config) + '\n')

        for epoch in range(1, params.epoch+1):
            for i, (data, anno) in enumerate(dataset):

                # -------------- forward -------------
                for key in data:
                    if key != 'name':
                        data[key] = data[key].to(device).float()

                anno = anno.to(device).long()
                loss = net.loss(data, anno)

                # -------------- Backward ------------
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 计算剩余时间
                rest = timer.step()/3600

                tb_writer.add_scalar('train/train_batch_loss', loss, i + (epoch - 1) * length)

                if i % 20 == 0:
                    log = f"[{epoch}/{params.epoch}]: " + \
                          f"[{i}/{length}] " +\
                          f"loss:{loss} " +\
                          f"lr:{ctools.GetLR(optimizer)} " +\
                          f"rest time:{rest:.2f}h"

                    print(log)
                    outfile.write(log + "\n")
                    sys.stdout.flush()
                    outfile.flush()

                    # Tensorboard 使用步骤 3，将 batch 训练的 loss 写入到 tensorboard 中

            # # Tensorboard 使用步骤 4，将每个 epoch 训练的平均 loss 写入到 tensorboard 中
            # tb_writer.add_scalar('train/train_loss', train_loss_sum / train_length, epoch)
            scheduler.step()

            if epoch % save.step == 0:
                torch.save(
                        net.state_dict(), 
                        os.path.join(
                            savepath, 
                            f"Iter_{epoch}_{save.model_name}.pt"
                            )
                        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    # 训练集配置文件
    parser.add_argument('-s', '--train', type=str,
                        help='The source config for training.')

    args = parser.parse_args()

    config = edict(yaml.load(open(args.train), Loader=yaml.FullLoader))

    print("=====================>> (Begin) Training params << =======================")
    # 可视化打印训练集配置文件
    print(ctools.DictDumps(config))
    print("=====================>> (End) Traning params << =======================")

    # Tensorboard 使用步骤 1，实例化 SummaryWriter 对象
    cur_time = time.time()
    tb_writer = SummaryWriter(log_dir=f'runs/{cur_time}')

    main(config.train)

