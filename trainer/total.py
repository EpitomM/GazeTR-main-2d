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

def main(config):

    #  ===================>> Setup <<=================================

    # 配置文件参数设置
    dataloader = importlib.import_module("reader." + config.reader)
    torch.cuda.set_device(config.device) 
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
    # 切换到训练模型
    net.train()
    # 将模型迁移到 cuda
    net.cuda()


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
                    betas=(0.9,0.999)
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
                    if key != 'name': data[key] = data[key].cuda().float()

                anno = anno.cuda().long()
                loss = net.loss(data, anno)

                # -------------- Backward ------------
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 计算剩余时间
                rest = timer.step()/3600


                if i % 20 == 0:
                    log = f"[{epoch}/{params.epoch}]: " + \
                          f"[{i}/{length}] " +\
                          f"loss:{loss} " +\
                          f"lr:{ctools.GetLR(optimizer)} " +\
                          f"rest time:{rest:.2f}h"

                    print(log); outfile.write(log + "\n")
                    sys.stdout.flush(); outfile.flush()

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

    main(config.train)

