import os
import cv2 
import torch
import random
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def Decode_MPII(line):
    # Face Left Right Grid Origin whicheye 2DPoint HeadRot HeadTrans ratio FaceCorner LeftEyeCorner RightEyeCorner
    #  0    1    2      3    4      5        6      7           8       9       10          11          12
    anno = edict()
    # 人脸，左眼，右眼图片的存储路径
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    # 将人脸网格图片的存储路径当作 name
    anno.name = line[3]

    # (whicheye, 2DPoint)，后续没有使用这两个值
    anno.gaze3d, anno.head3d = line[5], line[6]
    # (HeadRot, HeadTrans)  后续将这 HeadRot 值当作标签
    anno.gaze2d, anno.head2d = line[7], line[8]
    return anno

def Decode_Diap(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[4], line[5]
    anno.gaze2d, anno.head2d = line[6], line[7]
    return anno

def Decode_Gaze360(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d = line[4]
    anno.gaze2d = line[5]
    return anno

def Decode_ETH(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[1]
    anno.head2d = line[2]
    anno.name = line[3]
    return anno

def Decode_RTGene(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[6]
    anno.head2d = line[7]
    anno.name = line[0]
    return anno

def Decode_Dict():
    mapping = edict()
    mapping.mpiigaze = Decode_MPII
    mapping.eyediap = Decode_Diap
    mapping.gaze360 = Decode_Gaze360
    mapping.ethtrain = Decode_ETH
    mapping.rtgene = Decode_RTGene
    return mapping

# 寻找两个字符串中相同的字符数量
# 如 abcd 与 bef 中相同的字符数量是 1
def long_substr(str1, str2):
    # str1：当前数据集名字
    # str2: 遍历的每一个数据集名字
    substr = ''
    for i in range(len(str1)):
        for j in range(len(str1)-i+1):
            if j > len(substr) and (str1[i:i+j] in str2):
                substr = str1[i:i+j]
    return len(substr)

# 获取当前训练数据集对应的 Decoder
def Get_Decode(name):
    # 存储所有数据集各自对应的 Decoder
    mapping = Decode_Dict()
    # 所有数据集的名字列表
    keys = list(mapping.keys())
    # 将当前数据集名字转为小写
    name = name.lower()
    # 找到 keys 中与当前数据集对应的名字
    # 如果当前数据集名字 name=mpii，而 keys=['mpiigaze', 'eyediap', 'gaze360', 'ethtrain', 'rtgene']，则需要将 name 与 Keys 进行字符匹配，返回匹配字符个数最多的作为当前训练的数据集名字，此时返回结果为 [4, 1, 0, 1, 0]
    score = [long_substr(name, i) for i in keys]
    key = keys[score.index(max(score))] # 如 key=mpiigaze
    return mapping[key]
    

class trainloader(Dataset): 
  def __init__(self, dataset):

    # Read source data
    self.data = edict()
    # 存储一行一行的标签数据
    self.data.line = []
    # 存储数据集图片的根目录
    self.data.root = dataset.image
    # 数据集名字，如 mpii、gaze360 等
    # 获取当前训练数据集对应的 Decoder
    self.data.decode = Get_Decode(dataset.name)

    # 如果标签文件是一个列表
    if isinstance(dataset.label, list):
      # 读取每一个人的标签文件
      for i in dataset.label:
        with open(i) as f:
            # 读取这一个人的所有标签
            line = f.readlines()
        # dataset.header=True 表示第一行是表头
        if dataset.header:
            # 去除第一行表头，保留所有标签
            line.pop(0)
        # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
        self.data.line.extend(line)
    else:
      with open(dataset.label) as f:
          self.data.line = f.readlines()
      if dataset.header:
          self.data.line.pop(0)

    # build transforms
    self.transforms = transforms.Compose([
        transforms.ToTensor()
    ])


  def __len__(self):
    return len(self.data.line)


  def __getitem__(self, idx):

    # Read souce information
    line = self.data.line[idx]
    line = line.strip().split(" ")
    anno = self.data.decode(line)

    # 读取人脸图片
    img = cv2.imread(os.path.join(self.data.root, anno.face))
    # 人脸图片转为 Tensor
    img = self.transforms(img)

    # 转换标签类别 str -> FloatTensor
    label = np.array(anno.gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)

    data = edict()
    data.face = img
    data.name = anno.name

    return data, label

def loader(source, batch_size, shuffle=True,  num_workers=0):
    dataset = trainloader(source)
    print(f"-- [Read Data]: Source: {source.label}")
    print(f"-- [Read Data]: Total num: {len(dataset)}")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load

