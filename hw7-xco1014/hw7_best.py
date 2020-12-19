import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

import torch.nn.functional as F
import sys

patharg =sys.argv
input_dir = patharg[1]
output_file =  patharg[2]


class StudentNet(nn.Module):
    '''
      在這個Net裡面，我們會使用Depthwise & Pointwise Convolution Layer來疊model。
      你會發現，將原本的Convolution Layer換成Dw & Pw後，Accuracy通常不會降很多。

      另外，取名為StudentNet是因為這個Model等會要做Knowledge Distillation。
    '''

    def __init__(self, base=16, width_mult=1):
        '''
          Args:
            base: 這個model一開始的ch數量，每過一層都會*2，直到base*16為止。
            width_mult: 為了之後的Network Pruning使用，在base*8 chs的Layer上會 * width_mult代表剪枝後的ch數量。        
        '''
        super(StudentNet, self).__init__()
        multiplier = [1, 2, 4, 8, 16, 16, 16, 16]

        # bandwidth: 每一層Layer所使用的ch數量
        bandwidth = [ base * m for m in multiplier]

        # 我們只Pruning第三層以後的Layer
        for i in range(3, 7):
            bandwidth[i] = int(bandwidth[i] * width_mult)

        self.cnn = nn.Sequential(
            # 第一層我們通常不會拆解Convolution Layer。
            nn.Sequential(
                nn.Conv2d(3, bandwidth[0], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[0]),
                nn.ReLU6(),
                nn.MaxPool2d(2, 2, 0),
            ),
            # 接下來每一個Sequential Block都一樣，所以我們只講一個Block
            nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[0], 3, 1, 1, groups=bandwidth[0]),
                # Batch Normalization
                nn.BatchNorm2d(bandwidth[0]),
                # ReLU6 是限制Neuron最小只會到0，最大只會到6。 MobileNet系列都是使用ReLU6。
                # 使用ReLU6的原因是因為如果數字太大，會不好壓到float16 / or further qunatization，因此才給個限制。
                nn.ReLU6(),
                # Pointwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[1], 1),
                # 過完Pointwise Convolution不需要再做ReLU，經驗上Pointwise + ReLU效果都會變差。
                nn.MaxPool2d(2, 2, 0),
                # 每過完一個Block就Down Sampling
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[1], bandwidth[1], 3, 1, 1, groups=bandwidth[1]),
                nn.BatchNorm2d(bandwidth[1]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[1], bandwidth[2], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[2], bandwidth[2], 3, 1, 1, groups=bandwidth[2]),
                nn.BatchNorm2d(bandwidth[2]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[2], bandwidth[3], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            # 到這邊為止因為圖片已經被Down Sample很多次了，所以就不做MaxPool
            nn.Sequential(
                nn.Conv2d(bandwidth[3], bandwidth[3], 3, 1, 1, groups=bandwidth[3]),
                nn.BatchNorm2d(bandwidth[3]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[3], bandwidth[4], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[4], bandwidth[4], 3, 1, 1, groups=bandwidth[4]),
                nn.BatchNorm2d(bandwidth[4]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[4], bandwidth[5], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[5], bandwidth[5], 3, 1, 1, groups=bandwidth[5]),
                nn.BatchNorm2d(bandwidth[5]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[5], bandwidth[6], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[6], bandwidth[6], 3, 1, 1, groups=bandwidth[6]),
                nn.BatchNorm2d(bandwidth[6]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[6], bandwidth[7], 1),
            ),

            # 這邊我們採用Global Average Pooling。
            # 如果輸入圖片大小不一樣的話，就會因為Global Average Pooling壓成一樣的形狀，這樣子接下來做FC就不會對不起來。
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            # 這邊我們直接Project到11維輸出答案。
            nn.Linear(bandwidth[7], 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


import re
import torch
from glob import glob
from PIL import Image
import torchvision.transforms as transforms

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, folderName, transform=None):
        self.transform = transform
        self.data = []
        self.label = []

        for img_path in sorted(glob(folderName + '/*.jpg')):
            try:
                # Get classIdx by parsing image path
                class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
            except:
                # if inference mode (there's no answer), class_idx default 0
                class_idx = 0

            image = Image.open(img_path)
            # Get File Descriptor
            image_fp = image.fp
            image.load()
            # Close File Descriptor (or it'll reach OPEN_MAX)
            image_fp.close()

            self.data.append(image)
            self.label.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]


trainTransform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
testTransform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

def get_dataloader(mode='training', batch_size=32):

    assert mode in ['training', 'testing', 'validation']

    dataset = MyDataset(
        os.path.join(input_dir,mode),
        transform=trainTransform if mode == 'training' else testTransform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'))

    return dataloader


import pickle
import numpy as np
import lzma


bit = 9
def convert(number_array):
    temp = np.unpackbits(number_array.view(np.uint8))
    bit_array = temp.reshape(-1,2,8)[:,::-1].reshape(-1,16)
    bit_array = bit_array[:,16-bit:]
    bit_f  = bit_array.flatten()
    new_int = np.uint8(np.packbits(bit_f))
    return new_int

def inconvert(number_array,shape):
    zeroshape = list(shape)
    size = 1
    for i in shape:
        size *= i
    temp  = np.unpackbits(number_array)
    bit_array = temp[:size*bit]
    newshape = list(shape)
    newshape.append(bit)
    bit_array = bit_array.reshape(newshape)
    zeroshape.append(16-bit)
    bit_array  = np.concatenate([np.zeros(zeroshape),bit_array],axis = -1)
    ori_int = np.packbits(bit_array.astype(int).reshape(-1, 2, 8)[:, ::-1]).view(np.uint16).reshape(shape)
    return ori_int

      
def encode8(params, fname):
    custom_dict = {}
    t=0
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        if type(param) == np.ndarray:
            min_val = np.min(param)
            max_val = np.max(param)
            shape = param.shape
            param = np.round((param - min_val) / (max_val - min_val) * (2**bit-1))
            param = np.uint16(param)
            param =  convert(param)
            custom_dict[name] = (min_val, max_val,shape, param)
        else:
            custom_dict[name] = param


    pickle.dump(custom_dict, open(fname, 'wb'))
    with open("test.bin", 'wb') as fp:
        fp.write(lzma.compress(pickle.dumps(custom_dict,4)))
    

def decode8(fname):
    with open(fname, 'rb') as fp:
       data = lzma.decompress(fp.read())
    params = pickle.loads(data)
    custom_dict = {}
    t=1
    for (name, param) in params.items():
        if type(param) == tuple:
            min_val, max_val,shape, param = param
            param = inconvert(param,shape)
            param = np.float64(param)
            param = (param / (2**bit-1) * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)
        custom_dict[name] = param

    return custom_dict

test_dataloader = get_dataloader('testing', batch_size=32)

model_best = StudentNet(base=16).cuda()
model_best.load_state_dict(decode8('model.bin'))

model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        test_pred = model_best(data[0].cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)


with open(output_file, 'w') as f:
    f.write('Id,label\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))

print("done")