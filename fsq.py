# -*- coding: utf-8 -*-
# PyTorch简单实现FSQ-AE

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable

if not os.path.exists('samples'):
    os.mkdir('samples')

# 基本配置
img_paths = list_pictures('/mnt/vepfs/sujianlin/CelebA-HQ/train/', 'png')
img_paths += list_pictures('/mnt/vepfs/sujianlin/CelebA-HQ/valid/', 'png')
np.random.shuffle(img_paths)
img_size = 128
batch_size = 64
embedding_size = 128
num_layers = 6
min_pixel = 16

# 超参数选择
dim_codes = 4
num_codes = 6

class ImageDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img)

def save_image(path, tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img + 1) / 2 * 255
    img = np.clip(img, 0, 255).astype('uint8')
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def data_generator(img_paths):
    dataset = ImageDataset(img_paths)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class GroupNorm(nn.Module):
    def __init__(self, num_groups=32):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups

    def forward(self, x):
        return nn.functional.group_norm(x, self.num_groups)

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.group_norm = GroupNorm()
        self.activate = nn.SiLU()
        if in_dim != out_dim:
            self.shortcut = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = x
        if self.shortcut:
            shortcut = self.shortcut(x)
        x = self.activate(self.conv1(x))
        x = self.activate(self.conv2(x))
        x = self.group_norm(x + shortcut)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(3, embedding_size, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResidualBlock(embedding_size, embedding_size) for _ in range(num_layers)])
        self.pool = nn.AvgPool2d(2)
        self.group_norm = GroupNorm()
        self.conv_out = nn.Conv2d(embedding_size, dim_codes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        skip_pooling = 0
        for i in range(num_layers):
            x = self.res_blocks[i](x)
            if min(x.shape[2:]) > min_pixel:
                x = self.pool(x)
            else:
                skip_pooling += 1
        x = self.res_blocks[-1](x)
        x = self.group_norm(x)
        x = torch.sigmoid(self.conv_out(x))
        return x

class Decoder(nn.Module):
    def __init__(self, skip_pooling):
        super(Decoder, self).__init__()
        self.conv_in = nn.Conv2d(dim_codes, embedding_size, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResidualBlock(embedding_size, embedding_size) for _ in range(num_layers)])
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.group_norm = GroupNorm()
        self.conv_out = nn.Conv2d(embedding_size, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(num_layers):
            if i >= skip_pooling:
                x = self.upsample(x)
            x = self.res_blocks[i](x)
        x = self.group_norm(x)
        x = torch.tanh(self.conv_out(x))
        return x

class FSQ_AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(FSQ_AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        xq = self.encoder(x)
        xq_round = torch.round(xq * (num_codes - 1)) / (num_codes - 1)
        xq = xq + (xq_round - xq).detach()  # stop gradient
        return self.decoder(xq)

def l2_loss(y_true, y_pred):
    return torch.sum((y_true - y_pred) ** 2)

encoder = Encoder()
encoder.eval()
sample_input = torch.randn(1, 3, img_size, img_size)
encoder(sample_input)
skip_pooling = encoder.eval().skip_pooling
decoder = Decoder(skip_pooling)
fsq_ae = FSQ_AE(encoder, decoder)
fsq_ae = fsq_ae.cuda()

optimizer = optim.Adam(fsq_ae.parameters(), lr=2e-3)

def sample_ae_1(path, n=8):
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            if j % 2 == 0:
                x_sample = [ImageDataset(np.random.choice(img_paths))]
            else:
                z_sample = encoder(x_sample)
                x_sample = decoder(z_sample)
            digit = x_sample[0]
            figure[i * img_size:(i + 1) * img_size,
                   j * img_size:(j + 1) * img_size] = digit
    save_image(path, figure)

def sample_ae_2(path, n=8):
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            if j % 2 == 0:
                x_sample = [ImageDataset(np.random.choice(img_paths))]
            else:
                z_sample = encoder(x_sample)
                x_sample = decoder(z_sample)
            digit = x_sample[0]
            figure[i * img_size:(i + 1) * img_size,
                   j * img_size:(j + 1) * img_size] = digit
    save_image(path, figure)

class Trainer:
    def __init__(self):
        self.batch = 0
        self.n_size = 9
        self.iters_per_sample = 100

    def train(self):
        fsq_ae.train()
        for epoch in range(1000):
            for batch_imgs, _ in data_generator(img_paths):
                batch_imgs = batch_imgs.cuda()
                optimizer.zero_grad()
                outputs = fsq_ae(batch_imgs)
                loss = l2_loss(batch_imgs, outputs)
                loss.backward()
                optimizer.step()
                if self.batch % self.iters_per_sample == 0:
                    sample_ae_1('samples/test_ae_1_%s.png' % self.batch)
                    sample_ae_2('samples/test_ae_2_%s.png' % self.batch)
                    torch.save(fsq_ae.state_dict(), './train_model.pth')
                self.batch += 1

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
