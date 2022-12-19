import torch
from torch.nn import MSELoss
from torch.optim import RMSprop

import tnis


def train():
    discriminator = tnis.Discriminator()
    model = tnis.TNIS()

    optimizer = RMSprop(model.parameters(), lr=1e-3)
    optimizer_d = RMSprop(discriminator.parameters(), lr=1e-3)

    num_epochs = 10
    train_dl = None
    for epoch in range(num_epochs):
        for batch in train_dl:
            # 取出圖像和標籤
            images, labels = batch

            # 訓練 model
            model.train()
            discriminator.eval()
            optimizer.zero_grad()
            # 輸入圖像進 TNIS 得到調色後的圖像
            colored_images = model(images)
            loss = torch.mean(discriminator(colored_images))
            loss.backward()
            optimizer.step()

            # 訓練 discriminator
            discriminator.train()
            model.eval()
            optimizer_d.zero_grad()
            colored_images = model(images)
            loss = torch.mean(discriminator(colored_images)) - torch.mean(discriminator(images))
            loss.backward()
            optimizer_d.step()


def train_bak():
    torch.autograd.set_detect_anomaly(True)
    model = tnis.TNIS()
    discriminator = tnis.Discriminator()
    x = torch.rand((32, 3, 224, 224))
    sample = torch.rand((32, 3, 224, 224))
    x.requires_grad = True
    sample.requires_grad = True
    y, p = model(x)
    y_dis = discriminator(y)
    sample_dis = discriminator(sample)
    loss = -torch.sum(y)
    dis_loss = -torch.sum(sample_dis) + torch.sum(y_dis)
    loss.backward()
    dis_loss.backward()


train()
