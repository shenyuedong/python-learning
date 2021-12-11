import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from GanModel import Generator, Discriminator


# 设置超参数
BatchSize = 8
ImageSize = 96
Epoch = 25
Lr = 0.0002
Beta1 = 0.5
DataPath = './faces/'
OutPath = './imgs/'
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(netG, netD, dataloader):
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(netG.parameters(), lr=Lr, betas=(Beta1, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=Lr, betas=(Beta1, 0.999))

    label = torch.FloatTensor(BatchSize)
    real_label = 1
    fake_label = 0

    for epoch in range(1, Epoch + 1):
        for i, (imgs, _) in enumerate(dataloader):
            # 固定生成器G，训练鉴别器D
            optimizerD.zero_grad()
            # 让D尽可能的把真图片判别为1
            imgs = imgs.to(device)
            output = netD(imgs)
            label.data.fill_(real_label)
            label = label.to(device)
            errD_real = criterion(output, label)
            errD_real.backward()
            # 让D尽可能把假图片判别为0
            label.data.fill_(fake_label)
            noise = torch.randn(BatchSize, 100, 1, 1)
            noise = noise.to(device)
            fake = netG(noise)
            # 避免梯度传到G，因为G不用更新
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_fake + errD_real
            optimizerD.step()

            # 固定鉴别器D，训练生成器G
            optimizerG.zero_grad()
            # 让D尽可能把G生成的假图判别为1
            label.data.fill_(real_label)
            label = label.to(device)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()
            if i % 50 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
                      % (epoch, Epoch, i, len(dataloader), errD.item(), errG.item()))

        vutils.save_image(fake.data,
                          '%s/fake_samples_epoch_%03d.png' % (OutPath, epoch),
                          normalize=True)
        torch.save(netG.state_dict(), '%s/netG_%03d.pth' % (OutPath, epoch))
        torch.save(netD.state_dict(), '%s/netD_%03d.pth' % (OutPath, epoch))

if __name__ == "__main__":
    # 图像格式转化与归一化
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Scale(ImageSize),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.ImageFolder(DataPath, transform=transforms)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=BatchSize,
        shuffle=True,
        drop_last=True,
    )

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    train(netG, netD, dataloader)