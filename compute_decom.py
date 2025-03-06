from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
                                    padding=1, padding_mode='replicate')
    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)
        feats0 = self.net1_conv0(input_img)
        featss = self.net1_convs(feats0)
        outs = self.net1_recon(featss)
        R = torch.sigmoid(outs[:, 0:3, :, :])
        L = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L

def preprocess(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    # image /= 255.0
    return image

if __name__ == '__main__':
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # dark_image_dir = Path('/home/ubuntu/lowlight-texts/datasets/DarkFace_Train_2021/image')
    # R_save_dir = Path('/home/ubuntu/lowlight-texts/datasets/DarkFace_Train_2021/decom_R')
    # L_save_dir = Path('/home/ubuntu/lowlight-texts/datasets/DarkFace_Train_2021/decom_L')

    # dark_image_dir = Path('/home/ubuntu/lowlight-texts/datasets/DarkFace_Train_2021/image')
    # R_save_dir = Path('/home/ubuntu/lowlight-texts/datasets/DarkFace_Train_2021/decom_R')
    # L_save_dir = Path('/home/ubuntu/lowlight-texts/datasets/DarkFace_Train_2021/decom_L')

    # R_save_dir.mkdir(exist_ok=True)
    # L_save_dir.mkdir(exist_ok=True)

    # dark_images = sorted(dark_image_dir.glob('*'))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # decomnet = DecomNet()
    # decomnet.load_state_dict(torch.load('weights/DecomNet.pth'))
    # decomnet.eval()
    # decomnet.to(device)

    # for dark_image in dark_images:
    #     image = preprocess(dark_image, transform)

    #     with torch.no_grad():
    #         R, L = decomnet(image.to(device))
    #     R = R.squeeze(0)
    #     L = L.squeeze(0)
    #     R = R.permute(1, 2, 0).cpu().numpy()
    #     L = L.permute(1, 2, 0).cpu().numpy()
    #     L = L.repeat(3, axis=2)
    #     R = Image.fromarray((R * 255).astype('uint8'))
    #     L = Image.fromarray((L * 255).astype('uint8'))
    #     R.save(R_save_dir / dark_image.name)
    #     L.save(L_save_dir / dark_image.name)
    #     print(f'{dark_image} decomposed')

    # dark_image_dir = Path('/home/ubuntu/lowlight-texts/nuimages/night/train2017')
    # R_save_dir = Path('/home/ubuntu/lowlight-texts/nuimages/night_R/train2017')
    # L_save_dir = Path('/home/ubuntu/lowlight-texts/nuimages/night_L/train2017')

    # dark_image_dir = Path('/home/ubuntu/lowlight-texts/nuimages/night/val2017')
    # R_save_dir = Path('/home/ubuntu/lowlight-texts/nuimages/night_R/val2017')
    # L_save_dir = Path('/home/ubuntu/lowlight-texts/nuimages/night_L/val2017')

    # dark_image_dir = Path('/home/ubuntu/lowlight-texts/nuimages/total/train2017')
    # R_save_dir = Path('/home/ubuntu/lowlight-texts/nuimages/total_R/train2017')
    # L_save_dir = Path('/home/ubuntu/lowlight-texts/nuimages/total_L/train2017')

    # dark_image_dir = Path('/home/ubuntu/lowlight-texts/nuimages/total/val2017')
    # R_save_dir = Path('/home/ubuntu/lowlight-texts/nuimages/total_R/val2017')
    # L_save_dir = Path('/home/ubuntu/lowlight-texts/nuimages/total_L/val2017')

    dark_image_dir = Path('LOD/RGB_Dark')
    R_save_dir = Path('LOD/RGB_Dark_R')
    L_save_dir = Path('LOD/RGB_Dark_L')

    R_save_dir.mkdir(exist_ok=True, parents=True)
    L_save_dir.mkdir(exist_ok=True, parents=True)

    dark_images = sorted(dark_image_dir.glob('*'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    decomnet = DecomNet()
    decomnet.load_state_dict(torch.load('weights/DecomNet.pth'))
    decomnet.eval()
    decomnet.to(device)

    for dark_image in dark_images:
        image = preprocess(dark_image, transform)

        with torch.no_grad():
            R, L = decomnet(image.to(device))
        R = R.squeeze(0)
        L = L.squeeze(0)
        R = R.permute(1, 2, 0).cpu().numpy()
        L = L.permute(1, 2, 0).cpu().numpy()
        L = L.repeat(3, axis=2)
        R = Image.fromarray((R * 255).astype('uint8'))
        L = Image.fromarray((L * 255).astype('uint8'))
        R.save(R_save_dir / dark_image.name)
        L.save(L_save_dir / dark_image.name)
        print(f'{dark_image} decomposed')