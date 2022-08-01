import numpy as np

import torch
import torchvision
import matplotlib.pyplot as plt
from data_prepare import get_dataloader

from pycocotools.coco import COCO


class Block(torch.nn.Module):
    def __init__(self, inp_size, out_size, device):
        """
        :param inp_size: count filters of input tensor
        :param out_size: count filters of output tensor
        :param device:
        """
        super().__init__()
        # with padding=1 output shape of result
        # will be the same with input
        self.conv1 = torch.nn.Conv2d(inp_size, out_size,
                                     kernel_size=(3, 3), padding=1, device=device)
        self.conv2 = torch.nn.Conv2d(out_size, out_size,
                                     kernel_size=(3, 3), padding=1, device=device)
        self.ReLU = torch.nn.ReLU()

    def forward(self, x):
        out = self.ReLU(self.conv1(x))
        out = self.ReLU(self.conv2(out))
        return out


class Encoder(torch.nn.Module):
    def __init__(self, shapes, device):
        """
        :param shapes: list of filter sizes in blocks
        :param device:
        """
        super(Encoder, self).__init__()
        self.max_pooling = torch.nn.MaxPool2d((2, 2))
        # self.block_sizes = [3, 64, 128]
        self.block_sizes = shapes
        self.blocks = torch.nn.ModuleList(
            [Block(self.block_sizes[i], self.block_sizes[i + 1], device)
             for i, _ in enumerate(self.block_sizes[:-1])]
        )

    def forward(self, x):
        copy_crops = []  # for skip connection
        out = x
        for block in self.blocks[:-1]:
            out = block(out)
            copy_crops.append(out)
            out = self.max_pooling(out)

        out = self.blocks[-1](out)
        copy_crops = copy_crops[::-1]  # this will be putted in decoder in reversed order
        return out, copy_crops


class Decoder(torch.nn.Module):
    def __init__(self, shapes, device):
        """
        :param shapes: list of filter sizes in blocks
        :param device:
        """
        super(Decoder, self).__init__()
        # self.block_sizes = [128, 64]
        self.block_sizes = shapes
        self.blocks = torch.nn.ModuleList(
            [Block(self.block_sizes[i], self.block_sizes[i + 1], device)
             for i, _ in enumerate(self.block_sizes[:-1])]
        )
        self.transpose_convs = torch.nn.ModuleList(
            [torch.nn.ConvTranspose2d
             (self.block_sizes[i], self.block_sizes[i + 1], kernel_size=(2, 2), stride=(2, 2),
              padding=(0, 0), device=device)
             for i, _ in enumerate(self.block_sizes[:-1])]
        )

    def forward(self, x, copy_crops):
        # copy_crops for skip connection
        out = x
        for block, transpose_conv, skip in zip(self.blocks, self.transpose_convs, copy_crops):
            out = transpose_conv(out)
            out_width, out_height = out.shape[2:]
            skip_width, skip_height = skip.shape[2:]
            croped_height = (
                (skip_height - out_height) // 2, out_height + (skip_height - out_height) // 2)
            croped_width = (
                (skip_width - out_width) // 2, out_width + (skip_width - out_width) // 2)
            skip = skip[:, :, croped_height[0]:croped_height[1], croped_width[0]:croped_width[1]]
            out = torch.concat((skip, out), dim=1)
            out = block(out)
        return out


class UNet(torch.nn.Module):
    def __init__(self, shapes, device):
        """
        :param shapes: shapes of filter in each block of encoder and decoder
        :param device:
        """
        super(UNet, self).__init__()
        self.encoder = Encoder(shapes, device)
        self.decoder = Decoder(shapes[:0:-1], device)
        self.conv1 = torch.nn.Conv2d(64, 2, kernel_size=(1, 1), device=device)
        self.SoftMax = torch.nn.Softmax(dim=1)  # along chanel dimension

    def forward(self, x):
        out, copy_crops = self.encoder(x)
        out = self.decoder(out, copy_crops)
        out = self.conv1(out)
        out = self.SoftMax(out)
        return out


def check_of_model_work(coco, path_to_images, resize_shape, model, device, num_of_pictures,
                        plot_mask=False):
    """
    plots results of prediction

    :param coco: coco_module, which return function COCO from pycocotools

    :param path_to_images: directory with images

    :param resize_shape: all images will be resized to this shape

    :param device: device, where will be tensors stored.
    torch.device("cpu") or torch.device("cuda")

    :param num_of_pictures: how many pictures will be plotted

    :param model: trained neural network, which will return mask, this mask apply to input

    :param device: device, where will be tensors stored.
    torch.device("cpu") or torch.device("cuda")

    :param num_of_pictures: how many pictures will be plotted

    :param plot_mask: this flag define will be plot a multiplication of mask and input image
    or just mask, which return neural network

    :return:
    """
    to_pil = torchvision.transforms.ToPILImage()
    # here can change to test dataloader, to see how model performs on test data
    dataloader, _, _, _ = get_dataloader(coco, path_to_images, resize_shape, 1, device,
                                         need_transformations=False)
    dataloader_plot, _, _, _ = get_dataloader(coco, path_to_images, resize_shape, 1, device,
                                              need_normalize=False, need_transformations=False)
    plt.figure(figsize=(10, 10))
    n_rows = np.int(np.sqrt(num_of_pictures * 2))
    n_cols = n_rows
    if n_cols % 2 != 0:
        n_cols += 1
    if n_rows * n_cols < num_of_pictures * 2:
        n_rows += 1
    for i, ((x, y), (x_plot, _)) in enumerate(zip(dataloader, dataloader_plot)):
        if i == num_of_pictures:
            break
        predict = model(x)[0]
        if plot_mask:
            plt.subplot(n_rows, n_cols, i * 2 + 1)
            plt.axis("off")
            plt.imshow(to_pil(torch.argmax(y[0], dim=0).to(dtype=torch.float32).cpu()))
            plt.subplot(n_rows, n_cols, i * 2 + 2)
            plt.axis("off")
            plt.imshow(to_pil(torch.argmax(predict, dim=0).to(dtype=torch.float32).cpu()))
        else:
            x_plot = x_plot[0]
            processed_x = torch.mul(x_plot, torch.argmax(predict, dim=0))
            processed_x = processed_x.to(dtype=torch.int32).permute([1, 2, 0]).cpu()
            x_plot = x_plot.to(dtype=torch.int32).permute([1, 2, 0]).cpu()
            plt.subplot(n_rows, n_cols, i * 2 + 1)
            plt.axis("off")
            plt.imshow(x_plot)
            plt.subplot(n_rows, n_cols, i * 2 + 2)
            plt.axis("off")
            plt.imshow(processed_x)

    plt.show()


def main():
    coco_module = COCO("data/annotations/instances_train2017.json")
    unet_ = UNet([3, 64, 128, 256, 512], torch.device("cpu"))
    unet_.load_state_dict(torch.load(
        "trained_models/unet_ep_80_128BS_0.001lr_512params_128x128.pth")
    )
    check_of_model_work(coco_module, "data", (128, 128), unet_,
                        torch.device("cpu"), 18, False)


if __name__ == "__main__":
    main()
