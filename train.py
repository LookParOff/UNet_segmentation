import numpy as np

import torch
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torchmetrics import JaccardIndex

from model import UNet
from data_prepare import get_dataloader


def fit(model, model_save_path, epochs, loss_func, opt,
        train_dl, test_dl, metric, len_test):
    history_loss = []
    history_metric = []

    for ep in range(epochs):
        for x, y in train_dl:
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"№{ep}, loss {loss.item()}", end="\r")
        history_loss.append(loss.item())
        with torch.no_grad():
            history_metric.append(0)
            for x, y in test_dl:
                y_pred = model(x)
                history_metric[-1] += metric(y_pred, y.to(dtype=torch.int32))
            history_metric[-1] = history_metric[-1].cpu() / len_test

        if ep % 10 == 0:
            torch.save(model.state_dict(), model_save_path)
    return model, history_loss, history_metric


def main():
    images_path = "data"
    coco_module = COCO(f"{images_path}/annotations/instances_train2017.json")
    device = torch.device("cuda")
    print(device)
    unet = UNet([3, 64, 128, 256, 512], device)
    model_save_path = f"trained_models/unet.pth"

    images_shape = (128, 128)
    epochs = 80
    loss_func = torch.nn.CrossEntropyLoss()
    batch_size = 128

    train_dataloader, test_dataloader, len_train, len_test = \
        get_dataloader(coco_module, images_path, images_shape, batch_size, device,
                       limiter=10)

    optm = torch.optim.Adam(unet.parameters(), lr=0.001)
    metric_jaccard = JaccardIndex(2, average="macro").to(device=device)

    unet, history_loss, history_jac = \
        fit(unet, model_save_path, epochs, loss_func, optm,
            train_dataloader, test_dataloader,
            metric_jaccard, len_test)

    plt.plot(np.arange(0, len(history_loss)), history_loss)
    plt.title("train loss")
    plt.show()

    plt.plot(np.arange(0, len(history_jac)), history_jac)
    plt.title("test metric")
    plt.show()

    print("total stat:")
    print(history_loss)
    print(history_jac)

    return unet


if __name__ == "__main__":
    main()
