from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import IterableDataset, DataLoader

import numpy as np
from pycocotools.coco import COCO


def get_person_annotations(coco):
    """
    :param coco: coco_module, which return function COCO from pycocotools
    :return:
    """
    person_id = coco.getCatIds(catNms=["person"])[0]
    ids_images_with_persons = coco.getImgIds(catIds=[person_id])

    annotations_ids = []
    for id_img in ids_images_with_persons:
        anns = coco.getAnnIds(imgIds=[id_img])
        annotations_ids.append(anns)
    return annotations_ids, person_id


def get_filtered_data(coco):
    """
    :param coco: coco_module, which return function COCO from pycocotools
    :return: annotations to images, which pass several condition of filtering
    """
    annotations_ids, person_id = get_person_annotations(coco)
    filtered_ann = []
    for num_iter, ann_id in enumerate(annotations_ids):
        skip_image = False
        count_of_persons_on_image = 0
        annotations_of_img = coco.loadAnns(ann_id)
        if len(annotations_of_img) > 15:
            # too many objects
            continue
        img = coco.loadImgs(annotations_of_img[0]["image_id"])[0]
        img_area = int(img["height"]) * int(img["width"])
        area_of_person = -1
        max_area = -1
        index_person_in_ann = 0
        for i, ann in enumerate(annotations_of_img):
            if ann["category_id"] == person_id:
                count_of_persons_on_image += 1
                area_of_person = ann["area"]
                index_person_in_ann = i
            if max_area < ann["area"]:
                max_area = ann["area"]
            if count_of_persons_on_image > 1:
                skip_image = True
                break

        if skip_image:
            continue
        if area_of_person < img_area * 0.10:
            continue
        if area_of_person != max_area:
            continue
        filtered_ann.append(annotations_of_img[index_person_in_ann])
    print(f"{len(filtered_ann)} filtered images with person on it")
    return filtered_ann


class IterableDatasetCOCO(IterableDataset):
    def __init__(self, coco, path_to_images, annotations, resize_shape, device,
                 need_normalize=True, need_transformations=True):
        """
        :param coco: coco_module, which return function COCO from pycocotools

        :param path_to_images: directory with images

        :param annotations: list of annotations, gets function from get_filtered_data()

        :param resize_shape: all images will be resized to this shape

        :param device: device, where will be tensors stored.
        torch.device("cpu") or torch.device("cuda")

        :param need_normalize: flag need to be True, when dataset is used for train network.
        otherwise can be False, for example in case of plotting dataset

        :param need_transformations: flag need to be True,
        when to image need perform augmentation methods
        """
        super().__init__()
        self.coco = coco
        self.path_to_images = path_to_images
        self.annotation_of_images = annotations
        self.resize_shape = resize_shape
        self.device = device
        self.need_normalize = need_normalize
        self.need_transformations = need_transformations

        self.names_of_images = []

        # calculated empirical by training dataset
        # mean = torch.mean(tensor_image, dim=[1, 2])
        # std = torch.std(tensor_image, dim=[1, 2])
        mean = [111.8526, 103.3080, 95.0226]
        std = [75.0907, 72.1730, 73.3086]

        self.norm = torchvision.transforms.Normalize(mean, std)
        self.resize = torchvision.transforms.Resize(self.resize_shape)

        for ann in self.annotation_of_images:
            # recording names of files, which will go to the network
            img_id = ann["image_id"]
            img = coco.loadImgs([img_id])
            self.names_of_images.append(img[0]["file_name"])

    def get_next_data(self):
        """
        :return: tuple of two matrises.
        First is image in shape (chanel, resize_shape[0], resize_shape[1]).
        Second is mask in shape (2, resize_shape[0], resize_shape[1]).
        mask[0] is mask of background
        mask[1] is mask of person
        """
        for i, name in enumerate(self.names_of_images):
            img = Image.open(f"{self.path_to_images}/{name}")
            tensor_image = torch.tensor(np.array(img), device=self.device, dtype=torch.int32)
            if len(tensor_image.shape) != 3:
                # skip black and white  image
                continue
            tensor_image = tensor_image.permute((2, 0, 1))
            tensor_image = self.resize(tensor_image)

            image_mask = self.coco.annToMask(self.annotation_of_images[i])
            tensor_mask_person = torch.tensor(image_mask, device=self.device, dtype=torch.float32)
            tensor_mask_person = torch.unsqueeze(tensor_mask_person, dim=0)
            tensor_mask_person = self.resize(tensor_mask_person)
            tensor_mask_background = torch.abs(tensor_mask_person - 1)
            tensor_mask = torch.concat([tensor_mask_background, tensor_mask_person], dim=0)

            if self.need_transformations:
                tensor_image, tensor_mask = self.transformation(tensor_image, tensor_mask)

            tensor_image = tensor_image.to(dtype=torch.float32)

            if self.need_normalize:
                tensor_image = self.norm(tensor_image)

            yield tensor_image, tensor_mask

    def transformation(self, tensor_image, tensor_mask):
        """
        performs augmentation of image
        :param tensor_image:
        :param tensor_mask:
        :return:
        """
        p = np.random.random()
        if p < 0.5:
            tensor_image = F.hflip(tensor_image)
            tensor_mask = F.hflip(tensor_mask)
        p = np.random.random()
        if p < 0.5:
            angle = np.random.randint(-40, 40)
            tensor_image = F.rotate(tensor_image, angle)
            tensor_mask = F.rotate(tensor_mask, angle)
        p = np.random.random()
        if p < 0.5:
            factor = np.random.uniform(0.5, 1.5)
            tensor_image = F.adjust_contrast(tensor_image, factor)
        p = np.random.random()
        if p < 0.5:
            factor = np.random.uniform(0.5, 1.5)
            tensor_image = F.adjust_contrast(tensor_image, factor)
        return tensor_image, tensor_mask

    def __len__(self):
        return len(self.annotation_of_images)

    def __iter__(self):
        return iter(self.get_next_data())


def get_dataloader(coco, path_to_images, images_shape,
                   batch_size, device, limiter=0,
                   need_transformations=True, need_normalize=True):
    """
    :param coco: coco_module, which return function COCO from pycocotools

    :param path_to_images: directory with images

    :param images_shape: all images will be resized to this shape

    :param batch_size:

    :param device: device, where will be tensors stored.
    torch.device("cpu") or torch.device("cuda")

    :param limiter: returns dataloader with this count of images.
    Need for debugging neural networks. Zero means dataloader will have all possible images

    :param need_normalize: flag need to be True, when dataset is used for train network.
    otherwise can be False, for example in case of plotting dataset

    :param need_transformations: flag need to be True,
    when to image need perform augmentation methods

    :return: dataloader
    """
    filtered_annotation = get_filtered_data(coco)
    if limiter > 0:
        filtered_annotation = filtered_annotation[:limiter]
    length_train_data = int(len(filtered_annotation) * 0.8)
    length_test_data = len(filtered_annotation) - length_train_data
    dataset_train = IterableDatasetCOCO(coco, path_to_images,
                                        filtered_annotation[:length_train_data],
                                        images_shape, device,
                                        need_transformations=need_transformations,
                                        need_normalize=need_normalize)
    dataset_test = IterableDatasetCOCO(coco, path_to_images,
                                       filtered_annotation[length_train_data:],
                                       images_shape, device,
                                       need_transformations=need_transformations,
                                       need_normalize=need_normalize)
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, num_workers=0)
    test_dataloader = DataLoader(dataset_test, batch_size=1, num_workers=0)
    return train_dataloader, test_dataloader, length_train_data, length_test_data


def check_of_loader_work(coco, path_to_images, images_shape, device, num_of_pictures):
    """
    plotting images of dataset

    :param coco: coco_module, which return function COCO from pycocotools

    :param path_to_images: directory with images

    :param images_shape: all images will be resized to this shape

    :param device: device, where will be tensors stored.
    torch.device("cpu") or torch.device("cuda")

    :param num_of_pictures: how many pictures will be plotted

    :return:
    """
    dataloader, _, _, _ = get_dataloader(coco, path_to_images, images_shape, 1, device,
                                         need_normalize=False)

    plt.figure(figsize=(10, 10))
    n_rows = np.int(np.sqrt(num_of_pictures * 2))
    n_cols = n_rows
    if n_rows * n_cols < num_of_pictures * 2:
        n_rows += 1
        n_cols += 1
    for i, (x, y) in enumerate(dataloader):
        if i == num_of_pictures:
            break
        plt.subplot(n_rows, n_cols, i * 2 + 1)
        plt.axis("off")
        plt.imshow(x[0].to(dtype=torch.int32).permute([1, 2, 0]))
        plt.subplot(n_rows, n_cols, i * 2 + 2)
        plt.axis("off")
        plt.imshow(y[0][1])
    plt.show()
    print()


def main():
    path_to_images = "data"
    coco_module = COCO(f"{path_to_images}/annotations/instances_train2017.json")
    check_of_loader_work(coco_module, path_to_images, (256, 256), torch.device("cpu"), 8)


if __name__ == "__main__":
    main()
