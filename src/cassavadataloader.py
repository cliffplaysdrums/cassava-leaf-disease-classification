from torch.utils.data import Dataset, Subset
import torchvision.transforms
import torch
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# We need a custom class for batching to provide the dataloader info on how to batch our data & pin memory
class CustomBatch:
    def __init__(self, data):
        self.images = torch.stack([datum[0] for datum in data])
        self.labels = torch.stack([torch.tensor(datum[1]) for datum in data])

    def pin_memory(self):
        self.images = self.images.pin_memory()
        self.labels = self.labels.pin_memory()
        return self


def custom_collate_wrapper(batch):
    return CustomBatch(batch)


class CassavaDataset(Dataset):
    def __init__(self, images_path='train_images', validation=False, labels_manifest_path='..train.csv'):
        transforms = {
            'train': torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomResizedCrop(512),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
            'validate': torchvision.transforms.Compose(
                [
                    torchvision.transforms.CenterCrop(512),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        }

        self.preprocess = transforms['validate' if validation else 'train']
        self.labels_df = pd.read_csv(labels_manifest_path)
        self.images_path = images_path

    def __len__(self):
        return self.labels_df.shape[0]

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_path, self.labels_df['image_id'].iloc[idx])
        img = self.preprocess(Image.open(img_name))
        label = self.labels_df['label'][idx]

        return img, label


def get_train_validate_split(dataset, val_portion=.2):
    train_indices, validation_indices = train_test_split(list(range(len(dataset))), test_size=val_portion)

    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)

    return train_dataset, validation_dataset



