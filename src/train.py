import torch
from torch.utils.data import DataLoader
import os.path
import cassavadataloader

BATCH_SIZE = 2


def run(require_gpu=True):
    if not torch.cuda.is_available():
        print('No CUDA-enabled GPU detected.')
        if require_gpu:
            print('GPU required flag is set. Exiting.')
            exit()
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')

    print(f'Using CUDA-enabled device: {torch.cuda.get_device_name(device)}')

    full_dataset = cassavadataloader.CassavaDataset(images_path=os.path.join('..', 'train_images'), validation=False,
                                                    labels_manifest_path=os.path.join('..', 'train.csv'))
    train_subset, validation_subset = cassavadataloader.get_train_validate_split(full_dataset, val_portion=.3)

    train_dataloader = DataLoader(train_subset,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=cassavadataloader.custom_collate_wrapper,
                                  pin_memory=True,
                                  shuffle=True)
    validation_dataloader = DataLoader(validation_subset,
                                       batch_size=BATCH_SIZE,
                                       collate_fn=cassavadataloader.custom_collate_wrapper,
                                       pin_memory=True,
                                       shuffle=False)

    for batch_index, image_batch in enumerate(train_dataloader):
        if batch_index == 5:
            break

        print(f'Batch {batch_index} size: {len(image_batch.images[0])}. is_pinned: {image_batch.images[0].is_pinned()}')


if __name__ == '__main__':
    run(require_gpu=False)
