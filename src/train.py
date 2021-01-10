import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import os.path
import cassavadataloader
import cassava_resnet

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

    full_dataset_train = cassavadataloader.CassavaDataset(images_path=os.path.join('..', 'train_images'),
                                                          validation=False,
                                                          labels_manifest_path=os.path.join('..', 'train.csv'))
    full_dataset_eval = cassavadataloader.CassavaDataset(images_path=os.path.join('..', 'train_images'),
                                                          validation=True,
                                                          labels_manifest_path=os.path.join('..', 'train.csv'))

    train_indices, validation_indices = train_test_split(list(range(len(full_dataset_train))), test_size=.3)

    train_dataset = Subset(full_dataset_train, train_indices)
    validation_dataset = Subset(full_dataset_eval, validation_indices)
    train_eval_dataset = Subset(full_dataset_eval, train_indices[:1000])

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=cassavadataloader.custom_collate_wrapper,
                                  pin_memory=True,
                                  shuffle=True)
    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=BATCH_SIZE,
                                       collate_fn=cassavadataloader.custom_collate_wrapper,
                                       pin_memory=True,
                                       shuffle=False)
    train_eval_dataloader = DataLoader(train_eval_dataset,
                                       batch_size=BATCH_SIZE,
                                       collate_fn=cassavadataloader.custom_collate_wrapper,
                                       pin_memory=True,
                                       shuffle=True)

    model = cassava_resnet.train_model(device, train_dataloader, validation_dataloader, train_eval_dataloader,
                                       model_output_directory=os.path.join('..', 'saved_models'),
                                       epochs=1, max_samples=1000, warm_start_path=None)


if __name__ == '__main__':
    run(require_gpu=False)
