import torch
import os
import pickle


class CassavaResnet50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
        self.resnet50.fc = torch.nn.Linear(in_features=self.resnet50.fc.in_features, out_features=5)

    def forward(self, image):
        return self.resnet50(image)


def evaluate_model(model, validation_loader, device) -> float:
    """Evaluates a model's accuracy.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        validation_loader (torch.utils.data.Subset): The data with which to evaluate the model.
        device (torch.device): The device to which to move the data & model.

    Returns:
        float: The model's accuracy on the provided data expressed as a percentage.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(validation_loader):
            preds = model(batch.images.to(device))
            batch_correct = sum(torch.argmax(preds, 1) == batch.labels.to(device)).item()

            total += len(preds)
            correct += batch_correct

    model.train()
    return 100. * correct / total


def train_model(device, train_dataloader, validation_dataloader, train_eval_subset_dataloader,
                model_output_directory='saved_models', epochs=1, max_samples=None, warm_start_path=None):
    model = CassavaResnet50()
    if warm_start_path:
        model.load_state_dict(torch.load(warm_start_path))
        adjust_lr_after = 0
        with open(os.path.join(model_output_directory, f'train-validation-scores.pckl', 'rb')) as outfile:
            saved_data = pickle.load(outfile)
            train_scores = saved_data['train_scores']
            validation_scores = saved_data['validation_scores']
            images_processed = saved_data['images_processed']
    else:
        adjust_lr_after = 1000
        validation_scores = []
        train_scores = []
        images_processed = []

    model.to(device)
    model.train()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=10)

    train_losses = []
    lr = .005
    completed = 0
    for epoch in range(epochs):
        for batch_index, image_batch in enumerate(train_dataloader):
            if max_samples and completed >= max_samples:
                print(f'Max number of samples ({max_samples}) reached. Stopping training.')
                break
            batch_size = len(image_batch.labels)
            images = image_batch.images.to(device)
            labels = image_batch.labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            current_loss = loss_fn(preds, labels)
            train_losses.append(current_loss.item())
            current_loss.backward()
            optimizer.step()
            if completed >= adjust_lr_after:
                scheduler.step(current_loss.item())

            torch.save(model.state_dict(), os.path.join(model_output_directory, f'resnet50-batch{batch_index}.pt'))

            completed += batch_size
            train_scores.append(evaluate_model(model, train_eval_subset_dataloader, device))
            validation_scores.append(evaluate_model(model, validation_dataloader, device))
            images_processed.append(completed)

            with open(os.path.join(model_output_directory, f'train-validation-scores.pckl', 'wb')) as outfile:
                pickle.dump({'train_scores': train_scores,
                             'validation_scores': validation_scores,
                             'images_processed': images_processed},
                            outfile)

            for param_group in optimizer.param_groups:
                if 'lr' in param_group:
                    lr = param_group["lr"]
                    break

            print(f'{completed} total images processed, ' +
                  f'train batch loss: {train_losses[-1]:.4f}, ' +
                  f'training accuracy: {train_scores[-1]}, ' +
                  f'validation accuracy: {validation_scores[-1]}, ' +
                  f'learning rate: {lr}')

    print('Finished training.')
    torch.save(model.state_dict(), os.path.join(model_output_directory, 'resnet50-complete.pt'))
    return model
