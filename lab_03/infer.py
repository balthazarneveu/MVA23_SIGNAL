import torch
import numpy as np


def infer(model, dataloader, device, criterion=torch.nn.CrossEntropyLoss()):
    """Evaluate accuracy on the validation set"""
    model.eval()
    valid_loss = []
    correct_detection = []
    for signal, labels in dataloader:
        with torch.no_grad():
            signal = signal.to(device)
            labels = labels.to(device)
            prediction = model(signal)
            proba = torch.nn.Softmax()(prediction)
            class_prediction = torch.argmax(proba, axis=1)
            # print(class_prediction.shape, labels.shape)
            correct_predictions = (
                class_prediction == labels[:, 0]).detach().cpu()
            correct_detection.extend(correct_predictions)
            # print(correct_detection)
            valid_loss_batched = criterion(prediction, labels[:, 0])
        valid_loss.append(valid_loss_batched.cpu())
    accuracy = np.array(correct_detection).mean()
    valid_loss = np.array(valid_loss).mean()
    return accuracy, valid_loss
