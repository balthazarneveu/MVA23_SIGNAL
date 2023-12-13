import torch
import numpy as np
from typing import Optional
from sklearn.metrics import confusion_matrix


def infer(
        model, dataloader, device,
        criterion=torch.nn.CrossEntropyLoss(),
        has_confusion_matrix: Optional[bool]=False):
    """Evaluate accuracy on the validation set"""
    model.eval()
    valid_loss = []
    correct_detection = []
    classes_prediction = []
    true_classes = []

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
            correct_detection.extend(correct_predictions.numpy())

            # Collect true and predicted labels
            true_classes.extend(labels[:, 0].cpu().numpy())
            classes_prediction.extend(class_prediction.cpu().numpy())

            valid_loss_batched = criterion(prediction, labels[:, 0])
        valid_loss.append(valid_loss_batched.cpu())
    accuracy = np.array(correct_detection).mean()
    valid_loss = np.array(valid_loss).mean()
    if has_confusion_matrix:
        conf_matrix = confusion_matrix(true_classes, classes_prediction)
    else:
        conf_matrix = None

    return accuracy, valid_loss, conf_matrix