import torch
# try:
#     from IPython import get_ipython
#     from tqdm.notebook import tqdm
# except Exception as exc:
from tqdm import tqdm
from data_loader import TRAIN, VALID, BATCH_SIZE, get_dataloaders, CONFIG_DATALOADER
import numpy as np
from typing import Tuple, Optional
from dump import Dump
from pathlib import Path

ROOT_DIR = Path(__file__).parent/"__dump"


def train(model: torch.nn.Module,
          config_dataloader=CONFIG_DATALOADER,
          augment_config: Optional[dict] = {},
          device: str = "cuda",
          lr: float = 1E-4, n_epochs=5,
          batch_sizes: Optional[Tuple[int, int]] = None,
          out_dir: Path = None
          ):
    if out_dir is not None:
        out_dir.mkdir(exist_ok=True, parents=True)
    model = model.to(device)
    config = config_dataloader
    config[TRAIN] = {**config[TRAIN], **augment_config}
    if batch_sizes is not None:
        config[TRAIN][BATCH_SIZE], config[VALID][BATCH_SIZE] = batch_sizes
    dataloaders = get_dataloaders()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    training_losses = []
    valid_losses = []
    valid_accuracies = []
    valid_loss_previous = 10000000000.
    # Loop over epochs
    for epoch in range(n_epochs):
        training_losses_epoch = []
        model.train()
        # Loop of steps
        training_dl = dataloaders[TRAIN]
        total_steps = len(training_dl)
        for step, (signal, labels) in enumerate(tqdm(
            training_dl,
            desc=f"Epoch {epoch}",
            total=total_steps
        )):
            optimizer.zero_grad()
            signal = signal.to(device)
            labels = labels.to(device)
            # print(model)
            prediction = model(signal)
            loss = criterion(prediction, labels[:, 0])
            loss.backward()
            optimizer.step()
            training_losses_epoch.append(loss.detach().cpu())
        training_losses.extend(training_losses_epoch)
        training_loss = np.array(training_losses_epoch).mean()
        # Evaluate accuracy on the validation set
        model.eval()
        valid_loss = []
        correct_detection = []
        for signal, labels in dataloaders[VALID]:
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
        valid_accuracies.append(accuracy)
        valid_loss = np.array(valid_loss).mean()
        valid_losses.append(valid_loss)
        print(f"{epoch=} | {training_loss=:.3f} | {valid_loss=:.3} | {accuracy:.2%}")
        if valid_loss <= valid_loss_previous:
            torch.save(model, out_dir/"best_model.pth")
            valid_loss_previous = valid_loss
    metrics_dict = dict(
        training_losses=training_losses,
        valid_losses=valid_losses,
        valid_accuracies=valid_accuracies
    )

    Dump.save_pickle(metrics_dict, out_dir/"metrics.pkl")
    return model, metrics_dict


if __name__ == "__main__":
    import argparse
    # from argparse import ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument("-e",  "--exp", type=int, nargs="+", default=[0])
    parser.add_argument("-n",  "--n-epochs", type=int, required=False)
    args = parser.parse_args()
    for exp in args.exp:
        from model import get_experience
        model, hyperparams, augment_config = get_experience(exp)
        if args.n_epochs is not None:
            hyperparams["n_epochs"] = args.n_epochs
        model, metrics_dict = train(
            model,
            out_dir=ROOT_DIR/f"exp_{exp:04d}",
            augment_config=augment_config,
            **hyperparams,

        )
