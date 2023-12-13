import torch
# try:
#     from IPython import get_ipython
#     from tqdm.notebook import tqdm
# except Exception as exc:
from tqdm import tqdm
from data_loader import TRAIN, VALID, BATCH_SIZE, get_dataloaders, CONFIG_DATALOADER
from infer import infer
import numpy as np
from typing import Tuple, Optional, Callable
from dump import Dump
from pathlib import Path
import logging
from properties import ROOT_DIR
from copy import deepcopy
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(model: torch.nn.Module,
          config_dataloader=deepcopy(CONFIG_DATALOADER),
          augment_config: Optional[dict] = {},
          device: str = "cuda",
          lr: float = 1E-4, n_epochs=5,
          lr_scheduler: Optional[Callable] = None,
          needed_loss_scheduler: bool = False,
          batch_sizes: Optional[Tuple[int, int]] = None,
          out_dir: Path = None,
          **kwargs
          ):
    if out_dir is not None:
        out_dir.mkdir(exist_ok=True, parents=True)
    model = model.to(device)
    config = config_dataloader
    config[TRAIN] = {**config[TRAIN], **augment_config}
    if batch_sizes is not None:
        config[TRAIN][BATCH_SIZE], config[VALID][BATCH_SIZE] = batch_sizes
    dataloaders = get_dataloaders(config_data_paths=config)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if lr_scheduler is not None:
        scheduler = lr_scheduler(optimizer)
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
        if lr_scheduler is not None and needed_loss_scheduler:
            scheduler.step(training_loss)
        elif lr_scheduler is not None:
            scheduler.step()

        accuracy, valid_loss = infer(
            model, dataloaders[VALID],
            device, criterion=torch.nn.CrossEntropyLoss())
        valid_accuracies.append(accuracy)
        valid_losses.append(valid_loss)
        print(
            f"{epoch=} | lr={float(optimizer.param_groups[0]['lr']):.2e} | {training_loss=:.3f} | {valid_loss=:.3} | {accuracy:.2%}")
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


def classical_training_loop(exp_list, n_epochs=None, lr_list=[], device=DEVICE):
    for exp in exp_list:
        from model import get_experience
        if lr_list is None or len(lr_list) == 0:
            lr_list = [None]
        for lr in lr_list:
            model, hyperparams, augment_config = get_experience(exp)
            model = model.to(device)
            if n_epochs is not None:
                hyperparams["n_epochs"] = n_epochs
            if lr is not None:
                logging.warning(f"Forcing lr to {lr}")
                hyperparams["lr"] = lr
            suffix = f"_lr_{lr:.1E}" if lr is not None else ""
            train_folder = ROOT_DIR/f"exp_{exp:04d}{suffix}"
            save_dict = {
                **hyperparams,
                **augment_config
            }
            save_dict.pop("lr_scheduler", None)

            if train_folder.exists():
                logging.warning(
                    f"Skipping {train_folder} as it already exists")
                Dump.save_yaml(save_dict, train_folder/"config.yaml")
                Dump.save_json(save_dict, train_folder/"config.json")
            else:
                Dump.save_yaml(save_dict, train_folder/"config.yaml")
                Dump.save_json(save_dict, train_folder/"config.json")
                model, metrics_dict = train(
                    model,
                    out_dir=train_folder,
                    augment_config=augment_config,
                    device=device,
                    **hyperparams,
                )


if __name__ == "__main__":
    import argparse
    # from argparse import ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument("-e",  "--exp", type=int, nargs="+", default=[0])
    parser.add_argument("-lr",  "--lr", type=float, nargs="+", default=[])
    parser.add_argument("-n",  "--n-epochs", type=int, required=False)
    parser.add_argument("-d",  "--device", type=str, default=DEVICE)
    args = parser.parse_args()
    classical_training_loop(args.exp, n_epochs=args.n_epochs,
                            lr_list=args.lr, device=args.device)
