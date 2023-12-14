import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from properties import ROOT_DIR
from dump import Dump
from typing import List, Optional
from pathlib import Path
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from data_loader import get_dataloaders, TRAIN, VALID, CONFIG_DATALOADER, SNR_FILTER
from infer import infer
num_samples = len(get_dataloaders()[TRAIN].dataset)
device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_results(metrics_dict_comparison):
    colors = ["r"]
    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
    for exp_name, metrics_dict in metrics_dict_comparison.items():
        extra = ""
        prefix = ""
        params_count = metrics_dict["config"].get("param_count", None)
        if params_count is not None:
                prefix += f" {float(params_count)/1E3:.0f}k params"
        batch_sizes = metrics_dict["config"].get("batch_sizes", None)
        if batch_sizes is not None:
            normalization = num_samples/int(batch_sizes[0])
            extra += f" N={int(batch_sizes[0])}"
        else:
            normalization = 1
        lr = metrics_dict["config"].get("lr", None)
        if lr is not None:
            lr = float(lr)
            extra += f" lr={lr:.1E}"
        lr_scheduler_name = metrics_dict["config"].get(
            "lr_scheduler_name", None)
        if lr_scheduler_name is not None:
            extra += f" {lr_scheduler_name}"
        for aug_type in ["rotate", "trim", "noise"]:
            augment_val = metrics_dict["config"].get(f"augment_{aug_type}", {})
            if augment_val is not None:
                if augment_val not in ["0", "false"]:  # Fix yaml!
                    extra += f" {aug_type}"
        training_losses = metrics_dict["training_losses"]
        valid_losses = metrics_dict["valid_losses"]
        valid_accuracies = metrics_dict["valid_accuracies"]

        epoch_length = len(training_losses)/len(valid_losses)
        epoch_steps = np.linspace(epoch_length, len(
            training_losses), len(valid_losses))/normalization

        # axs[0].plot(training_losses, ".", alpha=0.01, label=f"training loss {exp_name}")
        label_name = exp_name
        annotation = metrics_dict["config"].get("annotation", None)
        if annotation is not None:
            label_name = annotation
        axs[0].plot(epoch_steps, valid_losses,  "-o",
                    label=f"{label_name}")
        max_acc = np.max(np.array(valid_accuracies))
        axs[1].plot(epoch_steps, 100*np.array(valid_accuracies),
                    "-", label=f"{prefix} | {label_name} {max_acc:.1%} {extra}")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].set_title("Losses")
    axs[0].grid()

    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy %")
    axs[1].legend()
    axs[1].set_title("Validation accuracy")
    axs[1].grid()
    axs[1].set_ylim(0, 100)
    plt.suptitle("Comparison of training results")
    plt.show()


def snr_based_metrics(metrics_dict_comparison: dict):
    colors = ["r"]
    # fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
    conf_matrices = {}
    for id_exp,  (exp_name, metrics_dict) in enumerate(metrics_dict_comparison.items()):
        extra = ""
        batch_sizes = metrics_dict["config"].get("batch_sizes", None)
        conf_matrices[id_exp] = []

        if batch_sizes is not None:
            normalization = num_samples/int(batch_sizes[0])
            extra += f"| batch size ={int(batch_sizes[0])}"
        else:
            normalization = 1
        lr = metrics_dict["config"].get("lr", None)
        if lr is not None:
            lr = float(lr)
            extra += f"| lr={lr:.1E}"
        lr_scheduler_name = metrics_dict["config"].get(
            "lr_scheduler_name", None)
        if lr_scheduler_name is not None:
            extra += f"| {lr_scheduler_name}"
        for aug_type in ["rotate", "trim", "noise"]:
            augment_val = metrics_dict["config"].get(f"augment_{aug_type}", {})
            if augment_val is not None:
                if augment_val not in ["0", "false"]:  # Fix yaml!
                    extra += f"| {aug_type}"
        n_epochs = int(metrics_dict["config"].get("n_epochs", 100))
        extra += f"| {n_epochs} epochs"
        label_name = exp_name
        annotation = metrics_dict["config"].get("annotation", None)
        if annotation is not None:
            label_name = annotation
        assert metrics_dict["config"]["model_path"].exists()
        model = torch.load(
            metrics_dict["config"]["model_path"], map_location=torch.device(device))
        from copy import deepcopy
        config_data_paths = deepcopy(CONFIG_DATALOADER)
        perf_regarding_snr = {}
        snrs = [0, 10, 20, 30]
        for snr in snrs:
            config_data_paths[VALID][SNR_FILTER] = [snr]
            dl = get_dataloaders(config_data_paths=config_data_paths)
            accuracy, valid_loss, conf_matrix = infer(
                model, dl[VALID], device=device, has_confusion_matrix=True)
            perf_regarding_snr[snr] = accuracy
            conf_matrices[id_exp].append(conf_matrix)
        # avg_acc = np.mean(list(perf_regarding_snr.values()))
        max_acc = np.max(np.array(metrics_dict["valid_accuracies"]))
        w = 0.5
        plt.bar([key+id_exp*w for key in perf_regarding_snr.keys()],
                perf_regarding_snr.values(), width=w, label=f"Acc: {max_acc:.1%} |" + label_name + extra)
        plt.xlabel("SNR")
        plt.ylabel("Accuracy")
    plt.title("Accuracy with regard to SNR")
    plt.legend()
    plt.grid()
    plt.show()
    for id_exp,  (exp_name, metrics_dict) in enumerate(metrics_dict_comparison.items()):
        num_matrices = len(conf_matrices[id_exp])
        n_rows = int(np.sqrt(num_matrices))
        n_cols = int(np.ceil(num_matrices / n_rows))

        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5, 5))
        # Flatten the axes if there's only one row or one column
        if num_matrices == 1:
            axes = axes.reshape(1, 1)
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, ax in enumerate(axes.flatten()):
            if i < num_matrices:
                # Plot confusion matrix
                cm = conf_matrices[id_exp][i]
                classes = {0: 'N-QAM16', 1: 'N-PSK8', 2: 'N-QPSK',
                           3: 'W-QAM16', 4: 'W-PSK8-V1', 5: 'W-PSK8-V2'}.values()
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)

                # Label axes
                ax.set(xticks=np.arange(len(classes)),
                       yticks=np.arange(len(classes)),
                       xticklabels=classes, yticklabels=classes,
                       title=f'Confusion Matrix snr {snrs[i]}',
                       ylabel='True label',
                       xlabel='Predicted label')
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")

            # Remove empty subplots
            else:
                ax.axis('off')

        # Adjust layout for better visualization
        plt.suptitle(f"Confusion matrix {exp_name}")
        plt.tight_layout()
        plt.show()


def results_comparisons(all_experiments_path: List[Path], selection: List[str] = [], evolution: bool = True):
    metrics_dict_comparison = {}
    for exp_dir in all_experiments_path:
        exp_name = exp_dir.name
        if len(selection) > 0:
            selected = False
            for sel in selection:
                if sel in exp_name:
                    selected = True
                    break
        else:
            selected = True
        if not selected:
            continue
        config_file = exp_dir/"config.yaml"
        if config_file.exists():
            config = Dump.load_yaml(config_file, safe_load=False)
        else:
            config = {}
        config["model_path"] = exp_dir/"best_model.pth"

        checkpoint_file = exp_dir/"metrics.pkl"
        if checkpoint_file.exists():
            metrics_dict_comparison[exp_name] = Dump.load_pickle(
                checkpoint_file)
        metrics_dict_comparison[exp_name]["config"] = config
    if evolution:
        plot_results(metrics_dict_comparison)
    else:
        snr_based_metrics(metrics_dict_comparison)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r",  "--root-dir", type=str, default=str(ROOT_DIR))
    parser.add_argument("-e",  "--selection", type=str, nargs="+", default=[])
    parser.add_argument("-m", "--metrics", action="store_true")
    args = parser.parse_args()

    exp_dirs = Path(args.root_dir).glob("*")
    all_experiments_path = sorted(list(exp_dirs))
    # selection = [f"{i:02d}" for i in [8, 9]]
    results_comparisons(all_experiments_path, args.selection,
                        evolution=not args.metrics)
