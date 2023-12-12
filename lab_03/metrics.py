import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from training import ROOT_DIR
from dump import Dump
from typing import List, Optional
from pathlib import Path
from data_loader import get_dataloaders, TRAIN
num_samples = len(get_dataloaders()[TRAIN].dataset)


def plot_results(metrics_dict_comparison):
    colors = ["r"]
    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
    for exp_name, metrics_dict in metrics_dict_comparison.items():
        extra = ""
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
        axs[0].plot(epoch_steps, valid_losses,  "-o",
                    label=f"{exp_name}")
        max_acc = np.max(np.array(valid_accuracies))
        axs[1].plot(epoch_steps, 100*np.array(valid_accuracies),
                    "-", label=f"{exp_name} {max_acc:.1%} {extra}")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].set_title("Losses")
    axs[0].grid()

    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Accuracy %")
    axs[1].legend()
    axs[1].set_title("Validation accuracy")
    axs[1].grid()
    axs[1].set_ylim(0, 100)
    plt.suptitle("Comparison of training results")
    plt.show()


def results_comparisons(all_experiments_path: List[Path], selection: List[str] = []):
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
        checkpoint_file = exp_dir/"metrics.pkl"
        if checkpoint_file.exists():
            metrics_dict_comparison[exp_name] = Dump.load_pickle(
                checkpoint_file)
        metrics_dict_comparison[exp_name]["config"] = config
    plot_results(metrics_dict_comparison)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r",  "--root-dir", type=str, default=str(ROOT_DIR))
    parser.add_argument("-e",  "--selection", type=str, nargs="+", default=[])
    args = parser.parse_args()

    exp_dirs = Path(args.root_dir).glob("*")
    all_experiments_path = sorted(list(exp_dirs))
    # selection = [f"{i:02d}" for i in [8, 9]]
    results_comparisons(all_experiments_path, args.selection)
