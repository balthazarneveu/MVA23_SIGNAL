import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from training import ROOT_DIR
from dump import Dump


def plot_results(metrics_dict_comparison):
    colors = ["r"]
    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
    for exp_name, metrics_dict in metrics_dict_comparison.items():
        training_losses = metrics_dict["training_losses"]
        valid_losses = metrics_dict["valid_losses"]
        valid_accuracies = metrics_dict["valid_accuracies"]

        epoch_length = len(training_losses)/len(valid_losses)
        epoch_steps = np.linspace(epoch_length, len(
            training_losses), len(valid_losses))
        # axs[0].plot(training_losses, ".", alpha=0.01, label=f"training loss {exp_name}")
        axs[0].plot(epoch_steps, valid_losses,  "-o", label=f"validation loss {exp_name}")
        axs[1].plot(epoch_steps, 100*np.array(valid_accuracies),
                    "-", label=f"validation accuracy {exp_name}")
    axs[0].set_xlabel("Step")
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


if __name__ == "__main__":
    exp_dirs = ROOT_DIR.glob("*")
    metrics_dict_comparison = {}
    for exp_dir in sorted(list(exp_dirs)):
        exp_name = exp_dir.name
        checkpoint_file = exp_dir/"metrics.pkl"
        if checkpoint_file.exists():
            metrics_dict_comparison[exp_name] = Dump.load_pickle(checkpoint_file)
    plot_results(metrics_dict_comparison)