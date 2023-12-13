from data_loader import AUGMENT_TRIM, AUGMENT_NOISE, AUGMENT_ROTATE, SNR_FILTER
from properties import ROOT_DIR
import torch
from typing import Tuple
from architecture import FlexiConv
from model import (
    VanillaClassifier, Convolutional_baseline, RnnBaseline, Slim_Convolutional
)


def get_experience(exp: int) -> Tuple[torch.nn.Module, dict, dict]:
    augment_config = {
        AUGMENT_TRIM: False,
        AUGMENT_NOISE: 0,
        AUGMENT_ROTATE: False
    }
    hyperparams = dict(
        lr=1E-4,
        n_epochs=100,
        batch_sizes=(256, 128),
        lr_scheduler=None,
        needed_loss_scheduler=False
    )
    if exp == 0:
        model = VanillaClassifier()  # 40%
        hyperparams["annotation"] = "Vanilla classifier"
    elif exp == 1:
        model = Convolutional_baseline(pool_temporal=8)  # 49%
    elif exp == 2:
        model = Convolutional_baseline(pool_temporal=4)  # 61%
        hyperparams = dict(
            lr=1E-4,
            n_epochs=300,
            batch_sizes=(256, 128)
        )
    elif exp == 3:
        model = RnnBaseline(bidirectional=False, h_dim=8)  # 20% h_dim=8
        # hyperparams["annotation"] = "RNN classifier"
    elif exp == 4:
        model = RnnBaseline(bidirectional=True, h_dim=16)
        hyperparams["annotation"] = "RNN classifier"
    elif exp == 5:
        model = Convolutional_baseline(rnn=True)  # 68.1%
        hyperparams["n_epochs"] = 500
    elif exp == 6:
        model = Slim_Convolutional(rnn=False)  # 73.3%
        hyperparams["n_epochs"] = 500
    elif exp == 7:
        model = Slim_Convolutional(rnn=False)  # 78.2%
        hyperparams["n_epochs"] = 1000
    # AUGMENTATION EXPERIMENTS
    elif exp == 8:
        model = Slim_Convolutional(rnn=False)  # ?
        hyperparams["n_epochs"] = 100
        augment_config[AUGMENT_ROTATE] = True
        hyperparams["batch_sizes"] = (512, 1024)
    elif exp == 9:
        model = Slim_Convolutional(rnn=False)  # ?
        hyperparams["n_epochs"] = 100
        augment_config[AUGMENT_NOISE] = 0.1
        hyperparams["batch_sizes"] = (512, 1024)
    elif exp == 10:
        model = Slim_Convolutional(rnn=False)  # ?
        hyperparams["n_epochs"] = 100
        augment_config[AUGMENT_TRIM] = True

    elif exp == 11:
        model = Slim_Convolutional(rnn=False)  # ?
        hyperparams["n_epochs"] = 100
        augment_config[AUGMENT_ROTATE] = True
        augment_config[AUGMENT_TRIM] = True
    elif exp == 12:
        model = Slim_Convolutional(rnn=False)  # ?
        hyperparams["n_epochs"] = 100
        augment_config[AUGMENT_ROTATE] = True
        augment_config[AUGMENT_TRIM] = True
        augment_config[AUGMENT_NOISE] = 0.1
    elif exp == 19:  # No augmentation
        model = Slim_Convolutional(rnn=False)  # ?
        hyperparams["n_epochs"] = 100
        hyperparams["batch_sizes"] = (512, 1024)
    # LR experiments
    elif exp == 20:
        model = Slim_Convolutional(rnn=False)  # 78%
        model = torch.load(ROOT_DIR/"exp_0007/best_model.pth",
                           map_location=torch.device('cpu'))
        # model.load_state_dict(a) # START from exp 6 TODO path and device hardcoded...
        hyperparams["n_epochs"] = 100
        hyperparams["lr"] = 1e-3  # default to 1E-4
    elif exp == 21:
        model = Slim_Convolutional(rnn=False)
        hyperparams["n_epochs"] = 10
        hyperparams["lr_scheduler_name"] = "ExponentialLR_0.9"
        hyperparams["lr_scheduler"] = lambda optimizer: torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.9)
    elif exp == 22:
        model = Slim_Convolutional(rnn=False)
        hyperparams["n_epochs"] = 10
        hyperparams["lr_scheduler_name"] = "Plateau_0.1_2_1e-4"
        hyperparams["lr_scheduler"] = lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=2, threshold=1e-4)
        hyperparams["needed_loss_scheduler"] = True

    elif exp == 30:  # 15it / sec -> H=4
        # NOK!!!!
        model = FlexiConv(h_dim=4)
        hyperparams["n_epochs"] = 100
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Flexconv H=4"
    elif exp == 31:  # 4 it / sec -> H=16
        model = FlexiConv(h_dim=16)
        hyperparams["n_epochs"] = 100
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Flexconv H=16"
    elif exp == 32:  # ??
        model = FlexiConv(h_dim=8)
        hyperparams["n_epochs"] = 100
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Flexconv H=8"
    elif exp == 33:  # 2 it / sec -> H=32
        model = FlexiConv(h_dim=32)
        hyperparams["n_epochs"] = 100
        hyperparams["batch_sizes"] = (512, 1024)
    elif exp == 34:
        model = FlexiConv(h_dim=16)
        hyperparams["n_epochs"] = 100
        hyperparams["batch_sizes"] = (512, 1024)
        augment_config[AUGMENT_TRIM] = True
        augment_config[AUGMENT_ROTATE] = True
        hyperparams["annotation"] = "Flexconv H=16 aug"
    elif exp == 35:  # 4 it / sec -> H=16
        model = FlexiConv(h_dim=16)
        hyperparams["n_epochs"] = 300
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Flexconv H=16"
    elif exp == 36:  # 4 it / sec -> H=16
        model = FlexiConv(h_dim=16)
        hyperparams["n_epochs"] = 20
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Flexconv H=16 train HighSNR"
        augment_config[SNR_FILTER] = [10, 20, 30]
    elif exp == 37:  # 4 it / sec -> H=16
        model = FlexiConv(h_dim=16)
        hyperparams["n_epochs"] = 100
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Flexconv H=16 train on HighSNR"
        augment_config[SNR_FILTER] = [10, 20, 30]
    elif exp == 38:
        # LR 1E-2 starts 17% NOK
        # LR 1E-3 starts 37.6% ... epoch 10 -> 52% epoch 13 -> 58%
        model = FlexiConv(h_dim=8, k_size=[5])
        hyperparams["n_epochs"] = 30
        hyperparams["lr"] = 1E-3
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Flexconv H=8 Large Kernels K=5"
    elif exp == 39:
        # LR 1E-2 starts 17% NOK, 5E-3 NOK, 2E-3 NOK
        # LR 1E-3 starts 28.8% ... epoch 10 -> 52% epoch 20 -> 57%
        model = FlexiConv(h_dim=8, k_size=[7])
        hyperparams["n_epochs"] = 30
        hyperparams["lr"] = 1E-3
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Flexconv H=8 Large Kernels K=7"
    elif exp == 40:
        # LR 1E-3 starts ?
        model = FlexiConv(h_dim=8, k_size=[7])
        hyperparams["n_epochs"] = 30
        hyperparams["lr"] = 1E-3
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Flexconv H=8 Large Kernels K=7"
    elif exp == 41:
        # LR 1E-3 starts 31%
        model = FlexiConv(h_dim=16, k_size=[5])
        hyperparams["n_epochs"] = 50
        hyperparams["lr"] = 1E-3
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Flexconv H=16 Large Kernels K=7"
        hyperparams["lr_scheduler_name"] = "Plateau_0.5_2"
        hyperparams["lr_scheduler"] = lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=2, threshold=2e-4)
        hyperparams["needed_loss_scheduler"] = True

    elif exp == 42:  # >78.35%
        model = FlexiConv(h_dim=8, k_size=[9])
        hyperparams["n_epochs"] = 1000
        hyperparams["lr"] = 5E-4
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Flexconv H=8 Large Kernels K=9"
    elif exp == 43:
        model = FlexiConv(h_dim=8, k_size=[9])
        model = torch.load(ROOT_DIR/"exp_0042/best_model.pth")
        hyperparams["n_epochs"] = 300
        hyperparams["lr"] = 1E-4
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Flexconv H=8 Large Kernels K=9"
        hyperparams["lr_scheduler_name"] = "Plateau_0.5_2"
        hyperparams["lr_scheduler"] = lambda optimizer: \
            torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=2, threshold=2e-4)
        hyperparams["needed_loss_scheduler"] = True

    elif exp == 50:  # 30% @epoch 72
        model = FlexiConv(h_dim=8, k_size=[9], augmented_inputs=True)
        hyperparams["n_epochs"] = 100
        hyperparams["lr"] = 1E-3
        hyperparams["batch_sizes"] = (1024, 1024)
        hyperparams["annotation"] = "Flexconv H=8 Large Kernels K=9"
        hyperparams["lr_scheduler_name"] = "Plateau_0.5_2"
        hyperparams["lr_scheduler"] = lambda optimizer: \
            torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=2, threshold=2e-4)
        hyperparams["needed_loss_scheduler"] = True
    elif exp == 60:  # ? 70% at 38 epochs
        hyperparams["batch_sizes"] = (1024, 1024)
        model = Slim_Convolutional(rnn=False)
        hyperparams["n_epochs"] = 500
        hyperparams["lr"] = 1E-3
        hyperparams["annotation"] = "Slim Convolutional h=16, h_c=2128 k=5, tpool=8"
    elif exp == 61:  # 60% at 22 epochs
        model = Slim_Convolutional(
            rnn=False,
            h_dim=32,  # 16 DEFAULT
            h_dim_classifier=256,  # 128 DEFAULT
            k_size=7,  # 5 DEFAULT
            pool_temporal=8
        )
        hyperparams["batch_sizes"] = (1024, 1024)
        hyperparams["n_epochs"] = 500
        hyperparams["lr"] = 1E-3
        hyperparams["annotation"] = "Fat Convolutional h=32, h_c=256, k=7, tpool=8"
    elif exp == 62:  # >83.4% at 200 epochs - OVervfits at 150
        model = FlexiConv(h_dim=8, k_size=[9])
        hyperparams["n_epochs"] = 500
        hyperparams["lr"] = 5E-4
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Flexconv H=8 Large Kernels K=9"
    elif exp == 63:  # 85.1%
        model = torch.load(ROOT_DIR/"exp_0062/best_model.pth",
                           map_location=torch.device('cpu'))
        hyperparams["n_epochs"] = 50
        hyperparams["lr"] = 5E-4
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Fine Tune ! Flexconv H=8 Large Kernels K=9"
        augment_config[AUGMENT_ROTATE] = True
        augment_config[AUGMENT_NOISE] = 0.01
    elif exp == 64:  # 84.8%
        model = torch.load(ROOT_DIR/"exp_0062/best_model.pth",
                           map_location=torch.device('cpu'))
        hyperparams["n_epochs"] = 50
        hyperparams["lr"] = 1E-3
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Fine Tune ! Flexconv H=8 Large Kernels K=9"
        augment_config[AUGMENT_ROTATE] = True
        augment_config[AUGMENT_NOISE] = 0.01
    elif exp == 65:  # 85.18%
        model = torch.load(ROOT_DIR/"exp_0062/best_model.pth",
                           map_location=torch.device('cpu'))
        hyperparams["n_epochs"] = 50
        hyperparams["lr"] = 2E-4
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Fine Tune ! Flexconv H=8 Large Kernels K=9"
        augment_config[AUGMENT_ROTATE] = True
        augment_config[AUGMENT_NOISE] = 0.01
    elif exp == 66:  # 85.24%
        model = torch.load(ROOT_DIR/"exp_0062/best_model.pth",
                           map_location=torch.device('cpu'))
        hyperparams["n_epochs"] = 50
        hyperparams["lr"] = 5E-4
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Fine Tune ! Flexconv H=8 Large Kernels K=9"
        augment_config[AUGMENT_ROTATE] = True
    elif exp == 67:  # 85%
        model = torch.load(ROOT_DIR/"exp_0062/best_model.pth",
                           map_location=torch.device('cpu'))
        hyperparams["n_epochs"] = 50
        hyperparams["lr"] = 5E-4
        hyperparams["batch_sizes"] = (512, 1024)
        hyperparams["annotation"] = "Fine Tune ! Flexconv H=8 Large Kernels K=9"
        augment_config[AUGMENT_TRIM] = True
        augment_config[AUGMENT_ROTATE] = True
    return model, hyperparams, augment_config
