import torch
from data_loader import AUGMENT_TRIM, AUGMENT_NOISE, AUGMENT_ROTATE
N_CLASSES = 6


class VanillaClassifier(torch.nn.Module):
    def __init__(self, ch_in: int = 2, n_classes: int = N_CLASSES,
                 h_dim=8, h_dim_classifier=128, k_size=5
                 ) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            ch_in, h_dim, k_size, padding=k_size//2)
        self.conv2 = torch.nn.Conv1d(
            h_dim, h_dim, k_size, padding=k_size//2)
        self.pool = torch.nn.MaxPool1d(kernel_size=4)
        self.relu = torch.nn.ReLU()
        self.feature_extractor = torch.nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool
        )
        self.fc1 = torch.nn.Linear(h_dim, h_dim_classifier)
        self.fc2 = torch.nn.Linear(h_dim_classifier, n_classes)
        self.classifier = torch.nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2
        )

    def forward(self, sig_in: torch.Tensor) -> torch.Tensor:
        """Perform feature extraction followed by classifier head

        Args:
            sig_in (torch.Tensor): [N, C, T]

        Returns:
            torch.Tensor: logits (not probabilities) [N, n_classes]
        """
        # Convolution backbone
        # [N, C, T] -> [N, h, T//16]
        features = self.feature_extractor(sig_in)
        # Global pooling
        # [N, h, ?] -> [N, h]
        vector = torch.mean(features, dim=-1)

        # Vector classifier
        # [N, h] -> [N, n_classes]
        logits = self.classifier(vector)
        return logits


class Convolutional_baseline(torch.nn.Module):
    def __init__(self, ch_in: int = 2, n_classes: int = N_CLASSES,
                 h_dim=8, h_dim_classifier=128, k_size=5,
                 pool_temporal=8,
                 rnn=False,
                 h_dim_rnn=32
                 ) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            ch_in, h_dim, k_size, padding=k_size//2)
        self.conv2 = torch.nn.Conv1d(
            h_dim, h_dim, k_size, padding=k_size//2)
        self.pool = torch.nn.MaxPool1d(kernel_size=4)
        self.relu = torch.nn.ReLU()
        self.feature_extractor = torch.nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool
        )
        self.rnn = rnn
        conv_length_output = 2048//4//4
        if rnn:

            self.gru = torch.nn.GRU(
                h_dim,
                h_dim_rnn,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
            classif_dim = h_dim_rnn*2
        else:
            self.pool_temporal = torch.nn.AvgPool1d(kernel_size=pool_temporal)
            classif_dim = h_dim*conv_length_output // pool_temporal
        self.fc1 = torch.nn.Linear(classif_dim, h_dim_classifier)
        self.fc2 = torch.nn.Linear(h_dim_classifier, n_classes)
        self.classifier = torch.nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2
        )

    def forward(self, sig_in: torch.Tensor) -> torch.Tensor:
        """Perform feature extraction followed by classifier head

        Args:
            sig_in (torch.Tensor): [N, C, T]

        Returns:
            torch.Tensor: logits (not probabilities) [N, n_classes]
        """
        # Convolution backbone
        # [N, C, T] -> [N, h, T//16]
        features = self.feature_extractor(sig_in)
        # Global pooling
        # [N, h, ?] -> [N, h]
        if self.rnn:
            features = features.transpose(1, 2)
            full_hidden_states, _ = self.gru(features)
            vector = full_hidden_states[:, -1, :]
        else:
            vector = self.pool_temporal(features)
            vector = vector.view(-1, vector.shape[-1]*vector.shape[-2])
        # Vector classifier
        # [N, 64] -> [N, n_classes]
        logits = self.classifier(vector)
        return logits


class Slim_Convolutional(torch.nn.Module):
    def __init__(self, ch_in: int = 2,
                 n_classes: int = N_CLASSES,
                 h_dim=16,
                 h_dim_classifier=128,
                 k_size=5,
                 pool_temporal=8,
                 rnn=False,
                 h_dim_rnn=32
                 ) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            ch_in, h_dim, k_size, padding=k_size//2)
        self.conv2 = torch.nn.Conv1d(
            h_dim, h_dim, k_size, padding=k_size//2)
        self.conv3 = torch.nn.Conv1d(
            h_dim, h_dim, k_size, padding=k_size//2)
        self.conv4 = torch.nn.Conv1d(
            h_dim, h_dim, k_size, padding=k_size//2)
        self.pool = torch.nn.MaxPool1d(kernel_size=2)
        self.relu = torch.nn.ReLU()
        self.feature_extractor = torch.nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.conv3,
            self.relu,
            self.pool,
            self.conv4,
            self.relu,
            self.pool
        )
        self.rnn = rnn
        conv_length_output = 2048//4//4
        if rnn:

            self.gru = torch.nn.GRU(
                h_dim,
                h_dim_rnn,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
            classif_dim = h_dim_rnn*2
        else:
            self.pool_temporal = torch.nn.AvgPool1d(kernel_size=pool_temporal)
            classif_dim = h_dim*conv_length_output // pool_temporal
        self.fc1 = torch.nn.Linear(classif_dim, h_dim_classifier)
        self.fc2 = torch.nn.Linear(h_dim_classifier, n_classes)
        self.classifier = torch.nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2
        )

    def forward(self, sig_in: torch.Tensor) -> torch.Tensor:
        """Perform feature extraction followed by classifier head

        Args:
            sig_in (torch.Tensor): [N, C, T]

        Returns:
            torch.Tensor: logits (not probabilities) [N, n_classes]
        """
        # Convolution backbone
        # [N, C, T] -> [N, h, T//16]
        features = self.feature_extractor(sig_in)
        # Global pooling
        # [N, h, ?] -> [N, h]
        if self.rnn:
            features = features.transpose(1, 2)
            full_hidden_states, _ = self.gru(features)
            vector = full_hidden_states[:, -1, :]
        else:
            vector = self.pool_temporal(features)
            vector = vector.view(-1, vector.shape[-1]*vector.shape[-2])
        # Vector classifier
        # [N, 64] -> [N, n_classes]
        logits = self.classifier(vector)
        return logits


class RnnBaseline(torch.nn.Module):
    def __init__(
        self,
        ch_in: int = 2,
        n_classes: int = N_CLASSES,
        h_dim: int = 8,
        n_layers: int = 2,
        h_dim_classifier=128,
        bidirectional=False
    ) -> None:
        super().__init__()
        self.gru = torch.nn.GRU(
            ch_in,
            h_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc1 = torch.nn.Linear(
            h_dim*(2 if bidirectional else 1), h_dim_classifier)
        self.fc2 = torch.nn.Linear(h_dim_classifier, n_classes)
        self.relu = torch.nn.ReLU()
        self.classifier = torch.nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2
        )

    def forward(self, sig_in: torch.Tensor) -> torch.Tensor:
        """Perform recurrent feature extraction followed by classifier head

        Args:
            sig_in (torch.Tensor): [N, C, T] -> need to transpose first

        Returns:
            torch.Tensor: logits (not probabilities) [N, n_classes]
        """
        sig_in = sig_in.transpose(1, 2)
        full_hidden_states, _ = self.gru(sig_in)
        hidden_features = full_hidden_states[:, -1, :]
        # Vector classifier
        logits = self.classifier(hidden_features)
        return logits


def get_experience(exp):
    augment_config = {
        AUGMENT_TRIM: False,
        AUGMENT_NOISE: 0,
        AUGMENT_ROTATE: False
    }
    hyperparams = dict(
        lr=1E-4,
        n_epochs=100,
        batch_sizes=(256, 128)
    )
    if exp == 0:
        model = VanillaClassifier()  # 40%
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
    elif exp == 4:
        model = RnnBaseline(bidirectional=True, h_dim=16)
    elif exp == 5:
        model = Convolutional_baseline(rnn=True)  # 68.1%
        hyperparams["n_epochs"] = 500
    elif exp == 6:
        model = Slim_Convolutional(rnn=False)  # 73.3%
        hyperparams["n_epochs"] = 500
    elif exp == 7:
        model = Slim_Convolutional(rnn=False)  # ?*
        hyperparams["n_epochs"] = 1000
    # AUGMENTATION EXPERIMENTS
    elif exp == 8:
        model = Slim_Convolutional(rnn=False)  # ?
        hyperparams["n_epochs"] = 1000
        augment_config[AUGMENT_ROTATE] = True
    elif exp == 9:
        model = Slim_Convolutional(rnn=False)  # ?
        hyperparams["n_epochs"] = 1000
        augment_config[AUGMENT_NOISE] = 0.1
    elif exp == 10:
        model = Slim_Convolutional(rnn=False)  # ?
        hyperparams["n_epochs"] = 1000
        augment_config[AUGMENT_TRIM] = True
    elif exp == 11:
        model = Slim_Convolutional(rnn=False)  # ?
        hyperparams["n_epochs"] = 1000
        augment_config[AUGMENT_ROTATE] = True
        augment_config[AUGMENT_TRIM] = True
    elif exp == 12:
        model = Slim_Convolutional(rnn=False)  # ?
        hyperparams["n_epochs"] = 1000
        augment_config[AUGMENT_ROTATE] = True
        augment_config[AUGMENT_TRIM] = True
        augment_config[AUGMENT_NOISE] = 0.1

    return model, hyperparams, augment_config
