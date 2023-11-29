import torch

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


class Exp_baseline(torch.nn.Module):
    def __init__(self, ch_in: int = 2, n_classes: int = N_CLASSES,
                 h_dim=8, h_dim_classifier=128, k_size=5,
                 pool_temporal=8
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
        self.pool_temporal = torch.nn.AvgPool1d(kernel_size=pool_temporal)
        flat_dim = h_dim*2048//4//4 // pool_temporal
        self.fc1 = torch.nn.Linear(flat_dim, h_dim_classifier)
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
        self.fc1 = torch.nn.Linear(h_dim, h_dim_classifier)
        self.fc2 = torch.nn.Linear(h_dim_classifier, n_classes)
        self.relu = torch.nn.ReLU()
        self.classifier = torch.nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2
        )

    def forward(self, sig_in: torch.Tensor) -> torch.Tensor:
        """Perform recrurent feature extraction followed by classifier head

        Args:
            sig_in (torch.Tensor): [N, C, T]

        Returns:
            torch.Tensor: logits (not probabilities) [N, n_classes]
        """
        # print(sig_in.shape)
        sig_in = sig_in.transpose(1, 2)
        # print(sig_in.shape)
        full, _ = self.gru(sig_in)
        hidden_features = full[:, -1, :]
        # print(hidden_features.shape)
        # Vector classifier
        logits = self.classifier(hidden_features)
        return logits


def get_experience(exp):
    hyperparams = dict(
        lr=1E-4,
        n_epochs=100,
        batch_sizes=(256, 128)
    )
    if exp == 0:
        model = VanillaClassifier()  # 40%
    elif exp == 1:
        model = Exp_baseline(pool_temporal=8)  # 49%
    elif exp == 2:
        model = Exp_baseline(pool_temporal=4)  # 61%
        hyperparams = dict(
            lr=1E-4,
            n_epochs=300,
            batch_sizes=(256, 128)
        )
    elif exp == 3:
        model = RnnBaseline(bidirectional=False)
    return model, hyperparams
