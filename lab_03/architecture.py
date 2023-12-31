import torch
from typing import Optional, List
from itertools import product
from properties import N_CLASSES
import logging


class BuildingBlockConvolutions(torch.nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: Optional[int] = None,
        ch_inner: Optional[int] = None,
        res: bool = True,
        kernel_sizes: int = [3],
        downsample: Optional[int] = 2
    ) -> None:
        super().__init__()
        self.downsample = downsample
        if ch_out is None:
            ch_out = ch_in
        if ch_inner is None:
            ch_inner = ch_in
        self.repeat_factor = 1
        if ch_inner != ch_in and res:
            if ch_inner < ch_in:
                logging.warning(
                    "Residual connection with different sizes: " +
                    f"{ch_in:=} < {ch_inner:=} -> Forcing ch_inner={ch_in}"
                )
                ch_inner = ch_in
            elif ch_inner > ch_in:
                # ch_inner = ch_in
                self.repeat_factor = ch_inner//ch_in
                assert self.repeat_factor*ch_in == ch_inner, \
                    "Cannot repeat residual connection properly"

        self.res = res
        self.conv_first = torch.nn.Conv1d(
            ch_in, ch_inner, kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0]//2
        )
        k_size_out = kernel_sizes[0]
        self.conv_downsample = torch.nn.Conv1d(
            ch_inner, ch_out, kernel_size=k_size_out,
            stride=downsample,
            padding=k_size_out//2)
        self.non_linearity = torch.nn.LeakyReLU()
        # @TODO: finish applying the residual connection to the downsample operator
        # self.residual_pooling = torch.nn.MaxPool1d(kernel_size=downsample)

    def forward(self, sig_in: torch.Tensor) -> torch.Tensor:
        """Building block of convolutions

        Args:
            sig_in (torch.Tensor): [N, C, T]

        Returns:
            torch.Tensor: filtered [N, C_out, T//downsample]
        """
        residual_filtered_1 = self.non_linearity(self.conv_first(sig_in))
        filtered_1 = residual_filtered_1
        if self.res:
            if self.repeat_factor > 1:
                # Repeating the input signal to match the residual size
                filtered_1 += sig_in.repeat(1, self.repeat_factor, 1)
            else:
                filtered_1 += sig_in
        residual_filtered_2 = self.conv_downsample(filtered_1)
        residual_filtered_2 = self.non_linearity(residual_filtered_2)
        filtered_2 = residual_filtered_2
        # @TODO: finish applying the residual connection to the downsample operator
        # if self.res:
        #     print(filtered_1.shape, self.residual_pooling(
        #         filtered_1).shape, filtered_2.shape, self.downsample)
        #     if self.downsample > 1:

        #         filtered_2 += self.residual_pooling(filtered_1)
        #     else:
        #         filtered_2 += filtered_1
        return filtered_2


class FlexiConv(torch.nn.Module):
    def __init__(
        self,
        dim_in: Optional[int] = 2,
        k_size: Optional[List[int]] = [3],
        h_dim: Optional[int] = 32,
        h_classifier: Optional[int] = 512,
        n_classes: Optional[int] = N_CLASSES,
        # augmented_inputs: Optional[bool] = False
    ):
        # self.augmented_inputs = augmented_inputs
        # if augmented_inputs:
        #     dim_in += 2
        
        logging.warning(
            f"FLEXICONV {dim_in=} {k_size=} {h_dim=} {h_classifier=} {n_classes=}")
        super().__init__()
        # -> C=2 , L=2048 (/1)
        self.block1 = BuildingBlockConvolutions(
            dim_in, ch_inner=h_dim, ch_out=h_dim, kernel_sizes=k_size)

        # -> C=32 (*1), L=1024 (/2)
        self.block2 = BuildingBlockConvolutions(
            h_dim, ch_inner=h_dim, ch_out=h_dim*2, kernel_sizes=k_size)

        # -> C=64 (*2), L=512  (/4)
        self.block3 = BuildingBlockConvolutions(
            h_dim*2, ch_inner=h_dim*4, ch_out=h_dim*4, kernel_sizes=k_size)

        # -> C=128 (*4), L=256 (/8)
        self.block4 = BuildingBlockConvolutions(
            h_dim*4, ch_inner=h_dim*4, ch_out=h_dim*8, kernel_sizes=k_size)

        # -> C=256 (*8), L=128 (/16)
        self.block5 = BuildingBlockConvolutions(
            h_dim*8, ch_inner=h_dim*8, ch_out=h_dim*16, kernel_sizes=k_size, downsample=4)

        # -> C=256 (*16), L=32 (/64)
        h_out = h_dim*16
        self.block6 = BuildingBlockConvolutions(
            h_dim*16, ch_inner=h_dim*16, ch_out=h_out, kernel_sizes=k_size, downsample=2)

        # -> C=512 (*16), L=16 (/128)

        self.classifier_1 = torch.nn.Conv1d(
            h_out, h_classifier, kernel_size=1)
        self.classifier_non_linearity = torch.nn.LeakyReLU()
        self.classifier_2 = torch.nn.Conv1d(
            h_classifier, n_classes, kernel_size=1)
        self.final_pool = torch.nn.AdaptiveMaxPool1d(1)

    def forward(self, sig_in: torch.Tensor) -> torch.Tensor:
        """Perform feature extraction followed by classifier head

        Args:
            sig_in (torch.Tensor): [N, C=2, T]

        Returns:
            torch.Tensor: logits (not probabilities) [N, n_classes]
        """
        # if self.augmented_inputs:
        #     x, y = sig_in[..., 0, ...], sig_in[..., 1, ...]
        #     # [N, C=2, T] -> [N, C=4, T]
        #     sig_in = torch.stack(
        #         [x, y, torch.arctan2(y, x), torch.sqrt(x**2+y**2)], dim=-2)
        #     assert sig_in.shape[-2] == 4, f"Wrong concatenation {sig_in.shape}"
        # L=2048 -> L=1024
        x1 = self.block1(sig_in)
        # L=1024 -> L=512
        x2 = self.block2(x1)
        # L=512  -> L=256
        x3 = self.block3(x2)
        # L=256  -> L=128
        x4 = self.block4(x3)
        # L=128  -> L=64
        x6 = self.block5(x4)
        # L=64  -> L=16
        y = self.classifier_1(x6)
        y = self.classifier_non_linearity(y)
        y = self.classifier_2(y)
        y = self.final_pool(y)

        return y.squeeze(-1)


if __name__ == "__main__":
    N, C, T = 4, 2, 2048
    for N, C, T in product([1], [2], [2048, 1024, 2000, 700, 400]):
        x = torch.randn(N, C, T)
        net = FlexiConv()
        y = net(x)
        print(f"in {x.shape} -> out {y.shape}")
    if False:
        N, C, T = 4, 2, 2048
        for k_size, ds, ch_out, ch_inner, res in product(
            [3, 5, 7],
            [1, 2, 4],
            [None, 16, 32],
            [None, 4, 12],
            [False, True]
        ):

            conv = BuildingBlockConvolutions(
                C,
                ch_out=ch_out,
                ch_inner=4,
                kernel_sizes=[k_size],
                downsample=ds,
                res=res
            )
            y = conv(x)
            print(x.shape, y.shape)
