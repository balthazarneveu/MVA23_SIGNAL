import torch
from typing import Optional
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
        if ch_out is None:
            ch_out = ch_in
        if ch_inner is None:
            ch_inner = ch_in
        if ch_inner != ch_in and res:
            logging.warning(
                "Residual connection with different sizes" +
                f"{ch_in:=} != {ch_inner:=} -> Forcing ch_inner={ch_in}")
            ch_inner = ch_in
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

    def forward(self, sig_in: torch.Tensor) -> torch.Tensor:
        """Building block of convolutions

        Args:
            sig_in (torch.Tensor): [N, C, T]

        Returns:
            torch.Tensor: filtered [N, C_out, T//factor]
        """
        residual_filtered_1 = self.non_linearity(self.conv_first(sig_in))
        if self.res:
            residual_filtered_1 += sig_in
        filtered_2 = self.conv_downsample(residual_filtered_1)
        filtered_2 = self.non_linearity(filtered_2)
        return filtered_2


if __name__ == "__main__":
    N, C, T = 4, 2, 1024
    x = torch.randn(N, C, T)

    for k_size, ds, ch_out in product([3, 5, 7], [1, 2, 4], [None, 16, 32]):

        conv = BuildingBlockConvolutions(
            C,
            ch_out=ch_out,
            ch_inner=4,
            kernel_sizes=[k_size],
            downsample=ds
        )
        y = conv(x)
        print(x.shape, y.shape)
