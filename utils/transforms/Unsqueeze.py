"""
Unsqueeze transform conforming to torchvision.transforms semantics.
"""

# pylint: disable=no-name-in-module

from torch.nn import Module

class Unsqueeze(Module):
    """
    Calls the tensor.unsqueeze function on a tensor using the semantics
    followed by torchvision.transforms.
    """

    def __init__(self, *args):
        super().__init__()
        self.unsqueeze_args = args

    def forward(self, sample):
        """
        Perform the main unsqueezing action described in the class description.
        """
        return sample.unsqueeze(*self.unsqueeze_args)
