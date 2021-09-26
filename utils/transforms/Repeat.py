"""
Repeat transform conforming to torchvision.transforms semantics.
"""

# pylint: disable=no-name-in-module

from torch.nn import Module

class Repeat(Module):
    """
    Calls the tensor.repeat function on a tensor using the semantics
    followed by torchvision.transforms.
    """

    def __init__(self, *args):
        super().__init__()
        self.repeats = args

    def forward(self, sample):
        """
        Perform the main repeating action described in the class description.
        """
        return sample.repeat(*self.repeats)
