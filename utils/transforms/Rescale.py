"""
Rescale transform conforming to torchvision.transforms semantics.
"""

# pylint: disable=no-name-in-module

from torch import min, max
from torch.nn import Module

class Rescale(Module):
    """
    Define a transform that transforms values within a sample to have the range
    [begin, end]. This is not done on a per-channel basis but is instead done on
    the whole sample.
    """

    def __init__(self, begin, end):
        """
        Initialize the beginning and ending limits of the desired interval.
        """
        super().__init__()
        if (begin >= end):
            raise ValueError("Begin must be larger than end.")

        self.begin = begin
        self.end = end

    def forward(self, sample):
        """
        Perform the main reshaping action described in the class description.
        """
        return (((sample - min(sample)) / (max(sample) - min(sample)))
            * (self.end - self.begin) + self.begin)
