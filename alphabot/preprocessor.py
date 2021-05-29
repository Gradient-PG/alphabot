"""Module for preprocessor static class."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .observation import Observation


def preprocess(observation: "Observation") -> "Observation":
    """
    The main preprocess function used in Observation
    :param observation: Observation to be preprocessed
    :return Preprocessed observation
    :raises ValueError: if invalid parameters passed
    """
    raise NotImplementedError
