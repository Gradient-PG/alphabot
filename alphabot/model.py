"""Module for Model base abstract class."""
import os

from abc import ABC, abstractmethod
from typing import Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .observation import Observation
    from .action import Action


class Model(ABC):
    """Abstract class for AI models used in Alphabot."""

    model_type: Optional[str] = None

    @abstractmethod
    def predict(self, observation: "Observation") -> "Action":
        """
        Predict next action based on observation passed as an argument.

        :param observation: Observation on which the prediction is based on
        :return: Predicted action
        :raises AlphabotException: if error encountered during predicting action
        """
        raise NotImplementedError

    @abstractmethod
    def save_weights(self, checkpoint_path: Union[str, bytes, os.PathLike]) -> None:
        """
        Save current weights of the model to selected path. Will create the checkpoint if does not exist.

        :param checkpoint_path: Path to checkpoint
        :raises AlphabotException: if could not save weights
        """
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, checkpoint_path: Union[str, bytes, os.PathLike]) -> None:
        """
        Load weights from provided checkpoint to current model.

        :param checkpoint_path: Path to checkpoint
        :raises AlphabotException: if could not load weights
        """
        raise NotImplementedError
