"""Module for Environment abstract class."""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray
    from .observation import Observation
    from .motorspeeds import MotorSpeeds


class Environment(ABC):
    """Abstract class for environment to train Alphabot."""

    current_motor_speed: "MotorSpeeds"
    current_observation: "Observation"
    action_space: "ndarray"
    state_space: "ndarray"

    @abstractmethod
    def steer(self, left: float, right: float) -> None:
        """
        Modify speed of each motor based on passed arguments.

        :param left: Modification of speed of left motor
        :param right: Modification of speed of right motor
        :raises ValueError: if invalid parameters passed
        :raises AlphabotException: if error encountered during modifying motor speeds
        """
        raise NotImplementedError

    @abstractmethod
    def get_next_observation(self) -> "Observation":
        """
        Return next valid observation.

        :return: Next valid observation object
        :raises AlphabotException: if error encountered during getting observation
        """
        raise NotImplementedError
