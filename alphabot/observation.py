"""Module for Observation class."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray
    import preprocessor


class Observation:
    """Class for environment's returns."""

    state: "ndarray"
    reward: float
    is_finished: bool
    info: dict

    def __init__(self, state, reward, is_finished, info=None):
        self.state = state
        self.reward = reward
        self.is_finished = is_finished
        self.info = info

    def preprocess(self):
        """Preprocesses the information obtained from the environment."""
        preprocessed_observation = preprocessor.preprocess(self)
        self.state = preprocessed_observation.state
        self.reward = preprocessed_observation.reward
        self.is_finished = preprocessed_observation.is_finished
        self.info = preprocessed_observation.info
