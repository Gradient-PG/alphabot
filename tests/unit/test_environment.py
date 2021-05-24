"""Tests for Environment class."""
import pytest

from alphabot import Environment


class TestEnvironment:
    @pytest.fixture()
    def environment_class_initializable(self, mocker):
        if hasattr(Environment, "__abstractmethods__"):
            Environment.__abstractmethods__ = []
        Environment.current_motor_speed = mocker.sentinel.current_motor_speed_values
        Environment.current_observation = mocker.sentinel.current_observations_values
        Environment.action_space = mocker.sentinel.action_space_array
        Environment.state_space = mocker.sentinel.state_space_array
        return Environment

    @pytest.fixture()
    def environment(self, environment_class_initializable, mocker):
        mocker.patch("alphabot.Environment.__init__", mocker.Mock(return_value=None))
        environment = Environment()
        return environment

    def test_environment_not_initializable(self):
        with pytest.raises(TypeError):
            _ = Environment()

    def test_get_next_observation_not_implemented(self, environment):
        with pytest.raises(NotImplementedError):
            _ = environment.get_next_observation()

    def test_steer_not_implemented(self, environment, mocker):
        left = mocker.create_autospec(float)
        right = mocker.create_autospec(float)
        with pytest.raises(NotImplementedError):
            _ = environment.steer(left, right)
