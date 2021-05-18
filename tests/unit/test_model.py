"""Tests for Model class."""
import pytest

from alphabot import Model, Observation
from os import PathLike


class TestModel:
    @pytest.fixture()
    def model_class_initializable(self, mocker):
        if hasattr(Model, "__abstractmethods__"):
            Model.__abstractmethods__ = []
        Model.model_type = mocker.sentinel.model_type_name
        return Model

    @pytest.fixture()
    def model(self, model_class_initializable, mocker):
        mocker.patch("alphabot.Model.__init__", mocker.Mock(return_value=None))
        model = Model()
        return model

    def test_model_not_initializable(self):
        with pytest.raises(TypeError):
            _ = Model()

    def test_predict_not_implemented(self, model, mocker):
        observation = mocker.create_autospec(Observation)
        with pytest.raises(NotImplementedError):
            _ = model.predict(observation)

    def test_save_weights_not_implemented(self, model, mocker):
        checkpoint_path = mocker.create_autospec(PathLike)
        with pytest.raises(NotImplementedError):
            _ = model.save_weights(checkpoint_path)

    def test_load_weights_not_implemented(self, model, mocker):
        checkpoint_path = mocker.create_autospec(PathLike)
        with pytest.raises(NotImplementedError):
            _ = model.load_weights(checkpoint_path)
