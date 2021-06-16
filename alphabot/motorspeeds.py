"""Module for MotorSpeeds class."""


class MotorSpeeds:
    def __init__(self, left_motor_speed: float = 0.0, right_motor_speed: float = 0.0) -> None:
        self._left_motor_speed = left_motor_speed
        self._right_motor_speed = right_motor_speed

    @property
    def left_motor_speed(self) -> float:
        return self._left_motor_speed

    @left_motor_speed.setter
    def left_motor_speed(self, speed: float) -> None:
        if speed > 1.0:
            self._left_motor_speed = 1.0
        elif speed < -1.0:
            self._left_motor_speed = -1.0
        else:
            self._left_motor_speed = speed

    @property
    def right_motor_speed(self) -> float:
        return self._right_motor_speed

    @right_motor_speed.setter
    def right_motor_speed(self, speed: float) -> None:
        if speed > 1.0:
            self._right_motor_speed = 1.0
        elif speed < -1.0:
            self._right_motor_speed = -1.0
        else:
            self._right_motor_speed = speed
