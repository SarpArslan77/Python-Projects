import numpy as np

class Light():
    def __init__(
        self,
        position_x: int,
        position_y: int,
        amplitude: float,
        frequency: float,
    ) -> None:
        self.position_x = position_x
        self.position_y = position_y
        self.amplitude = amplitude
        self.frequency = frequency # Hz

    def __call__(self, distance: int, time: int = 0) -> float:
        return self.amplitude * np.sin(distance * self.frequency - time)
