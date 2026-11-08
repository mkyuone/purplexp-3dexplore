"""Optional operator-selected range for depth gestures, in camera meters."""

from dataclasses import dataclass


MIN_DEPTH_M = 0.30
MAX_DEPTH_M = 1.50
MIN_DEPTH_RANGE_M = 0.20


@dataclass(frozen=True)
class DepthRange:
    near_m: float
    far_m: float

    def __post_init__(self) -> None:
        if any(type(value) not in (int, float) or not MIN_DEPTH_M <= value <= MAX_DEPTH_M
               for value in (self.near_m, self.far_m)):
            raise ValueError('Depth endpoints must be between 30 and 150 cm.')
        if self.far_m < self.near_m + MIN_DEPTH_RANGE_M - 1e-9:
            raise ValueError('The far point must be at least 20 cm farther from the camera.')

    @classmethod
    def from_dict(cls, value: object) -> 'DepthRange':
        if not isinstance(value, dict) or set(value) != {'nearM', 'farM'}:
            raise ValueError('Set both the near and far depth endpoints.')
        return cls(value['nearM'], value['farM'])

    def to_dict(self) -> dict[str, float]:
        return {'nearM': self.near_m, 'farM': self.far_m}

    def map_depth(self, depth_m: float) -> float:
        fraction = (depth_m - self.near_m) / (self.far_m - self.near_m)
        return min(1.0, max(0.0, fraction)) - 0.5
