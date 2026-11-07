"""Fit a comfortable 3D reach volume from five camera-space hand poses."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import dist, isfinite
from statistics import median


Vec3 = tuple[float, float, float]
TARGETS = ((0.12, 0.14), (0.88, 0.14), (0.88, 0.86), (0.12, 0.86), (0.5, 0.5))
TARGET_WIDTH = TARGETS[1][0] - TARGETS[0][0]
TARGET_HEIGHT = TARGETS[2][1] - TARGETS[1][1]
MIN_DEPTH_M = 0.30
MAX_DEPTH_M = 1.50
OVERSCAN = 0.10


def subtract(a: Vec3, b: Vec3) -> Vec3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def scale(point: Vec3, factor: float) -> Vec3:
    return point[0] * factor, point[1] * factor, point[2] * factor


def cross(a: Vec3, b: Vec3) -> Vec3:
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]


def average(points: Sequence[Vec3]) -> Vec3:
    x, y, z = zip(*points)
    return sum(x) / len(x), sum(y) / len(y), sum(z) / len(z)


def median_point(points: Sequence[Vec3]) -> Vec3:
    x, y, z = zip(*points)
    return median(x), median(y), median(z)


def valid_point(point: Vec3) -> bool:
    return len(point) == 3 and all(isfinite(value) for value in point) and MIN_DEPTH_M <= point[2] <= MAX_DEPTH_M


@dataclass(frozen=True)
class Pose:
    tip: Vec3
    palm: Vec3

    def valid(self) -> bool:
        return valid_point(self.tip) and valid_point(self.palm)


def steady_pose(poses: Sequence[Pose]) -> Pose | None:
    if len(poses) < 5 or not all(pose.valid() for pose in poses):
        return None
    tip = median_point([pose.tip for pose in poses])
    palm = median_point([pose.palm for pose in poses])
    if any(dist(pose.tip, tip) > 0.018 or dist(pose.palm, palm) > 0.018 for pose in poses):
        return None
    return Pose(tip, palm)


@dataclass(frozen=True)
class Calibration:
    origin: Vec3
    right: Vec3
    down: Vec3
    normal: Vec3
    width: float
    height: float
    skew: float
    points: tuple[Pose, ...]

    @property
    def viewport_width_m(self) -> float:
        return self.width / TARGET_WIDTH

    def aim(self, tip: Vec3) -> tuple[float, float]:
        relative = subtract(tip, self.origin)
        vertical = dot(relative, self.down) / self.height
        horizontal = (dot(relative, self.right) - vertical * self.skew) / self.width
        return 0.5 + horizontal * TARGET_WIDTH, 0.5 + vertical * TARGET_HEIGHT

    def project(self, pose: Pose) -> tuple[float, float, float] | None:
        """Return stage x/y and palm travel in viewport widths, positive toward the visitor."""
        if not pose.valid():
            return None
        x, y = self.aim(pose.tip)
        palm_travel = dot(subtract(pose.palm, self.points[-1].palm), self.normal)
        reach = min(0.28, max(0.14, self.viewport_width_m * 0.55))
        if not (-OVERSCAN <= x <= 1 + OVERSCAN and -OVERSCAN <= y <= 1 + OVERSCAN and abs(palm_travel) <= reach):
            return None
        return max(0.0, min(1.0, x)), max(0.0, min(1.0, y)), palm_travel / self.viewport_width_m

    def to_dict(self) -> dict:
        return {
            'version': 1,
            'points': [{'tip': list(pose.tip), 'palm': list(pose.palm)} for pose in self.points],
        }

    @classmethod
    def from_dict(cls, data: object) -> 'Calibration':
        if not isinstance(data, dict) or type(data.get('version')) is not int or data['version'] != 1:
            raise ValueError('Unsupported calibration profile.')
        points = data.get('points')
        if not isinstance(points, list) or len(points) != len(TARGETS):
            raise ValueError('Invalid calibration points.')

        def vector(value: object) -> Vec3:
            if not isinstance(value, (tuple, list)) or len(value) != 3 or any(type(v) not in (int, float) for v in value):
                raise ValueError('Invalid calibration coordinate.')
            try:
                return float(value[0]), float(value[1]), float(value[2])
            except OverflowError as error:
                raise ValueError('Invalid calibration coordinate.') from error

        if any(not isinstance(point, dict) for point in points):
            raise ValueError('Invalid calibration points.')
        return fit([Pose(vector(point.get('tip')), vector(point.get('palm'))) for point in points])


def fit(poses: Sequence[Pose]) -> Calibration:
    if len(poses) != len(TARGETS) or not all(pose.valid() for pose in poses):
        raise ValueError('Capture all five points with valid depth.')

    top_left, top_right, bottom_right, bottom_left = (pose.tip for pose in poses[:4])
    origin = average((top_left, top_right, bottom_right, bottom_left))
    horizontal = average((subtract(top_right, top_left), subtract(bottom_right, bottom_left)))
    vertical = average((subtract(bottom_left, top_left), subtract(bottom_right, top_right)))
    width = dot(horizontal, horizontal) ** 0.5
    if not 0.16 <= width <= 1.20:
        raise ValueError('Use a comfortable reach and spread the left and right points apart.')
    right = scale(horizontal, 1 / width)
    skew = dot(vertical, right)
    orthogonal = subtract(vertical, scale(right, skew))
    height = dot(orthogonal, orthogonal) ** 0.5
    if not 0.12 <= height <= 0.85 or abs(skew) > height * 0.30:
        raise ValueError('Keep the columns aligned and spread the top and bottom points apart.')
    down = scale(orthogonal, 1 / height)
    normal = cross(right, down)
    if dot(normal, origin) < 0:
        normal = scale(normal, -1)
    if normal[2] < 0.40:
        raise ValueError('Keep the five points in a flat area facing the camera.')

    calibration = Calibration(origin, right, down, normal, width, height, skew, tuple(poses))
    center = poses[-1].palm
    if any(abs(dot(subtract(pose.tip, origin), normal)) > 0.033 for pose in poses):
        raise ValueError('Keep all five fingertip positions in the same plane, parallel to the screen.')
    if any(dist(calibration.aim(pose.tip), target) > 0.09 for pose, target in zip(poses, TARGETS)):
        raise ValueError('Make an even rectangle with the center halfway between the four corners.')
    if any(abs(dot(subtract(pose.palm, center), normal)) > 0.085 for pose in poses):
        raise ValueError('Keep the same pointing pose and palm depth at all five points.')
    return calibration
