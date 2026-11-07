"""Fit a comfortable 3D reach volume from five camera-space hand poses."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import dist, isfinite
from statistics import median


Vec3 = tuple[float, float, float]
TARGETS = ((0.12, 0.14), (0.88, 0.14), (0.88, 0.86), (0.12, 0.86), (0.5, 0.5))
TARGET_WIDTH = TARGETS[1][0] - TARGETS[0][0]
TARGET_HEIGHT = TARGETS[2][1] - TARGETS[1][1]
MIN_DEPTH_M = 0.30
MAX_DEPTH_M = 1.50
OVERSCAN = 0.10
CORNERS = (0, 1, 2, 3)
CENTER = 4


def subtract(a: Vec3, b: Vec3) -> Vec3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def add(a: Vec3, b: Vec3) -> Vec3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


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


class CalibrationFitError(ValueError):
    def __init__(self, message: str, targets: tuple[int, ...]) -> None:
        super().__init__(message)
        self.targets = targets


def steady_pose(poses: Sequence[Pose]) -> Pose | None:
    if len(poses) < 5 or not all(pose.valid() for pose in poses):
        return None
    tip = median_point([pose.tip for pose in poses])
    palm = median_point([pose.palm for pose in poses])
    if any(dist(pose.tip, tip) > 0.018 or dist(pose.palm, palm) > 0.018 for pose in poses):
        return None
    return Pose(tip, palm)


def ray_point(point: Vec3, depth: float) -> Vec3:
    return point[0] * depth / point[2], point[1] * depth / point[2], depth


@dataclass(frozen=True)
class _ReachPlane:
    origin: Vec3
    right: Vec3
    down: Vec3
    normal: Vec3
    width: float
    height: float
    skew: float
    ray_depth: float | None = None

    @property
    def viewport_width_m(self) -> float:
        return self.width / TARGET_WIDTH

    def coordinates(self, tip: Vec3) -> tuple[float, float]:
        if self.ray_depth is not None:
            tip = ray_point(tip, self.ray_depth)
        relative = subtract(tip, self.origin)
        v = dot(relative, self.down)
        return dot(relative, self.right) - v / self.height * self.skew, v

    def aim(self, tip: Vec3) -> tuple[float, float]:
        u, v = self.coordinates(tip)
        return 0.5 + u / self.width * TARGET_WIDTH, 0.5 + v / self.height * TARGET_HEIGHT

    @property
    def depth_reach_m(self) -> float:
        return min(0.28, max(0.14, self.viewport_width_m * 0.55))


@dataclass(frozen=True)
class Calibration:
    plane: _ReachPlane
    points: tuple[Pose, ...]
    neutral_depth_m: float
    center_adjusted: bool

    @property
    def viewport_width_m(self) -> float:
        return self.plane.viewport_width_m

    @property
    def uses_ray_projection(self) -> bool:
        return self.plane.ray_depth is not None

    def aim(self, tip: Vec3) -> tuple[float, float]:
        return self.plane.aim(tip)

    def project(self, pose: Pose) -> tuple[float, float, float] | None:
        """Return stage x/y and palm travel in viewport widths, positive toward the visitor."""
        if not pose.valid():
            return None
        x, y = self.aim(pose.tip)
        palm_travel = dot(pose.palm, self.plane.normal) - self.neutral_depth_m
        reach = self.plane.depth_reach_m
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


def _fit_axes(tips: Sequence[Vec3]) -> _ReachPlane:
    top_left, top_right, bottom_right, bottom_left = tips
    origin = average((top_left, top_right, bottom_right, bottom_left))
    horizontal = average((subtract(top_right, top_left), subtract(bottom_right, bottom_left)))
    vertical = average((subtract(bottom_left, top_left), subtract(bottom_right, top_right)))
    width = dot(horizontal, horizontal) ** 0.5
    if not 0.14 <= width <= 1.20:
        raise CalibrationFitError('Use a comfortable reach and spread the left and right points apart.', CORNERS)
    right = scale(horizontal, 1 / width)
    skew = dot(vertical, right)
    orthogonal = subtract(vertical, scale(right, skew))
    height = dot(orthogonal, orthogonal) ** 0.5
    if not 0.10 <= height <= 0.85 or abs(skew) > height * 0.50:
        raise CalibrationFitError('Keep the columns aligned and spread the top and bottom points apart.', CORNERS)
    down = scale(orthogonal, 1 / height)
    normal = cross(right, down)
    if dot(normal, origin) < 0:
        normal = scale(normal, -1)
    if normal[2] < 0.40:
        raise CalibrationFitError('Keep your reach area facing the camera.', CORNERS)
    return _ReachPlane(origin, right, down, normal, width, height, skew)


def _check_corner_errors(values: Sequence[float], limit: float, message: str) -> None:
    outliers = tuple(sorted((i for i in CORNERS if values[i] > limit), key=values.__getitem__, reverse=True))
    if outliers:
        raise CalibrationFitError(message, outliers)


def _check_palm_depths(poses: Sequence[Pose], plane: _ReachPlane, limit: float) -> None:
    depths = [dot(pose.palm, plane.normal) for pose in poses]
    neutral = median(depths)
    errors = [abs(depth - neutral) for depth in depths]
    if sum(error > limit for error in errors) > 1:
        _check_corner_errors(errors, limit, 'Bring the highlighted hand position closer to the depth of the other corners.')


def _fit_projected_corners(poses: Sequence[Pose]) -> _ReachPlane:
    # A steady hand may reach different distances from the camera around the
    # screen. Intersect its camera rays with one reference depth for aim only.
    depths = [pose.tip[2] for pose in poses]
    depth = median(depths)
    _check_corner_errors([abs(value - depth) for value in depths], 0.30,
                         'Bring the highlighted fingertip closer to the depth of the other corners.')
    tips = [ray_point(pose.tip, depth) for pose in poses]
    plane = _fit_axes(tips)
    xy = [plane.coordinates(tip) for tip in tips]
    for i in CORNERS:
        a, b, c = xy[i], xy[(i + 1) % 4], xy[(i + 2) % 4]
        turn = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if turn <= 0.06 * plane.width * plane.height:
            raise CalibrationFitError('Move your hand around the four corners in the displayed order.', CORNERS)

    # Use only the common reach of both rows and columns. This keeps the
    # calibrated rectangle inside the observed convex shape on screen.
    u, v = zip(*xy)
    left, right = max(u[0], u[3]), min(u[1], u[2])
    top, bottom = max(v[0], v[1]), min(v[2], v[3])
    width, height = right - left, bottom - top
    if width < max(0.14, plane.width * 0.65) or height < max(0.10, plane.height * 0.65):
        raise CalibrationFitError('Spread your hand positions into a wider, taller rectangle.', CORNERS)
    mid_u, mid_v = (left + right) / 2, (top + bottom) / 2
    origin = add(plane.origin, add(scale(plane.right, mid_u + mid_v / plane.height * plane.skew), scale(plane.down, mid_v)))
    plane = replace(plane, origin=origin, width=width, height=height, skew=plane.skew * height / plane.height, ray_depth=depth)
    _check_palm_depths(poses, plane, min(0.22, plane.depth_reach_m))
    return plane


def _fit_corners(poses: Sequence[Pose]) -> _ReachPlane:
    if len(poses) != len(CORNERS):
        raise CalibrationFitError('Capture all four corners with valid depth.', CORNERS)
    invalid = tuple(i for i, pose in enumerate(poses) if not pose.valid())
    if invalid:
        raise CalibrationFitError('Capture this corner again with valid depth.', invalid)

    try:
        plane = _fit_axes([pose.tip for pose in poses])
        _check_corner_errors([abs(dot(subtract(pose.tip, plane.origin), plane.normal)) for pose in poses],
                             0.060, 'Keep your fingertip near the same plane as the other corners.')
        _check_corner_errors([dist(plane.aim(pose.tip), target) for pose, target in zip(poses, TARGETS)],
                             0.13, 'Aim at the highlighted corner to make an even rectangle.')
    except CalibrationFitError:
        return _fit_projected_corners(poses)

    _check_palm_depths(poses, plane, 0.11)
    return plane


def fit_corners(poses: Sequence[Pose]) -> None:
    """Validate the four ordered corners before requesting the advisory center."""
    _fit_corners(poses)


def fit(poses: Sequence[Pose]) -> Calibration:
    if len(poses) != len(TARGETS):
        raise CalibrationFitError('Capture all five points with valid depth.', (*CORNERS, CENTER))
    plane = _fit_corners(poses[:CENTER])
    center = poses[CENTER]
    if not center.valid():
        raise CalibrationFitError('Capture the center again with valid depth.', (CENTER,))

    depths = [dot(pose.palm, plane.normal) for pose in poses[:CENTER]]
    median_depth = median(depths)
    center_depth = dot(center.palm, plane.normal)
    use_center = (abs(dot(subtract(center.tip, plane.origin), plane.normal)) <= 0.075
                  and dist(plane.aim(center.tip), TARGETS[CENTER]) <= 0.13
                  and abs(center_depth - median_depth) <= 0.11
                  and all(abs(depth - center_depth) <= plane.depth_reach_m for depth in depths))
    return Calibration(plane, tuple(poses), center_depth if use_center else median_depth, not use_center)
