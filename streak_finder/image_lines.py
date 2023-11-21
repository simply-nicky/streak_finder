from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterator, List, NamedTuple, Optional, Set, Tuple, Union
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.special import loggamma
from .src import draw_line_table, fft_convolve, local_maxima, median
from tqdm.auto import tqdm

class LineBounds(NamedTuple):
    x0      : float
    y0      : float
    x1      : float
    y1      : float

    @property
    def x(self) -> np.ndarray:
        return np.array([self.x0, self.x1])

    @property
    def y(self) -> np.ndarray:
        return np.array([self.y0, self.y1])

    @property
    def tau(self) -> np.ndarray:
        return np.array([self.x1 - self.x0, self.y1 - self.y0])

    @property
    def theta(self) -> float:
        return np.arctan2(self.tau[1], self.tau[0])

    @property
    def norm(self) -> np.ndarray:
        return np.array([self.y1 - self.y0, self.x0 - self.x1])

    @property
    def magnitude(self) -> float:
        return (self.x1 - self.x0)**2 + (self.y1 - self.y0)**2

    def to_line(self, width: float) -> np.ndarray:
        return np.array([self.x0, self.y0, self.x1, self.y1, width])

class BoundList(NamedTuple):
    bounds  : List[LineBounds]

    def __add__(self, bset: BoundList) -> BoundList:
        return BoundList(self.bounds + bset.bounds)

    def __iter__(self) -> Iterator[LineBounds]:
        return iter(self.bounds)

    def __getitem__(self, indices: Any) -> Union[LineBounds, BoundList]:
        if isinstance(indices, int):
            return self.bounds[indices]
        return BoundList(self.bounds[indices])

    @property
    def size(self) -> int:
        return len(self.bounds)

    @property
    def x(self) -> np.ndarray:
        return np.stack([bound.x for bound in self])

    @property
    def y(self) -> np.ndarray:
        return np.stack([bound.y for bound in self])

    @property
    def tau(self) -> np.ndarray:
        return np.stack([bound.tau for bound in self])

    @property
    def theta(self) -> np.ndarray:
        return np.stack([bound.theta for bound in self])

    @property
    def norm(self) -> np.ndarray:
        return np.stack([bound.norm for bound in self])

    @property
    def magnitude(self) -> np.ndarray:
        return np.stack([bound.magnitude for bound in self])

    def to_line(self, width: float) -> np.ndarray:
        return np.stack([bound.to_line(width) for bound in self])

class Moments(NamedTuple):
    x0      : int
    y0      : int
    mu      : float
    mu_x    : float
    mu_y    : float
    mu_xx   : float
    mu_xy   : float
    mu_yy   : float

    @classmethod
    def new(cls, x0: int, y0: int, pixels: PixelSet) -> Moments:
        x, y, val = pixels.x, pixels.y, pixels.val
        return cls(x0, y0, np.sum(val), np.sum((x - x0) * val), np.sum((y - y0) * val),
                   np.sum((x - x0) * (x - x0) * val), np.sum((x - x0) * (y - y0) * val),
                   np.sum((y - y0) * (y - y0) * val))

    def __add__(self, m: Moments) -> Moments:
        if self.x0 != m.x0 or self.y0 != m.y0:
            m = m.update_seed(self.x0, self.y0)
        return Moments(self.x0, self.y0, self.mu + m.mu, self.mu_x + m.mu_x, self.mu_y + m.mu_y,
                       self.mu_xx + m.mu_xx, self.mu_xy + m.mu_xy, self.mu_yy + m.mu_yy)

    def __sub__(self, m: Moments) -> Moments:
        if self.x0 != m.x0 or self.y0 != m.y0:
            m = m.update_seed(self.x0, self.y0)
        return Moments(self.x0, self.y0, self.mu - m.mu, self.mu_x - m.mu_x, self.mu_y - m.mu_y,
                       self.mu_xx - m.mu_xx, self.mu_xy - m.mu_xy, self.mu_yy - m.mu_yy)

    def update_seed(self, x0: int, y0: int) -> Moments:
        dx, dy = self.x0 - x0, self.y0 - y0
        return Moments(x0, y0, self.mu,
                       self.mu_x + dx * self.mu, self.mu_y + dy * self.mu,
                       self.mu_xx + 2 * dx * self.mu_x + dx**2 * self.mu,
                       self.mu_xy + dx * self.mu_y + dy * self.mu_x + dx * dy * self.mu,
                       self.mu_yy + 2 * dy * self.mu_y + dy**2 * self.mu)

class PixelSet(NamedTuple):
    pixels  : Set[Tuple[int, int, float]]

    def __or__(self, pset: PixelSet) -> PixelSet:
        return PixelSet(self.pixels | pset.pixels)

    def __and__(self, pset: PixelSet) -> PixelSet:
        return PixelSet(self.pixels & pset.pixels)

    def __xor__(self, pset: PixelSet) -> PixelSet:
        return PixelSet(self.pixels ^ pset.pixels)

    @property
    def size(self) -> int:
        return len(self.pixels)

    @property
    def x(self) -> np.ndarray:
        return np.array([elem[0] for elem in self.pixels])

    @property
    def y(self) -> np.ndarray:
        return np.array([elem[1] for elem in self.pixels])

    @property
    def val(self) -> np.ndarray:
        return np.array([elem[2] for elem in self.pixels])

class LinePixels(NamedTuple):
    pixels  : PixelSet
    moments : Moments

    @classmethod
    def new(cls, pixels: PixelSet) -> LinePixels:
        mu = np.sum(pixels.val)
        if mu:
            x0 = np.sum(pixels.x * pixels.val) / mu
            y0 = np.sum(pixels.y * pixels.val) / mu
        else:
            x0 = np.mean(pixels.x)
            y0 = np.mean(pixels.y)
        return cls(pixels, Moments.new(x0, y0, pixels))

    def __or__(self, lpix: LinePixels) -> LinePixels:
        mdiff = Moments.new(self.moments.x0, self.moments.y0, self.pixels & lpix.pixels)
        moments = self.moments + lpix.moments - mdiff
        if moments.mu:
            x0 = moments.mu_x / moments.mu + self.x0
            y0 = moments.mu_y / moments.mu + self.y0
            return LinePixels(self.pixels | lpix.pixels, moments.update_seed(x0, y0))
        return LinePixels(self.pixels | lpix.pixels, moments)

    @property
    def size(self) -> int:
        return self.pixels.size

    @property
    def x(self) -> np.ndarray:
        return self.pixels.x

    @property
    def x0(self) -> int:
        return self.moments.x0

    @property
    def y(self) -> np.ndarray:
        return self.pixels.y

    @property
    def y0(self) -> int:
        return self.moments.y0

    @property
    def val(self) -> np.ndarray:
        return self.pixels.val

class Line(NamedTuple):
    """Maintains a list of pixels pertaining to a line."""
    pixels      : LinePixels
    linelets    : BoundList
    bounds      : LineBounds

    def shrink(self, dist: float) -> Line:
        vec = distance_to_line(self, self.pixels.x, self.pixels.y)
        mask = np.sqrt(np.sum(vec**2, axis=-1)) < dist
        pixels = PixelSet(set(zip(self.pixels.x[mask], self.pixels.y[mask], self.pixels.val[mask])))
        return Line(LinePixels.new(pixels), self.linelets, self.bounds)

    def update_bounds(self) -> Line:
        mx = self.pixels.moments.mu_x / self.pixels.moments.mu
        my = self.pixels.moments.mu_y / self.pixels.moments.mu
        mxx = self.pixels.moments.mu_xx / self.pixels.moments.mu - mx**2
        myy = self.pixels.moments.mu_yy / self.pixels.moments.mu - my**2
        mxy = self.pixels.moments.mu_xy / self.pixels.moments.mu - mx * my
        theta = 0.5 * np.arctan(2.0 * mxy / (mxx - myy))
        if myy > mxx:
            theta = theta + 0.5 * np.pi

        x0, y0 = mx + self.pixels.x0, my + self.pixels.y0
        taus = (self.pixels.x - x0) * np.cos(theta) + (self.pixels.y - y0) * np.sin(theta)
        tmin, tmax = np.min(taus), np.max(taus)

        bounds = LineBounds(x0 + tmin * np.cos(theta), y0 + tmin * np.sin(theta),
                            x0 + tmax * np.cos(theta), y0 + tmax * np.sin(theta))
        return Line(self.pixels, self.linelets, bounds)

class Structure(NamedTuple):
    idxs    : np.ndarray
    radius  : int
    rank    : int

    @classmethod
    def new(cls, radius: int, rank: int) -> Structure:
        struct = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=bool)
        struct[radius, radius] = True
        struct = binary_dilation(struct, iterations=rank)
        y, x = np.indices((2 * radius + 1, 2 * radius + 1)) - radius
        return cls(np.stack([x[struct], y[struct]], axis=-1), radius, rank)

    @property
    def size(self) -> int:
        return self.idxs.shape[0]

    @property
    def k0(self) -> np.ndarray:
        krn = np.zeros((2 * self.radius + 1, 2 * self.radius + 1))
        krn[self.idxs[:, 1] + self.radius, self.idxs[:, 0] + self.radius] = 1.0
        return krn

    @property
    def kx(self) -> np.ndarray:
        krn = np.zeros((2 * self.radius + 1, 2 * self.radius + 1))
        krn[self.idxs[:, 1] + self.radius, self.idxs[:, 0] + self.radius] = -self.idxs[:, 0]
        return krn

    @property
    def ky(self) -> np.ndarray:
        krn = np.zeros((2 * self.radius + 1, 2 * self.radius + 1))
        krn[self.idxs[:, 1] + self.radius, self.idxs[:, 0] + self.radius] = -self.idxs[:, 1]
        return krn

    @property
    def kxx(self) -> np.ndarray:
        krn = np.zeros((2 * self.radius + 1, 2 * self.radius + 1))
        krn[self.idxs[:, 1] + self.radius, self.idxs[:, 0] + self.radius] = self.idxs[:, 0]**2
        return krn

    @property
    def kxy(self) -> np.ndarray:
        krn = np.zeros((2 * self.radius + 1, 2 * self.radius + 1))
        krn[self.idxs[:, 1] + self.radius, self.idxs[:, 0] + self.radius] = np.prod(self.idxs, axis=-1)
        return krn

    @property
    def kyy(self) -> np.ndarray:
        krn = np.zeros((2 * self.radius + 1, 2 * self.radius + 1))
        krn[self.idxs[:, 1] + self.radius, self.idxs[:, 0] + self.radius] = self.idxs[:, 1]**2
        return krn

class Array(NamedTuple):
    data    : np.ndarray

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    def __getitem__(self, indices: Any) -> Any:
        return self.data[indices]

    def get(self, x: np.ndarray, y: np.ndarray, default: float=0.0) -> np.ndarray:
        out = np.full(np.array(x).shape, default, dtype=self.data.dtype)
        mask = (x < self.shape[1]) & (x >= 0) & (y < self.shape[0]) & (y >= 0)
        out[mask] = self.data[np.array(y)[mask], np.array(x)[mask]]
        return out

class Image(NamedTuple):
    data    : Array
    struct  : Structure
    mu      : Array
    mu_x    : Array
    mu_y    : Array
    mu_xx   : Array
    mu_xy   : Array
    mu_yy   : Array

    @classmethod
    def new(cls, data: np.ndarray, struct: Structure) -> Image:
        mu = fft_convolve(data, struct.k0)
        mu_x = fft_convolve(data, struct.kx)
        mu_y = fft_convolve(data, struct.ky)
        mu_xx = fft_convolve(data, struct.kxx)
        mu_xy = fft_convolve(data, struct.kxy)
        mu_yy = fft_convolve(data, struct.kyy)
        return cls(Array(data), struct, Array(mu), Array(mu_x), Array(mu_y),
                   Array(mu_xx), Array(mu_xy), Array(mu_yy))

    @property
    def radius(self) -> int:
        return self.struct.radius

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.arange(0, self.shape[0]), np.arange(0, self.shape[1])

    def background_level(self) -> float:
        return median(self.data[()], mask=(self.data[()] > 0.0), axis=(0, 1))

    def theta(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        mx = self.mu_x[y, x] / self.mu[y, x]
        my = self.mu_y[y, x] / self.mu[y, x]
        mxx = self.mu_xx[y, x] / self.mu[y, x] - mx**2
        mxy = self.mu_xy[y, x] / self.mu[y, x] - mx * my
        myy = self.mu_yy[y, x] / self.mu[y, x] - my**2
        theta = np.where(self.mu[y, x], 0.5 * np.arctan(2.0 * mxy / (mxx - myy)), 0.0)
        theta = np.where(myy > mxx, theta + 0.5 * np.pi, theta)
        return theta

    def update(self, x: np.ndarray, y: np.ndarray, vals: np.ndarray):
        xx, yy = x[:, None] + self.struct.idxs[:, 0], y[:, None] + self.struct.idxs[:, 1]
        mask = (xx < self.shape[1]) & (xx >= 0) & (yy < self.shape[0]) & (yy >= 0)
        vals = (self.data[y, x] * vals)

        np.subtract.at(self.data.data, (y, x), vals)
        np.subtract.at(self.mu.data, (yy[mask], xx[mask]),
                       np.broadcast_to(vals[:, None], mask.shape)[mask])
        np.subtract.at(self.mu_x.data, (yy[mask], xx[mask]),
                       (-self.struct.idxs[:, 0] * vals[:, None])[mask])
        np.subtract.at(self.mu_y.data, (yy[mask], xx[mask]),
                       (-self.struct.idxs[:, 1] * vals[:, None])[mask])

        np.subtract.at(self.mu_xx.data, (yy[mask], xx[mask]),
                       (self.struct.idxs[:, 0]**2 * vals[:, None])[mask])
        np.subtract.at(self.mu_xy.data, (yy[mask], xx[mask]),
                       (self.struct.idxs[:, 0] * self.struct.idxs[:, 1] * vals[:, None])[mask])
        np.subtract.at(self.mu_yy.data, (yy[mask], xx[mask]),
                       (self.struct.idxs[:, 1]**2 * vals[:, None])[mask])

def distance_to_streak(line: Line, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    vec0 = np.stack([line.bounds.x0 - x, line.bounds.y0 - y], axis=-1)

    if np.sum(line.bounds.tau**2):
        vec1 = np.stack([line.bounds.x1 - x, line.bounds.y1 - y], axis=-1)

        mag0 = np.sum(vec0**2, axis=-1, keepdims=True)
        mag1 = np.sum(vec1**2, axis=-1, keepdims=True)

        prd0 = np.sum(vec0 * line.bounds.tau, axis=-1, keepdims=True)
        prd1 = np.sum(vec1 * line.bounds.tau, axis=-1, keepdims=True)

        dist = np.where(mag0 < mag1,
                        vec0 - (prd0 / line.bounds.magnitude) * line.bounds.tau,
                        vec1 - (prd1 / line.bounds.magnitude) * line.bounds.tau)
        return np.where(prd0 * prd1 < 0.0, dist,
                        np.where(mag0 < mag1, vec0, vec1))

    return vec0

def distance_to_line(line: Line, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    vec0 = np.stack([line.bounds.x0 - x, line.bounds.y0 - y], axis=-1)

    if np.sum(line.bounds.tau**2):
        vec1 = np.stack([line.bounds.x1 - x, line.bounds.y1 - y], axis=-1)

        mag0 = np.sum(vec0**2, axis=-1, keepdims=True)
        mag1 = np.sum(vec1**2, axis=-1, keepdims=True)

        prd0 = np.sum(vec0 * line.bounds.tau, axis=-1, keepdims=True)
        prd1 = np.sum(vec1 * line.bounds.tau, axis=-1, keepdims=True)

        dist = np.where(mag0 < mag1,
                        vec0 - (prd0 / line.bounds.magnitude) * line.bounds.tau,
                        vec1 - (prd1 / line.bounds.magnitude) * line.bounds.tau)
        return dist

    return vec0

def project_to_line(line: Line, x: np.ndarray, y: np.ndarray, tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if line.bounds.magnitude:
        vec = np.stack([x - line.bounds.x0, y - line.bounds.y0], axis=-1)
        norm = np.stack((tau[..., 1], -tau[..., 0]), axis=-1)
        prd1 = np.sum(vec * line.bounds.norm, axis=-1)
        prd2 = np.sum(line.bounds.tau * norm, axis=-1)
        t = np.where((prd1 / prd2 > 0) & (prd1 / prd2 < 1e3), prd1 / prd2, 0)
        return x + t * tau[..., 0], y + t * tau[..., 1]

    return line.bounds.x0 * np.ones(x.shape), line.bounds.y0 * np.ones(x.shape)

def get_bounds(x: int, y: int, pixels: PixelSet, image: Image) -> LineBounds:
    if image.mu.get(x, y):
        x0 = image.mu_x[y, x] / image.mu[y, x] + x
        y0 = image.mu_y[y, x] / image.mu[y, x] + y
        theta = image.theta(x, y)
        taus = (pixels.x - x0) * np.cos(theta) + (pixels.y - y0) * np.sin(theta)
        tmin, tmax = np.min(taus), np.max(taus)

        # a = 0.5 * mu_yy / (mu_xx * mu_yy - mu_xy**2)
        # b = 0.5 * mu_xx / (mu_xx * mu_yy - mu_xy**2)
        # c = -0.5 * mu_xy / (mu_xx * mu_yy - mu_xy**2)
        # p = np.sqrt((a - b)**2 + 4 * c**2)

        # width = np.sqrt(1.0 / (a + b + p))

        return LineBounds(x0 + tmin * np.cos(theta), y0 + tmin * np.sin(theta),
                          x0 + tmax * np.cos(theta), y0 + tmax * np.sin(theta))

    return LineBounds(x, y, x, y)

def generate_line(x: int, y: int, image: Image, indices: np.ndarray) -> Line:
    xx, yy = x + indices[:, 0], y + indices[:, 1]
    vals = image.data.get(xx, yy)
    pixels = LinePixels.new(PixelSet(set(zip(xx, yy, vals))))
    bounds = get_bounds(x, y, pixels.pixels, image)
    return Line(pixels, BoundList([bounds,]), bounds)

def add_line(line: Line, new_line: Line) -> Line:
    pixels = line.pixels | new_line.pixels
    seeds = line.linelets + new_line.linelets
    return Line(pixels, seeds, line.bounds)

def grow_step(x: int, y: int, line: Line, image: Image,xtol: float, vmin: float) -> Line:
    new_line = generate_line(x, y, image, image.struct.idxs)
    vec = distance_to_line(line, new_line.bounds.x, new_line.bounds.y)
    dist = np.sqrt(np.sum(vec**2, axis=-1))
    lval = image.data.get(int(np.round(new_line.pixels.x0)),
                          int(np.round(new_line.pixels.y0)))

    if new_line.bounds.magnitude > 0 and np.all(dist < xtol) and lval > vmin:
        return add_line(line, new_line)
    return line

def region_grow(x: int, y: int, image: Image, max_iter: int, xtol: float, vmin: float, lookahead: int=1) -> Line:
    line = generate_line(x, y, image, image.struct.idxs)

    for _ in range(max_iter):
        x0, y0 = int(np.round(line.bounds.x0)), int(np.round(line.bounds.y0))
        bline = grow_step(x0, y0, line, image, xtol, vmin)
        for i in range(lookahead):
            if bline.linelets.size == line.linelets.size:
                x, y = (int(np.round(line.bounds.x0 - (i + 1) * np.cos(line.bounds.theta) * image.radius)),
                        int(np.round(line.bounds.y0 - (i + 1) * np.sin(line.bounds.theta) * image.radius)))
                bline = grow_step(x, y, line, image, xtol, vmin)

        x1, y1 = int(np.round(bline.bounds.x1)), int(np.round(bline.bounds.y1))
        fline = grow_step(x1, y1, bline, image, xtol, vmin)
        for i in range(lookahead):
            if fline.linelets.size == bline.linelets.size:
                x, y = (int(np.round(bline.bounds.x1 + (i + 1) * np.cos(bline.bounds.theta) * image.radius)),
                        int(np.round(bline.bounds.y1 + (i + 1) * np.sin(bline.bounds.theta) * image.radius)))
                fline = grow_step(x, y, bline, image, xtol, vmin)

        if fline.linelets.size == line.linelets.size:
            break

        line = fline.update_bounds()

    return line

def logbinom(n: int, k: int, p: float) -> float:
    # binomial_tail(n, k, p) = sum_{i = k}^n bincoef(n, i) * p^i * (1 - p)^{n - i}
    # bincoef(n, k) = gamma(n + 1) / (gamma(k + 1) * gamma(n - k + 1))
    if n == k:
        return n * np.log(p)

    # term_i = bincoef(n, i) * p^i * (1 - p)^(n - i)
    # term_i / term_{i - 1} = [(n - i + 1) / i] * [p / (1 - p)]

    log_term = loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1) \
             + k * np.log(p) + (n - k) * np.log(1 - p)
    term = np.exp(log_term)
    bin_tail = term
    p_term = p / (1 - p)

    for i in range(k + 1, n + 1):
        term *= (n - i + 1) / i * p_term
        bin_tail += term

    return np.log(bin_tail)

def log_nfa(line: Line, xtol: float, min_size: int, p: float) -> float:
    vec = distance_to_line(line, line.linelets.x, line.linelets.y)
    dist = np.sqrt(np.sum(vec**2, axis=-1))
    k = int(np.sum(np.all(dist < xtol, axis=-1)))
    return -min_size * np.log(p) + logbinom(line.linelets.size, k, p)

@dataclass
class DetState():
    lines   : List[Line]
    indices : np.ndarray
    used    : np.ndarray

    @classmethod
    def new(cls, image: Image, vmin: float) -> DetState:
        y, x = np.unravel_index(np.argsort(image.data, axis=None)[::-1], image.shape)
        mask = image.mu[y, x] > image.struct.size * vmin
        return cls([], np.stack((x[mask], y[mask]), axis=-1), np.zeros(image.shape, dtype=bool))

    @classmethod
    def new_sparse(cls, image: Image, vmin: Optional[float]=None, axis: int=0) -> DetState:
        if vmin is None:
            bgd_lvl = image.background_level()
            vmin = np.mean(image.data[image.mu[()] > bgd_lvl * image.struct.size])

        idxs = np.arange(0, image.shape[1 - axis], image.radius)
        lines = np.take(image.data[()], idxs, axis=1 - axis)
        peaks = local_maxima(lines, axis=axis)[:, ::-1]
        peaks[:, axis] *= image.radius
        peaks = peaks[image.mu[peaks[:, 1], peaks[:, 0]] > vmin * image.struct.size]
        peaks = peaks[np.argsort(image.data[peaks[:, 1], peaks[:, 0]])[::-1]]
        return cls([], peaks, np.zeros(image.shape, dtype=bool))

    @property
    def size(self) -> int:
        return self.indices.shape[0]

    @property
    def x(self) -> np.ndarray:
        return self.indices[..., 0]

    @property
    def y(self) -> np.ndarray:
        return self.indices[..., 1]

    def __iter__(self) -> DetState:
        return self

    def __next__(self) -> np.ndarray:
        if self.size:
            return self.indices[0]
        raise StopIteration

    def update(self, x: np.ndarray, y: np.ndarray):
        self.used[y, x] = True
        self.indices = self.indices[np.invert(self.used[self.y, self.x])]

def find_streaks(image: Image, state: DetState, xtol: float, log_eps: float=np.log(1e-1), n_grow: int=100,
                 lookahead: int=1, min_size: int=5) -> DetState:
    bgd_lvl = image.background_level()
    with tqdm(state, desc="Detecting lines",
              bar_format='{desc}: {n_fmt} checked{postfix} [{elapsed}, {rate_fmt}]') as pbar:
        for x, y in pbar:
            pbar.set_postfix_str(f"{len(state.lines)} detected")
            line = region_grow(x, y, image, n_grow, xtol, bgd_lvl, lookahead)
            x, y, val = draw_line_table(line.bounds.to_line(2 * image.radius + 1),
                                        image.shape, profile='tophat')
            state.update(x[val > 0.0], y[val > 0.0])
            if log_nfa(line, xtol, min_size, xtol / (image.radius + 0.5)) < log_eps:
                shrinked = line.shrink(1.0)
                if shrinked.pixels.moments.mu > bgd_lvl * shrinked.pixels.size:
                    image.update(x, y, val)
                    state.lines.append(line)
    return state
