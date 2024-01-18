from typing import Dict, List, Tuple, Union
import pytest
import numpy as np
from streak_finder import Pattern
from streak_finder.src import draw_line_image, median_filter, robust_mean, robust_lsq, Peaks, Structure

Line = Tuple[float, float, float, float]

def generate_image(Y: int, X: int, n_lines: int, length: float, width: float) -> np.ndarray:
    lengths = length * np.random.rand(n_lines)
    thetas = 2 * np.pi * np.random.rand(n_lines)
    x0, y0 = np.array([[X], [Y]]) * np.random.rand(2, n_lines)
    lines = np.stack((x0 - 0.5 * lengths * np.cos(thetas),
                      y0 - 0.5 * lengths * np.sin(thetas),
                      x0 + 0.5 * lengths * np.cos(thetas),
                      y0 + 0.5 * lengths * np.sin(thetas),
                      width * np.ones(n_lines)), axis=1)
    return draw_line_image(lines, (Y, X))

@pytest.fixture(params=[{'radius': 3, 'rank': 2}, {'radius': 4, 'rank': 3}],
                scope='session')
def structure(request: pytest.FixtureRequest) -> Structure:
    return Structure(**request.param)

@pytest.fixture(params=[{'Y': 500, 'X': 500, 'n_lines': 100, 'length': 50, 'width': 3.0},],
                scope='session')
def pattern(request: pytest.FixtureRequest, structure: Structure) -> Pattern:
    data = generate_image(**request.param)
    mask = np.ones(data.shape, dtype=bool)
    return Pattern(data, mask, structure)

@pytest.fixture(params=[{'vmin': 0.2, 'npts': 5}],
                scope='session')
def peaks(request: pytest.FixtureRequest, pattern: Pattern) -> Peaks:
    return pattern.find_peaks(**request.param)

@pytest.mark.streak_finder
def test_find_peaks(pattern: Pattern, peaks: Peaks):
    assert peaks.size > 0
    assert np.min(peaks.x, axis=0) >= 0 and np.min(peaks.y, axis=0) >= 0
    assert np.max(peaks.x, axis=0) < pattern.shape[1]
    assert np.max(peaks.y, axis=0) < pattern.shape[0]

@pytest.fixture(params=[{'vmin': 0.2, 'xtol': 1.8, 'lookahead': 1, 'min_size': 1}],
                scope='session')
def lines(request: pytest.FixtureRequest, pattern: Pattern, peaks: Peaks) -> List[Line]:
    return pattern.find_streaks(peaks, **request.param)

@pytest.mark.streak_finder
def test_find_streaks(pattern: Pattern, lines: List[Line]):
    assert len(lines) > 0
    assert np.all(np.min(lines, axis=0) >= -1)
    assert np.all(np.max(np.array(lines)[:, ::2], axis=0) < pattern.shape[1])
    assert np.all(np.max(np.array(lines)[:, 1::2], axis=0) < pattern.shape[0])

@pytest.fixture(params=[{'shape': (10, 10, 10), 'shape': (20, 20)}],
                scope='session')
def shape(request: pytest.FixtureRequest) -> Tuple[int, ...]:
    return request.param['shape']

@pytest.mark.src
def test_median_filter(shape: Tuple[int, ...]):
    inp = np.random.rand(*shape)
    mask = np.random.randint(0, 1, shape).astype(bool)
    out = median_filter(inp, size=len(shape) * [3,], mask=mask)
    assert np.all(np.array(inp.shape) == np.array(out.shape))

@pytest.mark.src
def test_robust_mean(shape: Tuple[int, ...]):
    inp = np.random.rand(*shape)
    mask = np.random.randint(0, 1, shape).astype(bool)
    out = robust_mean(inp, mask=mask, axis=0)
    assert np.all(np.array(inp.shape[1:]) == np.array(out.shape))

@pytest.mark.src
def test_robust_lsq(shape: Tuple[int, ...]):
    y = np.random.rand(*shape)
    W = np.random.rand(*shape[1:])
    x = robust_lsq(W=W, y=y, axis=np.arange(1, len(shape)))
    assert np.all(np.array(x.shape) == np.array([shape[0], 1]))
