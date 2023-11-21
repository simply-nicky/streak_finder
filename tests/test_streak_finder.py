from typing import Dict, Union
import pytest
import numpy as np
from streak_finder import Structure, Image, DetState, find_streaks
from streak_finder.src import draw_line_image

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
    return Structure.new(**request.param)

@pytest.fixture(params=[{'Y': 500, 'X': 500, 'n_lines': 100, 'length': 50, 'width': 3.0},],
                scope='session')
def image(request: pytest.FixtureRequest, structure: Structure) -> Image:
    return Image.new(generate_image(**request.param), structure)

@pytest.fixture(params=[{'axis': 0}, {'axis': 1}],
                scope='session')
def state(request: pytest.FixtureRequest, image: Image) -> DetState:
    return DetState.new_sparse(image, axis=request.param['axis'])

def test_state(state: DetState):
    assert state.size > 0

@pytest.fixture(params=[{'xtol': 1.1, 'lookahead': 2, 'min_size': 3}],
                scope='session')
def parameters(request: pytest.FixtureRequest) -> Dict[str, Union[float, int]]:
    return dict(xtol=request.param['xtol'], lookahead=request.param['lookahead'],
                min_size=request.param['min_size'])

def test_find_streaks(image: Image, state: DetState, parameters: Dict[str, Union[float, int]]):
    state = find_streaks(image, state, **parameters)
    assert len(state.lines) > 0
    lines = np.array([[line.bounds.x0, line.bounds.y0, line.bounds.x1, line.bounds.y1]
                      for line in state.lines])
    assert np.all(np.min(lines, axis=0) >= 0)
    assert np.all(np.max(lines[:, ::2], axis=0) < image.shape[1])
    assert np.all(np.max(lines[:, 1::2], axis=0) < image.shape[0])
