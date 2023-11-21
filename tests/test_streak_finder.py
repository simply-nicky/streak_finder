from copy import deepcopy
import pytest
import numpy as np
from streak_finder import Structure, Image, DetState, find_streaks
from streak_finder.src import draw_line_image

def generate_image(Y: int, X: int, n_lines: int, width: float) -> np.ndarray:
    lines = np.random.rand(n_lines, 4)
    lines[:, ::2] *= X
    lines[:, 1::2] *= Y
    lines = np.concatenate((lines, width * np.ones((n_lines, 1))))
    return draw_line_image(lines, (Y, X))

@pytest.fixture(params=[{'radius': 3, 'rank': 2}, {'radius': 4, 'rank': 3}],
                scope='session')
def structure(request: pytest.FixtureRequest) -> Structure:
    return Structure.new(**request.param)

@pytest.fixture(params=[{'Y': 500, 'X': 500, 'n_lines': 100, 'width': 3.0},],
                scope='session')
def image(request: pytest.FixtureRequest, structure: Structure) -> Image:
    return Image.new(generate_image(**request.param), structure)

@pytest.fixture(params=[{'axis': 0}, {'axis': 1}],
                scope='session')
def state(request: pytest.FixtureRequest, images: np.ndarray) -> DetState:
    return DetState.new_sparse(images, axis=request.param['axis'])

def test_find_streaks(image: Image, state: DetState, xtol: float, lookahead: int, min_size: int):
    state = find_streaks(deepcopy(image), state, xtol=xtol, lookahead=lookahead, min_size=min_size)
    lines = np.stack([line.bounds.to_line() for line in state.lines])
    assert len(state.lines) > 0
    assert np.min(lines, axis=0) >= 0
    assert np.all(np.max(lines[:, ::2], axis=0) < image.shape[1])
    assert np.all(np.max(lines[:, 1::2], axis=0) < image.shape[0])
