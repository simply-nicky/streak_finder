from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import numpy as np
from .src import draw_line_image, draw_line_mask, draw_line_table
from .data_container import DataContainer
from .annotations import (Indices, NDIntArray, NDRealArray, Pattern, PatternWithHKL,
                          PatternWithHKLID, Shape)

@dataclass
class Streaks(DataContainer):
    """Detector streak lines container. Provides an interface to draw a pattern for a set of
    lines.

    Args:
        x0 : x coordinates of the first point of a line.
        y0 : y coordinates of the first point of a line.
        x1 : x coordinates of the second point of a line.
        y1 : y coordinates of the second point of a line.
        length: Line's length in pixels.
        h : First Miller index.
        k : Second Miller index.
        l : Third Miller index.
        hkl_id : Bragg reflection index.
    """
    x0          : NDRealArray
    y0          : NDRealArray
    x1          : NDRealArray
    y1          : NDRealArray
    idxs        : NDIntArray = field(default_factory=lambda: np.array([], dtype=int))
    length      : NDRealArray = field(default_factory=lambda: np.array([]))
    h           : Optional[NDIntArray] = field(default=None)
    k           : Optional[NDIntArray] = field(default=None)
    l           : Optional[NDIntArray] = field(default=None)
    hkl_id      : Optional[NDIntArray] = field(default=None)

    def __post_init__(self):
        if self.idxs.shape != self.x0.shape:
            self.idxs = np.zeros(self.x0.shape, dtype=int)
        if self.length.shape != self.x0.shape:
            self.length = np.sqrt((self.x1 - self.x0)**2 + (self.y1 - self.y0)**2)

    @property
    def hkl(self) -> Optional[NDIntArray]:
        if self.h is None or self.k is None or self.l is None:
            return None
        return np.stack((self.h, self.k, self.l), axis=1)

    def __len__(self) -> int:
        return self.length.shape[0]

    def mask_streaks(self, idxs: Indices) -> Streaks:
        """Return a new streaks container with a set of streaks discarded.

        Args:
            idxs : A set of indices of the streaks to discard.

        Returns:
            A new :class:`streak_finder.Streaks` container.
        """
        return Streaks(**{attr: self[attr][idxs] for attr in self.contents()})

    def pattern_dict(self, width: float, shape: Shape, kernel: str='rectangular'
                     ) -> Union[Pattern, PatternWithHKL, PatternWithHKLID]:
        """Draw a pattern in the :class:`dict` format.

        Args:
            width : Lines width in pixels.
            shape : Detector grid shape.
            kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
                are available:

                * 'biweigth' : Quartic (biweight) kernel.
                * 'gaussian' : Gaussian kernel.
                * 'parabolic' : Epanechnikov (parabolic) kernel.
                * 'rectangular' : Uniform (rectangular) kernel.
                * 'triangular' : Triangular kernel.

        Returns:
            A pattern in dictionary format.
        """
        table = draw_line_table(lines=self.to_lines(width), shape=shape,
                                idxs=self.idxs, kernel=kernel)
        ids, idxs = np.array(list(table)).T
        normalised_shape = (np.prod(shape[:-2], dtype=int),) + shape[-2:]
        frames, y, x = np.unravel_index(idxs, normalised_shape)
        vals = np.array(list(table.values()))

        if self.hkl is not None:
            h, k, l = self.hkl[ids].T

            if self.hkl_id is not None:
                return PatternWithHKLID(ids, frames, y, x, vals, h, k, l,
                                        self.hkl_id[ids])
            return PatternWithHKL(ids, frames, y, x, vals, h, k, l)
        return Pattern(ids, frames, y, x, vals)

    def pattern_image(self, width: float, shape: Tuple[int, int],
                      kernel: str='gaussian') -> NDRealArray:
        """Draw a pattern in the :class:`numpy.ndarray` format.

        Args:
            width : Lines width in pixels.
            shape : Detector grid shape.
            kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
                are available:

                * 'biweigth' : Quartic (biweight) kernel.
                * 'gaussian' : Gaussian kernel.
                * 'parabolic' : Epanechnikov (parabolic) kernel.
                * 'rectangular' : Uniform (rectangular) kernel.
                * 'triangular' : Triangular kernel.

        Returns:
            A pattern in :class:`numpy.ndarray` format.
        """
        return draw_line_image(self.to_lines(width), shape=shape, idxs=self.idxs, kernel=kernel)

    def pattern_mask(self, width: float, shape: Tuple[int, int], max_val: int=1,
                     kernel: str='rectangular') -> NDIntArray:
        """Draw a pattern mask.

        Args:
            width : Lines width in pixels.
            shape : Detector grid shape.
            max_val : Mask maximal value.
            kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
                are available:

                * 'biweigth' : Quartic (biweight) kernel.
                * 'gaussian' : Gaussian kernel.
                * 'parabolic' : Epanechnikov (parabolic) kernel.
                * 'rectangular' : Uniform (rectangular) kernel.
                * 'triangular' : Triangular kernel.

        Returns:
            A pattern mask.
        """
        return draw_line_mask(self.to_lines(width), shape=shape, idxs=self.idxs, max_val=max_val,
                              kernel=kernel)

    def to_lines(self, width: float) -> NDRealArray:
        """Export a streaks container into line parameters ``x0, y0, x1, y1, width``:

        * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
        * `width` : Line's width.

        Returns:
            An array of line parameters.
        """
        widths = width * np.ones(len(self))
        return np.stack((self.x0, self.y0, self.x1, self.y1, widths), axis=1)
