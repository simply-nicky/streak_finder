from typing import Callable, List, Literal, Tuple, Type, TypeVar, overload
from dataclasses import dataclass, field
from .annotations import (ArrayNamespace, BoolArray, NDArray, NDRealArray, NumPy, ReadOut,
                          RealArray, ROI)
from .data_container import ArrayContainer, Container, array_namespace
from .data_processing import CrystData, RegionDetector, StreakDetector, Streaks, Peaks
from .label import Structure2D, Structure3D, Regions2D
from .parser import INIParser, JSONParser, Parser
from .streak_finder import StreakFinderResult

P = TypeVar("P", bound='BaseParameters')
Detected = StreakFinderResult | List[StreakFinderResult]

class BaseParameters(Container):
    @classmethod
    def parser(cls, ext: str='ini') -> Parser:
        if ext == 'ini':
            return INIParser.from_container(cls, default='parameters')
        if ext == 'json':
            return JSONParser.from_container(cls, default='parameters')
        raise ValueError(f"Invalid format: {ext}")

    @classmethod
    def read(cls: Type[P], file: str, ext: str='ini') -> P:
        return cls.from_dict(**cls.parser(ext).read(file))

@dataclass
class ROIParameters(BaseParameters):
    xmin    : int = 0
    xmax    : int = 0
    ymin    : int = 0
    ymax    : int = 0

    def to_roi(self) -> ROI:
        return (self.ymin, self.ymax, self.xmin, self.xmax)

    def size(self) -> int:
        return max((self.ymax - self.ymin) * (self.xmax - self.xmin), 0)

@dataclass
class StructureParameters(BaseParameters):
    radius          : int
    rank            : int

    @overload
    def to_structure(self, kind: Literal['2d']) -> Structure2D: ...

    @overload
    def to_structure(self, kind: Literal['3d']) -> Structure3D: ...

    def to_structure(self, kind: Literal['2d', '3d']) -> Structure2D | Structure3D:
        if kind == '2d':
            return Structure2D(self.radius, self.rank)
        if kind == '3d':
            return Structure3D(self.radius, self.rank)
        raise ValueError(f"Invalid kind keyword: {kind}")

@dataclass
class CrystMetadata(ArrayContainer):
    mask        : BoolArray
    std         : RealArray
    whitefield  : RealArray

@dataclass
class MaskParameters(BaseParameters):
    method  : Literal['all-bad', 'no-bad', 'range', 'snr', 'std']
    vmin    : int = 0
    vmax    : int = 65535
    snr_max : float = 3.0
    std_min : float = 0.0

@dataclass
class BackgroundParameters(BaseParameters):
    method  : Literal['mean-poisson', 'median-poisson', 'robust-mean-scale', 'robust-mean-poisson']
    r0      : float = 0.0
    r1      : float = 0.5
    n_iter  : int = 12
    lm      : float = 9.0

@dataclass
class MetadataParameters(BaseParameters):
    mask        : MaskParameters
    background  : BackgroundParameters
    roi         : ROIParameters = field(default_factory=ROIParameters)
    num_threads : int = 1

def create_metadata(frames: NDRealArray, params: MetadataParameters) -> CrystMetadata:
    xp = array_namespace(frames)
    data = CrystData(data=frames)

    if params.mask.method == 'all-bad':
        if params.roi.size() == 0:
            raise ValueError("No ROI is provided")
        data = data.update_mask(method='all-bad', roi=params.roi.to_roi())

    if params.mask.method == 'range':
        data = data.update_mask(method='range', vmin=params.mask.vmin, vmax=params.mask.vmax)

    if params.background.method == 'mean-poisson':
        data.whitefield = xp.mean(data.data * data.mask, axis=0)
        data = data.update_std(method='poisson')

    if params.background.method == 'median-poisson':
        data.whitefield = xp.median(data.data * data.mask, axis=0)
        data = data.update_std(method='poisson')

    if params.background.method == 'robust-mean-scale':
        data = data.update_whitefield(method='robust-mean-scale', r0=params.background.r0,
                                      r1=params.background.r1, n_iter=params.background.n_iter,
                                      lm=params.background.lm, num_threads=params.num_threads)

    if params.background.method == 'robust-mean-poisson':
        data = data.update_whitefield(method='robust-mean', r0=params.background.r0,
                                      r1=params.background.r1, n_iter=params.background.n_iter,
                                      lm=params.background.lm, num_threads=params.num_threads)
        data = data.update_std(method='poisson')

    if params.mask.method == 'snr':
        data = data.update_mask(method='snr', snr_max=params.mask.snr_max)

    if params.mask.method == 'std':
        data = data.import_mask(data.std > params.mask.std_min)

    return CrystMetadata(data.mask, data.std, data.whitefield)

@dataclass
class RegionParameters(Container):
    structure   : StructureParameters
    vmin        : float
    npts        : int

@dataclass
class RegionFinderParameters(BaseParameters):
    regions     : RegionParameters
    num_threads : int

def find_regions(frames: NDArray, metadata: CrystMetadata, params: RegionFinderParameters
                 ) -> Tuple[RegionDetector, List[Regions2D]]:
    if frames.ndim < 2:
        raise ValueError("Frame array must be at least 2 dimensional")
    data = CrystData(data=frames.reshape((-1,) + frames.shape[-2:]), mask=metadata.mask,
                     std=metadata.std, whitefield=metadata.whitefield)
    data = data.scale_whitefield(method='median', num_threads=params.num_threads)
    data = data.update_snr()
    det_obj = data.region_detector(params.regions.structure.to_structure('2d'))
    regions = det_obj.detect_regions(params.regions.vmin, params.regions.npts, params.num_threads)
    return det_obj, regions

@dataclass
class PatternRecognitionParameters(RegionFinderParameters):
    threshold   : float

def pattern_recognition(metadata: CrystMetadata, params: PatternRecognitionParameters,
                        xp: ArrayNamespace=NumPy) -> Callable[[NDArray], ReadOut]:
    def pattern_goodness(frames: NDArray) -> Tuple[float, float]:
        det_obj, regions = find_regions(frames, metadata, params)
        masses = det_obj.total_mass(regions)[0]
        fits = det_obj.ellipse_fit(regions)[0]
        if fits.size:
            values = xp.tanh((fits[:, 0] / fits[:, 1] - params.threshold)) * masses
            positive, negative = xp.sum(values[values > 0]), -xp.sum(values[values < 0])
            return (float(positive), float(negative))
        return (0.0, 0.0)

    return pattern_goodness

@dataclass
class StreakParameters(Container):
    structure   : StructureParameters
    xtol        : float
    vmin        : float
    min_size    : int
    nfa         : int

@dataclass
class StreakFinderParameters(BaseParameters):
    peaks               : RegionParameters
    streaks             : StreakParameters
    center              : Tuple[float, float] | None = None
    roi                 : ROIParameters = field(default_factory=ROIParameters)
    scale_whitefield    : bool = False
    num_threads         : int = 1

def find_streaks(frames: NDArray, metadata: CrystMetadata, params: StreakFinderParameters
                 ) -> Tuple[Streaks, Detected, List[Peaks], StreakDetector]:
    if frames.ndim < 2:
        raise ValueError("Frame array must be at least 2 dimensional")
    data = CrystData(data=frames.reshape((-1,) + frames.shape[-2:]), mask=metadata.mask,
                     std=metadata.std, whitefield=metadata.whitefield)
    if params.roi.size():
        data = data.crop(params.roi.to_roi())
    if params.scale_whitefield:
        data = data.scale_whitefield(method='median', num_threads=params.num_threads)
    data = data.update_snr()
    det_obj = data.streak_detector(params.streaks.structure.to_structure('2d'))
    peaks = det_obj.detect_peaks(params.peaks.vmin, params.peaks.npts,
                                 params.peaks.structure.to_structure('2d'), params.num_threads)
    detected = det_obj.detect_streaks(peaks, params.streaks.xtol, params.streaks.vmin,
                                      params.streaks.min_size, nfa=params.streaks.nfa,
                                      num_threads=params.num_threads)
    streaks = det_obj.to_streaks(detected)
    if params.center is not None:
        mask = streaks.concentric_only(params.center[0], params.center[1])
        streaks = streaks[mask]
    return streaks, detected, peaks, det_obj
