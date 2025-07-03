# Convergent beam streak finder
Connection-based streak finding algorithm for convergent beam diffraction patterns.

## Dependencies

- [Pybind11](https://github.com/pybind/pybind11) 2.11 or later.
- [Python](https://www.python.org/) 3.7 or later (Python 2.x is **not** supported).
- [h5py](https://www.h5py.org) 2.10.0 or later.
- [NumPy](https://numpy.org) 1.19.0 or later.
- [SciPy](https://scipy.org) 1.5.2 or later.
- [tqdm](https://github.com/tqdm/tqdm) 4.66 or later.

## Installation from source
In order to build the package from source simply execute the following command:

    python setup.py install

or:

    pip install -r requirements.txt -e . -v

That builds the C++ extensions into ``/streak_finder/src``.

## Data processing pipeline
This library provides a bespoke solution to detect diffraction signal of convergent beam diffraction (CBD) patterns. A CBD pattern comprises a set of streaks. The streak detection algorithm has multiple stages:

### Creating metadata
Firstly, one needs to generate background images of scattered signal without a sample present from a series of snapshots beforehand. Namely, one needs three images, an average of the background signal (`whitefield`), a standard deviation of the background signal (`std`), and a bad pixels mask (`mask = 1 for good pixels`).

Use `streak_finder.scripts.create_metadata` to generate the background images:

```python
import streak_finder as sf
from streak_finder.scripts import MetadataParameters, MaskParameters, BackgroundParameters, create_metadata

input_file = sf.CXIStore('test.h5')
data = input_file.load('data', idxs=np.arange(0, 100, 5), processes=16)

params = MetadataParameters(MaskParameters('range', vmax=1000000),
                            BackgroundParameters('robust-mean-poisson', r0=0.25, r1=0.99, lm=10),
                            num_threads=64)
metadata = create_metadata(data, params)
```

The generated `streak_finder.CrystMetadata` can be generated once and used for later patterns.

### Streak detection
Secondly, one can generate SNR frames used in streak detection. The SNR frame is calculated as follows (see `streak_finder.CrystData.update_snr`):

```python
snr = (metadata.mask * data - metadata.whitefield) / metadata.std
```

These SNR frames are then used to find the streaks. Use `streak_finder.scripts.find_streaks` to perform the streak detection:

```python
from streak_finder.scripts import StreakParameters, RegionParameters, StructureParameters, StreakFinderParameters
from streak_finder.scripts import find_streaks

params = StreakFinderParameters(RegionParameters(StructureParameters(2, 2), 1.5, 15),
                                StreakParameters(StructureParameters(5, 4), 2.0, 1.5, 25, 1),
                                (2206, 2325), num_threads=16)
streaks, detected, peaks, det_obj = find_streaks(data[0], metadata, params)
```

The functions returns a tuple of four elements:
- The detected lines array where each line contains four coordinates `(x0, y0, x1, y1)`.
- The streak detection result object (`streak_finder.streak_finder.StreakFinderResult`). It can be used to inspect each streak in detail and to obtain a list of points belonging to each streak (see `streak_finder.streak_finder.StreakFinderResult.to_regions`).
- The set of detected peaks (`streak_finder.streak_finder.Peaks`). Used as a precursor for streak detection algorithm.
- The streak detector (`streak_finder.StreakDetector`). The class provides a high level interface to perform the streak detection.

### Photon counts
Finally, one can find photon counts for detected streaks. I provide two ways depending on what counts one wants to obtain:

1.  The `streak_finder.streak_finder.Streak` object keeps track of what pixels belong to the given streak and the total count of SNR values belonging to the streak. One can see access them as follows:

    ```python
    for streak in detected.streaks.values():
        print(streak.x, streak.y)   # List of x and y coordinates of the pixels pertaining to the streak
        print(streak.total_mass())  # Total count of SNR values of the pixels pertaining to the streak
    ```

    One can obtain an array of SNR counts for each detected streak as follows:

    ```python
    import numpy as np

    if isinstance(detected, list):
        counts = []
        for pattern in detected:
            counts.append(np.array([streak.total_mass() for streak in pattern.streaks.values()]))
        counts = np.concatenate(counts)
    else:
        counts = np.array([streak.total_mass() for streak in detected.streaks.values()])
    ```

2.  Alternatively, one can extract a list of pixels pertaining to each streak and apply them to the raw array of photon counts (see `streak_finder.label.Regions` for more info):

    ```python
    import streak_finder as sf

    if isinstance(detected, list):
        regions = [pattern.to_regions() for pattern in detected]
    else:
        regions = detected.to_regions()
    counts = sf.label.total_mass(regions, data[0])
    ```
