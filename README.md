# SaliencyMapInPython

This code is a Python adaptation of the saliency map calculation from

```
@incollection{Achanta,
  author={Achanta, Radhakrishna and Estrada, Francisco and Wils, Patricia and Süsstrunk, Sabine},
  editor={Gasteratos, Antonios and Vincze, Markus and Tsotsos, John K.},
  booktitle={Computer Vision Systems},
  title={Salient Region Detection and Segmentation},
  year={2008},
  pages={66--75},
  volume={5008},
  series={Lecture Notes in Computer Science},
  publisher={Springer Berlin Heidelberg},
  isbn={978-3-540-79546-9},
  doi={10.1007/978-3-540-79547-6_7}
}
```


For more details see http://ivrgwww.epfl.ch/~achanta/SalientRegionDetection/SalientRegionDetection.html


Example files can be found at the URL given above.

Example usage:
```
$ $ python saliency.py -v input.png output.png
```

## Usage Notes
* The front end (`colorizer.py`) requires [`argparse`](https://docs.python.org/3.4/library/argparse.html) for parsing command line arguments, [`skimage`](http://scikit-image.org/) for converting the image from RGB to LAB colour space (and back again), and [`matplotlib`](http://matplotlib.org/) for loading, displaying, and saving the image.
* The back end (`colorizationSolver.py` and `colorConversion.py`) requires [`numpy`](http://www.numpy.org/) and [`scipy`](http://www.scipy.org/).

### License

Copyright 2015 Samuel Bucheli

This file is part of SaliencyMapInPython.

SaliencyMapInPython is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

SaliencyMapInPython is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SaliencyMapInPython.  If not, see <http://www.gnu.org/licenses/>.
