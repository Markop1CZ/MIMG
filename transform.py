import numpy
import scipy.fft._pocketfft.pypocketfft as pfft
from scipy.fft._pocketfft.helper import (_asfarray, _init_nd_shape_and_axes, _datacopied,
                     _fix_shape, _fix_shape_1d, _normalization, _workers)

num_workers = _workers(None)

def img_dct(x, type=2, n=None, axis=-1, norm="ortho"):
    tmp = x.astype(numpy.float32)
    norm = _normalization(norm, True)
    out = tmp

    return pfft.dct(tmp, type, (axis,), norm, out, num_workers, None)

def img_idct(x, type=3, n=None, axis=-1, norm="ortho"):
    tmp = x.astype(numpy.float32)
    norm = _normalization(norm, True)
    out = tmp

    return pfft.dct(tmp, type, (axis,), norm, out, num_workers, None)


if __name__ == "__main__":
    x = img_dct(numpy.array([1, 2, 3, 4]))
    print(x)
    x = img_idct(x)
    print(x)