import numpy
import math
from PIL import Image

## Using YCoCg

def sample_array(arr, ratio):
    return numpy.interp(numpy.arange(0, len(arr), len(arr)/ratio), numpy.arange(0, len(arr)), arr)

def subsample(yuv, vertical=2, horizontal=4):
    pass
    
class ImageCompression:
    def __init__(self, pil_image):
        self.pil_image = pil_image

    def compress(self):
        img_data = numpy.array(self.pil_image)
        ## convert to YCoCg
        m = [[1/4, 1/2, 1/4],
             [1/2, 0, -1/2],
             [-1/4, 1/2, -1/4]]
        
        yuv = numpy.dot(numpy.divide(img_data, 255), m)


        ## subsample

        return yuv

def test():
    pil_img = Image.open("test-image.bmp")
    img = ImageCompression(pil_img)
    img.compress()
    
test()
