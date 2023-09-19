import numpy
import math
from PIL import Image

## Using YCoCg

class ImageCompression:
    def __init__(self, pil_image):
        self.pil_image = pil_image

    def compress(self):
        img_data = numpy.array(self.pil_img)
        ## convert to YCoCg
        m = [[1/4, 1/2, 1/4],
             [1/2, 0, -1/2],
             [-1/4, 1/2, -1/4]]
        
        yuv = numpy.dot(numpy.divide(img_data, 255), m)

def test():
    i = ImageCompression()
    