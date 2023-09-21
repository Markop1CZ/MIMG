import numpy
import math
from PIL import Image

## Using YCoCg
def rgb_ycocg(pixels):
    m = [[1/4, 1/2, 1/4],
         [1/2, 0, -1/2],
         [-1/4, 1/2, -1/4]]

    return numpy.dot(numpy.divide(pixels, 255), m)

def ycocg_rgb(pixels):
    m = [[1, 1, -1],
         [1, 0, 1],
         [1, -1, -1]]

    return numpy.rint(numpy.multiply(numpy.dot(pixels, m), 255)).astype(numpy.uint8)

def subsample(pixels, vertical=2, horizontal=4):
    pass
    
class ImageCompression:
    def __init__(self, pil_image):
        self.pil_image = pil_image

    def get_pixels(self):
        return numpy.array(self.pil_image)

    def compress(self):
        yuv = rgb_ycocg(self.get_pixels())  

        return yuv

    def decompress(self, yuv):
        pixels = ycocg_rgb(yuv)

        return pixels

def test():
    pil_img = Image.open("test-image.bmp")
    img = ImageCompression(pil_img)

    pix1 = img.get_pixels()
    pix2 = ycocg_rgb(rgb_ycocg(pix1))

    assert pix1.all() == pix2.all(), "yuv conversion fail!"
    
if __name__ == "__main__": 
    test()
