import numpy
import math
import struct
import scipy
import zlib
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

def subsample(pixels, vertical=0.25, horizontal=0.5):
    output = scipy.ndimage.zoom(pixels, (horizontal, vertical))

    return output
    
class ImageCompression:
    def __init__(self, pil_image):
        self.pil_image = pil_image

    def get_pixels(self):
        return numpy.array(self.pil_image)

    def compress(self):
        buffer = bytearray()

        print(self.pil_image.size)

        buffer += struct.pack("II", *self.pil_image.size)
        
        yuv = rgb_ycocg(self.get_pixels())

        y,u,v = numpy.dsplit(yuv, 3)
        y = y.reshape(y.shape[0], y.shape[1])
        u = subsample(u[:,:,0])
        v = subsample(v[:,:,0])

        print(y, u, v)
        print(y.shape, u.shape, v.shape)

        for channel in y,u,v:
            data = zlib.compress(channel.astype(numpy.single).tobytes())
            length = len(data)

            buffer += struct.pack("III", channel.shape[0], channel.shape[1], length)
            buffer += data

        return buffer

    def decompress(self, buffer):
        img_h,img_w = struct.unpack("II", buffer[0:8])
        del buffer[0:8]

        channels = []
        
        for i in range(3):
            w,h,length = struct.unpack("III", buffer[0:12])
            del buffer[0:12]

            data = buffer[0:length]
            del buffer[0:length]

            channels.append(numpy.frombuffer(zlib.decompress(data), numpy.single).reshape(w,h))

        y,u,v = channels
        print(y,u,v)
        print(y.shape, u.shape, v.shape)
        u = subsample(u, img_h/h, img_w/w)
        v = subsample(v, img_h/h, img_w/w)

        pixels = numpy.empty((img_w, img_h, 3), dtype=numpy.single)
        pixels[:,:,0] = y
        pixels[:,:,1] = u
        pixels[:,:,2] = v

        pixels = ycocg_rgb(pixels)

        return pixels

def test():
    from io import BytesIO
    pil_img = Image.open("test-image.bmp") ## <--- test image path
    img = ImageCompression(pil_img)

    pix1 = img.get_pixels()
    pix2 = ycocg_rgb(rgb_ycocg(pix1))

    assert pix1.all() == pix2.all(), "yuv conversion fail!"

    compressed = img.compress()
    c_len = len(compressed)
    result = img.decompress(compressed)
    im = Image.fromarray(result)
    b = BytesIO()
    pil_img.save(b, format="jpeg")
    jpg_len = b.getbuffer().nbytes
    print(c_len, jpg_len)
    im.show()
    
if __name__ == "__main__": 
    test()
