import numpy
import math
import struct
import scipy
import zlib
from PIL import Image

## Using YCoCg
def rgb_ycocg(pixels):
    m = [[1/4, 1/2, -1/4],
         [1/2, 0, 1/2],
         [1/4, -1/2, -1/4]]

    return numpy.dot(numpy.divide(pixels, 255), m)

def ycocg_rgb(pixels):
    m = [[1, 1, 1],
         [1, 0, -1],
         [-1, 1, -1]]
    
    return numpy.rint(numpy.multiply(numpy.dot(pixels, m), 255)).astype(numpy.uint8)

def subsample(pixels, vertical=0.25, horizontal=0.5):
    output = scipy.ndimage.zoom(pixels, (horizontal, vertical))

    return output
    
class ImageCompression:
    def __init__(self, pil_image):
        self.pil_image = pil_image.convert("RGB")

    def get_pixels(self):
        return numpy.array(self.pil_image)

    def compress(self):
        print("Compressing image...")
        buffer = bytearray()
        img_w,img_h = self.pil_image.size
    
        print("w={0} h={1}".format(img_w, img_h))

        buffer += struct.pack("II", img_w, img_h)
        
        yuv = rgb_ycocg(self.get_pixels())

        y,u,v = numpy.dsplit(yuv, 3)
        y = y.reshape(y.shape[0], y.shape[1])
        u = subsample(u[:,:,0])
        v = subsample(v[:,:,0])

        print("u={1} v={1}".format(y.shape, u.shape, v.shape))

        for channel in y,u,v:
            data = zlib.compress(channel.astype(numpy.single).tobytes())
            length = len(data)

            buffer += struct.pack("III", channel.shape[0], channel.shape[1], length)
            buffer += data

        return buffer, (y, u, v)

    def decompress(self, buffer):
        print("Decompressing image...")
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
    import os
    
    images_folder = "test-images"
    output_folder = "test-output"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        
    for file in os.listdir(images_folder):
        print("Converting {0}:".format(file))
        img_name = os.path.splitext(file)[0]
        
        pil_img = Image.open(os.path.join(images_folder, file))
        img = ImageCompression(pil_img)

        compressed, debug_channels = img.compress()
        for i,c in enumerate(list("yuv")):
            debug_channel = debug_channels[i]
            channel_data = numpy.empty((*debug_channel.shape, 3), dtype=numpy.single)
            channel_data[:,:,i] = debug_channel
            print(c, numpy.min(debug_channel), numpy.max(debug_channel))
            channel_pixels = ycocg_rgb(channel_data)
            channel_img = Image.fromarray(channel_pixels, mode="RGB")
            channel_img.save(os.path.join(output_folder, "{0}-{1}.png".format(img_name, c)))
        
        compressed_len = len(compressed)
        result_img = Image.fromarray(img.decompress(compressed))

        ## compare against PNG
        b = BytesIO()
        pil_img.save(b, format="png")
        png_len = b.getbuffer().nbytes

        print("-> {0} compressed={1:.2f}Mb png={2:.2f}Mb ratio={3:.2f}%".format(file, compressed_len/1048576, png_len/1048576, (compressed_len/png_len)*100))

        result_img.save(os.path.join(output_folder, img_name+"-color.png"))

## should output a single color red image
def ycocg_test():
    import os
    w,h = 256, 256
    channel_data = numpy.full((w, h, 3), 0, dtype=numpy.single)
    channel_data[:,:,0] = numpy.full((w, h), 1/4, dtype=numpy.single)
    channel_data[:,:,1] = numpy.full((w, h), 1/2, dtype=numpy.single)
    channel_data[:,:,2] = numpy.full((w, h), -1/4, dtype=numpy.single)
    channel_pixels = ycocg_rgb(channel_data)
    channel_img = Image.fromarray(channel_pixels, mode="RGB")
    channel_img.save("ycocg-test.png")
          
if __name__ == "__main__": 
    test()
    ycocg_test()
