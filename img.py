import math
import numpy
import struct
import os
import zlib
import cv2
from PIL import Image
from scipy.fftpack import dct, idct
from io import BytesIO

FILE_SIG = b"MIMG"
FILE_SIG_L = 4

DCT_QUANT_DEFAULT = \
    numpy.array([[16,  11,  10,  16,  24,  40,  51,  61],
                 [12,  12,  14,  19,  26,  58,  60,  55],
                 [14,  13,  16,  24,  40,  57,  69,  56],
                 [14,  17,  22,  29,  51,  87,  80,  62],
                 [18,  22,  37,  56,  68, 109, 103,  77],
                 [24,  35,  55,  64,  81, 104, 113,  92],
                 [49,  64,  78,  87, 103, 121, 120, 101],
                 [72,  92,  95,  98, 112, 100, 103,  99]])

DCT_QUANT_LOW = \
    numpy.array([[2,  9,  11, 12, 14, 16, 21, 25],
                 [9,  11, 12, 14, 16, 20, 24, 28],
                 [11, 12, 14, 16, 19, 23, 26, 43],
                 [12, 14, 16, 18, 22, 22, 21, 62],
                 [14, 16, 18, 21, 23, 21, 42, 77],
                 [16, 19, 22, 25, 21, 38, 76, 86],
                 [20, 23, 25, 37, 38, 77, 82, 95],
                 [24, 28, 46, 62, 75, 88, 98, 101]])

def RGB_YCOCG(pixels):
    m = [[1/4, 1/2, -1/4],
         [1/2, 0, 1/2],
         [1/4, -1/2, -1/4]]
     
    return numpy.dot(pixels/255, numpy.asarray(m))

def YCOCG_RGB(pixels):
    m = [[1, 1, 1],
         [1, 0, -1],
         [-1, 1, -1]]

    rgb = numpy.dot(pixels, numpy.asarray(m))*255
    rgb = numpy.clip(rgb, 0, 255)
    return numpy.rint(rgb).astype(numpy.uint8)

def resize_channel(pixels, w, h):
    return cv2.resize(pixels, (h, w), interpolation=cv2.INTER_AREA) 

class DCTCompression:
    def __init__(self, tile_size=8, dct_quant=None):
        self.dct_quant = dct_quant if not dct_quant is None else DCT_QUANT_DEFAULT
        self.tile_size = 8

    def compress_channels(self, f, channels):
        quant_data = self.dct_quant.astype(numpy.int32).tobytes()
        tile_size = self.tile_size

        f.write(struct.pack("HHI", self.tile_size, len(quant_data), len(channels)))
        f.write(quant_data)

        for channel in channels:
            chan_w, chan_h = channel.shape
            f.write(struct.pack("II", chan_w, chan_h))

            num_tiles_pos = f.tell()
            f.write(struct.pack("I", 0))

            k = 0
            for j in range(math.ceil(chan_h/tile_size)):
                for i in range(math.ceil(chan_w/tile_size)):
                    tile = numpy.zeros((tile_size, tile_size))

                    pixels = channel[i*tile_size : (i+1)*tile_size, j*tile_size : (j+1)*tile_size]
                    tile[0:pixels.shape[0], 0:pixels.shape[1]] = pixels

                    tile = dct(dct(tile.T, norm='ortho').T, norm='ortho')
                    tile /= self.dct_quant
                    tile = numpy.around(tile, 0)

                    compressed_tile = zlib.compress(tile.astype(numpy.int8).tobytes())
                    f.write(struct.pack("H", len(compressed_tile)))
                    f.write(compressed_tile)

                    k += 1

            f.seek(num_tiles_pos, 0)
            f.write(struct.pack("I", k))
            f.seek(0, 2)

    @staticmethod
    def decompress_channels(f):
        tile_size, quant_data_len, num_channels = struct.unpack("HHI", f.read(8))
        quant_data = f.read(quant_data_len)
        dct_quant = numpy.frombuffer(quant_data, dtype=numpy.int32).reshape(tile_size, tile_size)

        output_channels = []
        for c in range(num_channels):
            chan_w, chan_h, num_tiles = struct.unpack("III", f.read(12))
            print(c, chan_w, chan_h, num_tiles)

            output_channel = numpy.zeros((chan_w + tile_size, chan_h + tile_size))

            i = 0
            j = 0
            for t in range(num_tiles):
                tile_length, = struct.unpack("H", f.read(2))

                compressed_tile = f.read(tile_length)
                tile = numpy.frombuffer(zlib.decompress(compressed_tile), numpy.int8).reshape(tile_size, tile_size)
                
                tile = tile*dct_quant
                tile = idct(idct(tile.T, norm='ortho').T, norm='ortho')

                output_channel[i:i+tile_size, j:j+tile_size] = tile

                i += tile_size
                if i >= chan_w:
                    i = 0
                    j += tile_size

            output_channels.append(output_channel[0:chan_w, 0:chan_h])

        return output_channels

class ImageCompression:
    def __init__(self, chroma_scale, dct_quant):
        self.chroma_scale = chroma_scale
        self.dct = DCTCompression(dct_quant)

    @staticmethod
    def decompress(f):
        f.seek(0)

        if f.read(FILE_SIG_L) != FILE_SIG:
            raise Exception("Signature mismatch!")

        img_w, img_h = struct.unpack("II", f.read(8))

        channels = DCTCompression.decompress_channels(f)
        for i, channel in enumerate(channels):
            channel /= (128 if i == 0 else 256)

        y,u,v = channels
        u = resize_channel(u, img_h, img_w)
        v = resize_channel(v, img_h, img_w)

        print(y.shape, u.shape, v.shape)

        yuv = numpy.dstack((y, u, v))
        print(yuv.shape)
        pixels = YCOCG_RGB(yuv).astype(numpy.uint8)

        return Image.fromarray(pixels)
        
    def compress(self, f, pil_image):
        img_w, img_h = pil_image.size

        f.write(FILE_SIG + struct.pack("II", img_w, img_h))

        yuv = RGB_YCOCG(numpy.array(pil_image))

        chroma_size = (math.floor(img_w*self.chroma_scale[0]), math.floor(img_h*self.chroma_scale[1]))

        y,u,v = numpy.dsplit(yuv, 3)
        y = y.reshape(img_h, img_w)
        u = resize_channel(u[:,:,0], *chroma_size)
        v = resize_channel(v[:,:,0], *chroma_size)

        channels = [y, u, v]

        for i, channel in enumerate(channels):
            channel *= (128 if i == 0 else 256)
            channel = numpy.floor(numpy.clip(channel, -128, 127))

            channels[i] = channel

        self.dct.compress_channels(f, channels)

class MImg:
    def __init__(self, pil_img, chroma_scale=(0.5, 0.5), dct_quant=None):
        self._image = pil_img
        self.compression = ImageCompression(chroma_scale, dct_quant)

    @staticmethod
    def from_image(filename):
        pil_img = Image.open(filename).convert("RGB")
        return MImg(pil_img)

    @staticmethod
    def from_buffer(buf):
        return MImg.load(BytesIO(buf))

    @staticmethod
    def from_file(filename):
        return MImg.load(open(filename, "rb"))

    @staticmethod
    def load(f): 
        pil_image = ImageCompression.decompress(f)
        f.close()

        return MImg(pil_image)

    def save(self, f):
        self.compression.compress(f, self._image)
        f.close()

    def to_file(self, filename):
        self.save(open(filename, "wb"))

def test_compress_images():
    input_dir = "test-images"
    output_dir = "test-images-output"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for file in os.listdir(input_dir):
        print(file)
        name,ext = os.path.splitext(file)

        output_filename = os.path.join(output_dir, name+".buf")

        img = MImg.from_image(os.path.join(input_dir, file))
        img.to_file(output_filename)

        img2 = MImg.from_file(output_filename)
        img2._image.save(os.path.join(output_dir, file))

if __name__ == "__main__":
    test_compress_images()