import math
import numpy
import struct
import zlib
import cv2
import scipy.fft._pocketfft.pypocketfft as pfft
from PIL import Image
from io import BytesIO
from scipy.fft._pocketfft.helper import (_asfarray, _init_nd_shape_and_axes, _datacopied,
                     _fix_shape, _fix_shape_1d, _normalization, _workers)

FILE_SIG = b"MIMG"

DCT_QUANT_DEFAULT = \
    numpy.array([[16,  11,  10,  16,  24,  40,  51,  61],
                 [12,  12,  14,  19,  26,  58,  60,  55],
                 [14,  13,  16,  24,  40,  57,  69,  56],
                 [14,  17,  22,  29,  51,  87,  80,  62],
                 [18,  22,  37,  56,  68, 109, 103,  77],
                 [24,  35,  55,  64,  81, 104, 113,  92],
                 [49,  64,  78,  87, 103, 121, 120, 101],
                 [72,  92,  95,  98, 112, 100, 103,  99]])

ENTROPY_ZIGZAG_INVERSE = \
    numpy.array([[0,  1,  5,  6,  14, 15, 27, 28],
                 [2,  4,  7,  13, 16, 26, 29, 42],
                 [3,  8,  12, 17, 25, 30, 41, 43],
                 [9,  11, 18, 24, 31, 40, 44, 53],
                 [10, 19, 23, 32, 39, 45, 52, 54],
                 [20, 22, 33, 38, 46, 51, 55, 60],
                 [21, 34, 37, 47, 50, 56, 59, 61],
                 [35, 36, 48, 49, 57, 58, 62, 63]]).flatten()

ENTROPY_ZIGZAG = numpy.argsort(ENTROPY_ZIGZAG_INVERSE)

## 
## RGB -> YCoCg
##

## R(0, 255) -> Y(0, 1)
## G(0, 255) -> U(-0.5, 0.5)
## B(0, 255) -> V(-0.5, 0.5)

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

##
## DCT
##

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

class DCTCompression:
    def __init__(self, tile_size=8, dct_quant=None):
        self.dct_quant = dct_quant if not dct_quant is None else DCT_QUANT_DEFAULT
        self.tile_size = tile_size

    def compress_channels(self, f, channels):
        quant_data = self.dct_quant.astype(numpy.int32).tobytes()
        tile_size = self.tile_size

        f.write(struct.pack("HHI", self.tile_size, len(quant_data), len(channels)))
        f.write(quant_data)

        for c,channel in enumerate(channels):
            chan_w, chan_h = channel.shape
            f.write(struct.pack("II", chan_w, chan_h))

            compressor = zlib.compressobj()
            zlib_header = b""

            num_tiles = 0
            for j in range(math.ceil(chan_h/tile_size)):
                for i in range(math.ceil(chan_w/tile_size)):
                    tile = numpy.zeros((tile_size, tile_size), dtype=numpy.float32)

                    pixels = channel[i*tile_size : (i+1)*tile_size, j*tile_size : (j+1)*tile_size]
                    tile[0:pixels.shape[0], 0:pixels.shape[1]] = pixels

                    tile = img_dct(img_dct(tile.T).T)

                    tile /= self.dct_quant
                    tile = numpy.round(tile, decimals=0).astype(numpy.uint8)

                    tile = tile.flatten()[ENTROPY_ZIGZAG]
                    
                    tile_length = 0
                    for k in range((tile_size*tile_size)-1, -1, -1):
                        if tile[k] != 0:
                            tile_length = k+1
                            break

                    zlib_header += compressor.compress(bytes([tile_length]))
                    zlib_header += compressor.compress(tile[0:tile_length])
                    num_tiles += 1

            compressed_channel = zlib_header + compressor.flush()

            f.write(struct.pack("II", num_tiles, len(compressed_channel)))
            f.write(compressed_channel)

    @staticmethod
    def decompress_channels(f):
        tile_size, quant_data_len, num_channels = struct.unpack("HHI", f.read(8))
        quant_data = f.read(quant_data_len)
        dct_quant = numpy.frombuffer(quant_data, dtype=numpy.int32).reshape(tile_size, tile_size)

        output_channels = []
        for c in range(num_channels):
            chan_w, chan_h, num_tiles, compressed_channel_length = struct.unpack("IIII", f.read(16))

            output_channel = numpy.zeros((chan_w + tile_size, chan_h + tile_size), dtype=numpy.single)

            channel = BytesIO(zlib.decompress(f.read(compressed_channel_length)))

            i = 0
            j = 0
            for t in range(num_tiles):
                tile_length, = channel.read(1)

                compressed_tile = channel.read(tile_length)
                tile = numpy.zeros(tile_size*tile_size)
                tile[0:tile_length] = numpy.frombuffer(compressed_tile, numpy.int8)
                tile = tile[ENTROPY_ZIGZAG_INVERSE].reshape(tile_size, tile_size)
                
                tile = tile*dct_quant
                tile = img_idct(img_idct(tile.T).T)

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
        self.dct = DCTCompression(dct_quant=dct_quant)

    @staticmethod
    def decompress(f):
        f.seek(0)

        if f.read(len(FILE_SIG)) != FILE_SIG:
            raise Exception("Signature mismatch!")

        img_w, img_h = struct.unpack("II", f.read(8))

        channels = DCTCompression.decompress_channels(f)
        for i, channel in enumerate(channels):
            channel /= (128 if i == 0 else 256)

        y,u,v = channels
        u = resize_channel(u, img_h, img_w)
        v = resize_channel(v, img_h, img_w)

        yuv = numpy.dstack((y, u, v))
        pixels = YCOCG_RGB(yuv).astype(numpy.uint8)

        return Image.fromarray(pixels)
        
    def compress(self, f, pil_image):
        img_w, img_h = pil_image.size

        f.write(FILE_SIG + struct.pack("II", img_w, img_h))

        yuv = RGB_YCOCG(numpy.array(pil_image))

        chroma_size = math.floor(img_w*self.chroma_scale[0]), math.floor(img_h*self.chroma_scale[1])

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
    def _from_file(f):
        pil_image = ImageCompression.decompress(f)
        return MImg(pil_image)
    
    @staticmethod
    def from_buffer(buf):
        return MImg._from_file(BytesIO(buf))
    
    
    @staticmethod
    def open(filename):
        with open(filename, "rb") as f:
            return MImg._from_file(f)

    def save(self, filename):
        with open(filename, "wb") as f:
            self.compression.compress(f, self._image)

if __name__ == "__main__":
    image = MImg.from_image("test-images/photo-01.png")
    image.save("test-images-output/photo-01.buf")
    
    image2 = MImg.open("test-images-output/photo-01.buf")
    image2._image.show()