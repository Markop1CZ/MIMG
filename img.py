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

def rle_encode(array):
    t = array[0]
    c = 0

    out = []
    for i in range(len(array)):
        if array[i] != t:
            out.extend([c, t])
            t = array[i]
            c = 0

        c += 1

    out.extend([c, t])

    return numpy.asarray(out)

def rle_decode(array):
    if len(array) %2 != 0:
        raise Exception("Not RLE!")

    out = []
    for i in range(math.floor(len(array)/2)):
        idx = i*2
        cnt = array[idx]
        val = array[idx+1]

        out.extend([val]*cnt)

    return numpy.asarray(out)

def resize_channel(pixels, w, h):
    return cv2.resize(pixels, (h, w), interpolation=cv2.INTER_AREA) 

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
                    tile = numpy.zeros((tile_size, tile_size))

                    pixels = channel[i*tile_size : (i+1)*tile_size, j*tile_size : (j+1)*tile_size]
                    tile[0:pixels.shape[0], 0:pixels.shape[1]] = pixels

                    tile = dct(dct(tile.T, norm='ortho').T, norm='ortho')
                    tile /= self.dct_quant
                    tile = numpy.around(tile, 0)

                    tile = tile.flatten()[ENTROPY_ZIGZAG]

                    rle_tile = rle_encode(tile).astype(numpy.int8).tobytes()

                    zlib_header += compressor.compress(struct.pack("H", len(rle_tile)))
                    zlib_header += compressor.compress(rle_tile)
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
                tile_length, = struct.unpack("H", channel.read(2))

                compressed_tile = channel.read(tile_length)
                tile = rle_decode(numpy.frombuffer(compressed_tile, numpy.int8))
                tile = tile[ENTROPY_ZIGZAG_INVERSE].reshape(tile_size, tile_size)
                
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

def format_size(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def test_compress_images():
    input_dir = "test-images"
    output_dir = "test-images-output"

    stats = []

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("testing compression in {0}".format(input_dir))
    
    for file in os.listdir(input_dir):
        name,ext = os.path.splitext(file)

        output_filename = os.path.join(output_dir, name+".buf")

        png_size = os.stat(os.path.join(input_dir, file)).st_size
        pil_img = Image.open(os.path.join(input_dir, file)).convert("RGB")

        b = BytesIO()
        pil_img.save(b, format="jpeg")
        jpeg_size = b.getbuffer().nbytes

        try:
            t = time.time()
            img = MImg(pil_img)
            img.to_file(output_filename)
            compressed_size = os.stat(output_filename).st_size
            t1 = time.time()
        except Exception as e:
            print("error compressing {0}: {1}".format(file, e))
            traceback.print_exc()
            continue

        try:
            img2 = MImg.from_file(output_filename)
            img2._image.save(os.path.join(output_dir, file))
            t2 = time.time()
        except Exception as e:
            print("error decompressing {0}: {1}".format(file, e))
            traceback.print_exc()
            continue
        
        stats.append([name, *pil_img.size, (pil_img.size[0]*pil_img.size[1]*3)/1048576, png_size/1048576, jpeg_size/1048576, compressed_size/1048576])
        print("file={0} compression={1:.1f}s decompression={2:.1f}s png={3} compressed={4} ratio={5:.2%}".format(file, t1-t, t2-t1, format_size(png_size), format_size(compressed_size), 1-(compressed_size/png_size)))
    
    f = open("{0}.csv".format(time.time()), "w")
    for s in stats:
        f.write(";".join([str(x).replace(",", ".") for x in s]))
        f.write("\n")
    f.close()

if __name__ == "__main__":
    import time
    import traceback
    test_compress_images()