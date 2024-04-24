import math
import numpy
import struct
import scipy
import zlib
import time
import cv2
from PIL import Image
from scipy.fftpack import dct, idct
from io import BytesIO

## Using YCoCg
def rgb_ycocg(pixels):
    m = [[1/4, 1/2, -1/4],
         [1/2, 0, 1/2],
         [1/4, -1/2, -1/4]]

    return numpy.dot(pixels/255, m)

def ycocg_rgb(pixels):
    m = [[1, 1, 1],
         [1, 0, -1],
         [-1, 1, -1]]

    pixels = numpy.dot(pixels, m)*255
    pixels = numpy.clip(pixels, 0, 255)
    return numpy.rint(pixels).astype(numpy.uint8)

def rle_tile(array):
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

def unrle_tile(array):
    if len(array) %2 != 0:
        raise Exception("Not RLE!")

    out = []
    for i in range(math.floor(len(array)/2)):
        idx = i*2
        cnt = array[idx]
        val = array[idx+1]

        out.extend([val]*cnt)

    return numpy.asarray(out)

def zigzag_tile(tile):
    return numpy.concatenate([numpy.diagonal(tile[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-tile.shape[0], tile.shape[0])])

def compress_channel_dct(pixels, dct_quant, tile_size=8):
    t = time.time()
      
    img_h,img_w = pixels.shape

    buf = bytearray(struct.pack("H", tile_size))
    quant_data = dct_quant.astype(numpy.int32).tobytes()
    buf += struct.pack("H", len(quant_data))
    buf += quant_data
    
    for j in range(math.ceil(img_h/tile_size)):
        for i in range(math.ceil(img_w/tile_size)):
            tile = numpy.zeros((tile_size, tile_size))
            
            tmp = pixels[j*tile_size:(j+1)*tile_size, i*tile_size:(i+1)*tile_size]
            tile[0:tmp.shape[0],0:tmp.shape[1]] = tmp

            tile = dct(dct(tile.T, norm='ortho').T, norm='ortho')
            tile /= dct_quant
            tile = numpy.around(tile, 0) + 0.0

            tile = zigzag_tile(tile)
            tile = rle_tile(tile.flatten())
            
            compressed = zlib.compress(tile.astype(numpy.int8).tobytes())
            buf += struct.pack("H", len(compressed)) 
            buf += compressed
            
    print("dct compress took: {0:.2f}s".format(time.time()-t))
            
    return buf

def decompress_channel_dct(buf, img_w, img_h):
    t = time.time()
    
    tile_size, = struct.unpack("H", buf[0:2])
    offset = 2
    
    quant_len, = struct.unpack("H", buf[offset:offset+2])
    offset += 2
    dct_quant = numpy.frombuffer(buf[offset:offset+quant_len], dtype=numpy.int32).reshape(tile_size, tile_size)
    offset += quant_len

    output_pixels = numpy.zeros((img_h + tile_size, img_w + tile_size))

    i = 0
    j = 0
    while offset < len(buf):
        length, = struct.unpack("H", buf[offset:offset+2])
        offset += 2
        rle = numpy.frombuffer(zlib.decompress(buf[offset:offset+length]), numpy.int8)
        offset += length

        pixels = unrle_tile(rle).astype(numpy.int32).reshape(tile_size, tile_size)
        ## todo: unzigzag!
        ##pixels = zigzag_tile(pixels).reshape(tile_size, tile_size)

        pixels *= dct_quant
        pixels = idct(idct(pixels.T, norm='ortho').T, norm='ortho')

        output_pixels[j:j+tile_size, i:i+tile_size] = pixels
        
        i += tile_size
        if i >= img_w:
            i = 0
            j += tile_size

    print("dct decompress took: {0:.2f}s".format(time.time()-t))
            
    return output_pixels[0:img_h, 0:img_w]

## vertical=0.25, horizontal=0.5
def subsample(pixels, horizontal, vertical):
    t = time.time()
    nw = pixels.shape[0]*vertical
    nh = pixels.shape[1]*horizontal
    
    output = cv2.resize(pixels, (math.floor(nh), math.floor(nw)), interpolation=cv2.INTER_AREA) 
    ##output = scipy.ndimage.zoom(pixels, (vertical, horizontal))

    print("subsample took: {0:.2f}s".format(time.time()-t))
    
    return output

dct_quant = numpy.array([[8,  9,  11, 12, 14, 16, 21, 25],
                         [9,  11, 12, 14, 16, 20, 24, 28],
                         [11, 12, 14, 16, 19, 23, 26, 43],
                         [12, 14, 16, 18, 22, 22, 21, 62],
                         [14, 16, 18, 21, 23, 21, 42, 77],
                         [16, 19, 22, 25, 21, 38, 76, 86],
                         [20, 23, 25, 37, 38, 77, 82, 95],
                         [24, 28, 46, 62, 75, 88, 98, 101]])
    
class ImageCompression:
    def __init__(self, pil_image, subsample=(0.25, 0.5), dct_tile=8, dct_quant=dct_quant):
        self._image = pil_image.convert("RGB")
        self._subsample = subsample
        self._dct_tile = dct_tile
        self._dct_quant = dct_quant

    def get_image(self):
        return self._image

    def get_pixels(self):
        return numpy.array(self._image)

    def compress(self):
        print("Compressing image...")
        t = time.time()
        buffer = bytearray()
        img_w,img_h = self._image.size
    
        print("w={0} h={1}".format(img_w, img_h))

        buffer += struct.pack("II", img_w, img_h)
        
        yuv = rgb_ycocg(self.get_pixels())

        y,u,v = numpy.dsplit(yuv, 3)
        y = y.reshape(y.shape[0], y.shape[1])
        u = subsample(u[:,:,0], *self._subsample)
        v = subsample(v[:,:,0], *self._subsample)

        for i,channel in enumerate((y,u,v)):
            if i == 0: ## y channel range: -128 127
                channel_data = numpy.rint(numpy.clip(numpy.multiply(channel, 128), -128, 127)).astype(numpy.int8)
            else: ## u v channel range: -128 127
                channel_data = numpy.rint(numpy.clip(numpy.multiply(channel, 256), -128, 127)).astype(numpy.int8)
                
            chan_h,chan_w = channel.shape
            compressed_data = compress_channel_dct(channel_data, self._dct_quant)
            length = len(compressed_data)

            buffer += struct.pack("III", chan_w, chan_h, length)
            buffer += compressed_data

        print("compressing took: {0:.2f}s".format(time.time()-t))
        return buffer, (y, u, v)

    def decompress(self, buffer):
        print("Decompressing image...")
        offset = 0
        img_w,img_h = struct.unpack("II", buffer[offset:offset+8])
        offset += 8

        channels = []
        
        for i in range(3):
            w,h,length = struct.unpack("III", buffer[offset:offset+12])
            offset += 12

            data = buffer[offset:offset+length]
            offset += length

            channel = decompress_channel_dct(data, w, h)
            if i == 0:
                channel = numpy.divide(channel, 128)
            else:
                channel = numpy.divide(channel, 256)

            channels.append(channel)

        y,u,v = channels
        u = subsample(u, img_w/w, img_h/h)
        v = subsample(v, img_w/w, img_h/h)

        pixels = numpy.dstack((y, u, v))
        pixels = ycocg_rgb(pixels)

        self._image = Image.fromarray(pixels)

        return self._image, (y,u,v)

def ycocg_test():
    w,h = 256, 256
    channel_data = numpy.full((w, h, 3), 0, dtype=numpy.single)
    channel_data[:,:,0] = numpy.full((w, h), 1/4, dtype=numpy.single)
    channel_data[:,:,1] = numpy.full((w, h), 1/2, dtype=numpy.single)
    channel_data[:,:,2] = numpy.full((w, h), -1/4, dtype=numpy.single)
    channel_pixels = ycocg_rgb(channel_data)
    channel_img = Image.fromarray(channel_pixels, mode="RGB")
    channel_img.save("ycocg-test.png")
          
if __name__ == "__main__": 
    ycocg_test()