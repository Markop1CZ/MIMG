from PIL import Image
from scipy.fftpack import dct, idct
import numpy
import math
import zlib
import struct

## https://docs.python.org/3/library/struct.html#format-characters

def img_to_tiles(img, size):
    pixels = numpy.array(img)
    
    tile_w,tile_h = size
    img_w,img_h = img.size
    last_w, last_h = img_w%tile_w, img_h%tile_h

    tiles = []

    for j in range(math.ceil(img_h/tile_h)):
        for i in range(math.ceil(img_w/tile_w)):
            tile = pixels[j*tile_h:(j+1)*tile_h, i*tile_w:(i+1)*tile_w]
            tiles.append(tile)
            
    return tiles

def tile_to_bytes(img_tile):
    buf = b""
    
    pixels = numpy.array(img_tile)
    
    pixels = numpy.divide(pixels, 255)
    pixels = dct(dct(pixels.T, norm = 'ortho').T, norm = 'ortho')
    pixels = numpy.add(numpy.around(pixels, 1), 0.0)
    ##print(arr)
    
    compressed = zlib.compress(pixels.astype(numpy.half).tobytes())
    buf += struct.pack("BBH", *pixels.shape, len(compressed)) 
    buf += compressed
    
    return buf

def read_tiles(data):
    offset = 0
    tiles = []

    while offset < len(data):
        w,h,length = struct.unpack("BBH", data[offset:offset+4])
        offset += 4
        pixels = numpy.frombuffer(zlib.decompress(data[offset:offset+length]), numpy.half).reshape(w, h)
        offset += length
        
        pixels = idct(idct(pixels.T, norm = 'ortho').T, norm = 'ortho')
        pixels = numpy.clip(numpy.rint(numpy.multiply(pixels, 255)), 0, 255).astype(numpy.uint8)

        tiles.append(pixels)

    return tiles

## read/write functions ("api" use)
def write_image_tiles(im, tile_size):
    im = im.convert("L")

    tiles = img_to_tiles(im, tile_size)
    
    buf = struct.pack("IIBB", *im.size, *tile_size)
    for t in tiles:
        buf += tile_to_bytes(t)

    return buf

def read_image_tiles(buf):
    im_w,im_h,tile_w,tile_h = struct.unpack("IIBB", buf[:10])

    decompressed_tiles = read_tiles(buf[10:])
    output_image = Image.new("L", (im_w,im_h))
    num = 0
    
    for j in range(math.ceil(im_h/tile_h)):
        for i in range(math.ceil(im_w/tile_w)):
            tile = Image.fromarray(decompressed_tiles[num], "L")
            output_image.paste(tile, (i*tile_w, j*tile_h))
            
            num += 1

    return output_image

## test
def test():
    tile_size = 16
    im = Image.open("dct-test.png").convert("L")
    w,h = im.size
    
    buf = write_image_tiles(im, (tile_size, tile_size))
    print(len(buf))

    output_image = read_image_tiles(buf)
    output_image.save("dct-test-out.png")
    
if __name__ == "__main__":
    test()
