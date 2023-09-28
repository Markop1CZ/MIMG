from PIL import Image
from scipy.fftpack import dct, idct
import numpy
import math
import zlib

def img_to_tiles(img, size):
    pixels = numpy.array(img)
    
    tile_w,tile_h = size,size
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
    
    compressed = zlib.compress(pixels.astype(numpy.half))
    buf += bytes([*pixels.shape, len(compressed)]) + compressed
    
    return buf

def read_tiles(data):
    offset = 0
    tiles = []

    while offset < len(data):
        w,h,length = data[offset:offset+3]
        pixels = numpy.frombuffer(zlib.decompress(data[offset+3:offset+3+length]), numpy.half).reshape(w, h)
        
        offset += length+3
        
        pixels = idct(idct(pixels.T, norm = 'ortho').T, norm = 'ortho')
        pixels = numpy.clip(numpy.rint(numpy.multiply(pixels, 255)), 0, 255).astype(numpy.uint8)

        tiles.append(pixels)

    return tiles

def test():
    tile_size = 16
    im = Image.open("dct-test.png").convert("L")
    w,h = im.size
    
    tiles = img_to_tiles(im, tile_size)

    buf = b""
    for t in tiles:
        buf += tile_to_bytes(t)

    print(len(buf))

    decompressed_tiles = read_tiles(buf)
    output_image = Image.new("L", (w,h))
    num = 0
    
    for j in range(math.ceil(h/tile_size)):
        for i in range(math.ceil(w/tile_size)):
            tile = Image.fromarray(decompressed_tiles[num], "L")
            output_image.paste(tile, (i*tile_size, j*tile_size))
            
            num += 1

    output_image.save("dct-test-out.png")
    
if __name__ == "__main__":
    test()
