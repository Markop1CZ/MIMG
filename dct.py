from PIL import Image
from scipy.fftpack import dct, idct
import numpy
import math
import zlib

def img_to_tiles(img):
    pixels = numpy.array(img)
    
    tile_w,tile_h = 16,16
    img_w,img_h = img.size
    last_w, last_h = img_w%tile_w, img_h%tile_h

    tiles = []

    for j in range(math.ceil(img_h/tile_h)):
        for i in range(math.ceil(img_w/tile_w)):
            tile = pixels[j*tile_h:(j+1)*tile_h, i*tile_w:(i+1)*tile_w]
            tiles.append(tile)
    return math.ceil(img_w/tile_w),math.ceil(img_h/tile_h),tiles

def tile_to_bytes(img_tile):
    data = b""
    
    arr = numpy.array(img_tile)
    
    arr = numpy.divide(arr, 255)
    arr = dct(dct(arr.T, norm = 'ortho').T, norm = 'ortho')
    arr = numpy.around(arr, 1)
    compressed = zlib.compress(arr.astype(numpy.half))

    data += bytes([*arr.shape, len(compressed)]) + compressed
    return data

def read_tiles(data):
    offset = 0

    tiles = []

    while offset < len(data):
        w,h,length = data[offset:offset+3]
        arr = numpy.frombuffer(zlib.decompress(data[offset+3:offset+3+length]), numpy.half).reshape(w, h)
        offset += length+3
        
        arr = idct(idct(arr.T, norm = 'ortho').T, norm = 'ortho')

        arr = numpy.clip(numpy.rint(numpy.multiply(arr, 255)), 0, 255).astype(numpy.uint8)

        tiles.append(arr)

    return tiles

def test():
    im = Image.open("dct-test-2.png").convert("L")

    w,h = im.size

    tilesi,tilesj,tiles = img_to_tiles(im)

    buf = b""
    for t in tiles:
        buf += tile_to_bytes(t)

    print(len(buf))

    tiles2 = read_tiles(buf)
    im2 = Image.new("L", (w,h))
    num = 0
    for j in range(tilesj):
        for i in range(tilesi):
            tile = Image.fromarray(tiles2[num], "L")
            im2.paste(tile, (i*16, j*16))
            num += 1

    im2.save("test.png")
    
if __name__ == "__main__":
    test()
