import numpy
import math
import struct
import scipy
import zlib
import os
import signal
import traceback
from io import BytesIO
from scipy.fftpack import dct, idct
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

    pixels = numpy.multiply(numpy.dot(pixels, m), 255)
    pixels = numpy.clip(pixels, 0, 255)
    return numpy.rint(pixels).astype(numpy.uint8)

dct_quant = numpy.array([[8,  12, 24, 36, 11, 13, 15, 18],
                         [5,  7,  9,  11, 13, 15, 17, 23],
                         [7,  9,  11, 13, 15, 17, 19, 43],
                         [9,  11, 13, 15, 17, 19, 21, 62],
                         [11, 13, 15, 17, 19, 21, 42, 77],
                         [13, 15, 17, 19, 21, 38, 76, 86],
                         [15, 17, 19, 27, 38, 45, 77, 95],
                         [19, 28, 46, 62, 75, 88, 98, 101]])

def compress_channel_dct(pixels, tile_size=8):
    img_h,img_w = pixels.shape

    buf = struct.pack("H", tile_size)

    for j in range(math.ceil(img_h/tile_size)):
        for i in range(math.ceil(img_w/tile_size)):
            tile = numpy.zeros((tile_size, tile_size))
            tmp = pixels[j*tile_size:(j+1)*tile_size, i*tile_size:(i+1)*tile_size]
            tile[0:tmp.shape[0],0:tmp.shape[1]] = tmp

            ## mirror pixels in order to make the tiles square
            h_d = tile.shape[0]-tmp.shape[0]
            w_d = tile.shape[1]-tmp.shape[1]
            tile[tmp.shape[0]:] = tile[tmp.shape[0]-h_d-1:tile.shape[0]-h_d-1][::-1]
            tile[:,tmp.shape[1]:] = tile[:,tmp.shape[1]-w_d-1:tile.shape[1]-w_d-1][::-1]
            
            tile = dct(dct(tile.T, norm = 'ortho').T, norm = 'ortho')
            tile = numpy.divide(tile, dct_quant[:tile.shape[0], :tile.shape[1]])
            tile = numpy.add(numpy.around(tile, 1), 0.0)
            tile = tile.astype(numpy.int8)
    
            compressed = zlib.compress(tile.tobytes())
            buf += struct.pack("H", len(compressed)) 
            buf += compressed
    
    return buf

def decompress_channel_dct(buf, img_w, img_h):
    offset = 0
    tile_size, = struct.unpack("H", buf[offset:offset+2])
    offset += 2

    output_pixels = numpy.zeros((img_h + tile_size, img_w + tile_size))
    
    i = 0
    j = 0
    while offset < len(buf):
        length, = struct.unpack("H", buf[offset:offset+2])
        offset += 2
        pixels = numpy.frombuffer(zlib.decompress(buf[offset:offset+length]), numpy.int8).reshape(tile_size, tile_size)
        offset += length

        pixels = numpy.multiply(pixels, dct_quant[:pixels.shape[0], :pixels.shape[1]])
        pixels = idct(idct(pixels.T, norm = 'ortho').T, norm = 'ortho')

        output_pixels[j:j+tile_size, i:i+tile_size] = pixels
        
        i += tile_size
        if i >= img_w:
            i = 0
            j += tile_size
            
    return output_pixels[0:img_h, 0:img_w]

## vertical=0.25, horizontal=0.5
def subsample(pixels, horizontal=0.5, vertical=0.25):
    output = scipy.ndimage.zoom(pixels, (vertical, horizontal))

    return output
    
class ImageCompression:
    def __init__(self, pil_image):
        self._image = pil_image.convert("RGB")

    def get_image(self):
        return self._image

    def get_pixels(self):
        return numpy.array(self._image)

    def compress(self):
        buffer = bytearray()
        img_w,img_h = self._image.size
    
        print("w={0} h={1}".format(img_w, img_h))

        buffer += struct.pack("II", img_w, img_h)
        
        yuv = rgb_ycocg(self.get_pixels())

        y,u,v = numpy.dsplit(yuv, 3)
        y = y.reshape(y.shape[0], y.shape[1])
        u = subsample(u[:,:,0])
        v = subsample(v[:,:,0])

        for i,channel in enumerate((y,u,v)):
            if i == 0: ## y channel range: -128 127
                channel_data = numpy.rint(numpy.clip(numpy.multiply(channel, 128), -128, 127)).astype(numpy.int8)
            else: ## u v channel range: -128 127
                channel_data = numpy.rint(numpy.clip(numpy.multiply(channel, 256), -128, 127)).astype(numpy.int8)

            chan_h,chan_w = channel.shape
            compressed_data = compress_channel_dct(channel_data)
            
            length = len(compressed_data)

            buffer += struct.pack("III", chan_w, chan_h, length)
            buffer += compressed_data

        return buffer, (y, u, v)

    def decompress(self, buffer):
        img_w,img_h = struct.unpack("II", buffer[0:8])
        del buffer[0:8]

        channels = []
        
        for i in range(3):
            w,h,length = struct.unpack("III", buffer[0:12])
            del buffer[0:12]

            data = buffer[0:length]
            del buffer[0:length]

            channel = decompress_channel_dct(data, w, h)
            ##channel = numpy.frombuffer(zlib.decompress(data), numpy.uint8).astype(numpy.single).reshape(w,h)
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

## TEST
def _multiprocess_convert_image(filename, output_folder, debug_yuv=True):
    try:
        print("Converting: {0}".format(filename))

        img_name = os.path.splitext(os.path.basename(filename))[0]
        pil_img = Image.open(filename).convert("RGB")
        img = ImageCompression(pil_img)

        compressed, debug_channels = img.compress()
        compressed_len = len(compressed)
        result_img, debug_channels = img.decompress(compressed)

        ## Debug output if requested
        if debug_yuv:
            print("Exporting YUV")
            for i, debug_channel in enumerate(debug_channels):
                channel_data = numpy.empty((*debug_channel.shape, 3), dtype=numpy.single)
                channel_data[:,:,0] = numpy.full(debug_channel.shape, 0.5, dtype=numpy.single)
                channel_data[:,:,i] = debug_channel
                channel_pixels = ycocg_rgb(channel_data)
                channel_img = Image.fromarray(channel_pixels, mode="RGB")
                channel_img.save(os.path.join(output_folder, "{0}-chan-{1}.png".format(img_name, i)))

        ## Comparison
        print("Comparing against PNG/JPEG")
            
        ## Compare against PNG
        b = BytesIO()
        pil_img.save(b, format="png")
        png_len = b.getbuffer().nbytes
        
        # JPEG
        b = BytesIO()
        pil_img.save(b, format="jpeg")
        jpeg_len = b.getbuffer().nbytes

        raw_size_mb = (pil_img.size[0]*pil_img.size[1]*3)/10**6
        compressed_size_mb = compressed_len/10**6
        png_size_mb = png_len/10**6
        jpeg_size_mb = jpeg_len/10**6

        print("-> {0} compressed={1:.2f}MB png={2:.2f}MB ratio={3:.2f}% jpg={4:.2f}MB ratio={5:.2f}%".format(img_name,
                                                                                                            compressed_size_mb,
                                                                                                            png_size_mb,
                                                                                                            (compressed_size_mb/png_size_mb)*100,
                                                                                                            jpeg_size_mb,
                                                                                                            (compressed_size_mb/jpeg_size_mb)*100))
        ## Save converted color image and also the original for comparison
        result_img.save(os.path.join(output_folder, img_name+"-color.png"))
        pil_img.save(os.path.join(output_folder, img_name+"-!orig.png"))

        return [img_name, *pil_img.size, raw_size_mb, compressed_size_mb, compressed_size_mb/raw_size_mb, png_size_mb, compressed_size_mb/png_size_mb, jpeg_size_mb, compressed_size_mb/jpeg_size_mb]
    except Exception as e:
        print("[!] Error with file {0}: {1}".format(filename, e))
        traceback.print_exc()

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

## Converts all images in "test-image" folder into "test-output" folder and returns statistics
def test_convert_folder_threaded(input_folder, output_folder, debug_yuv=True, num_threads=6):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for fn in os.listdir(output_folder):
        f = os.path.join(output_folder, fn)
        if os.path.isfile(f):
            os.remove(f)

    images = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
    test_stats = []
    pending = []
    results = []

    pool = multiprocessing.Pool(multiprocessing.cpu_count(), init_worker)
    try:
        for img in images:
            r = pool.apply_async(_multiprocess_convert_image, (img, output_folder, debug_yuv))
            pending.append(r)

        while len(pending) > 0:
            r = pending.pop(0)
            r = r.get()
            if r != None:
                results.append(r)
        
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
            
    return results
    
def test():
    images_folder = "test-images"
    output_folder = "test-output"

    stats = test_convert_folder_threaded(images_folder, output_folder, num_threads=1)
    print(stats)

    csv = ""
    for r in stats:
        for i in r:
            csv += (i if type(i) == str else str(i).replace(".", ",")) + ";"
        csv += "\n"
    print(csv)
    
## should output a single color red image
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
    import multiprocessing
    
    test()
    ycocg_test()
