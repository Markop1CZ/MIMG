import numpy
import struct
import math
import os
import zlib
from io import BytesIO
from PIL import Image

FLAGS_RLE = 1 << 0
FLAGS_ZLIB = 1 << 1
FLAGS_RGB = 1 << 2
FLAGS_YUV = 1 << 3

FILE_SIG = b"IMG"
FILE_SIG_L = 3

def RGB_YUV(imgdata):
    m = numpy.array([[0.29900, -0.16874,  0.50000],
                     [0.58700, -0.33126, -0.41869],
                     [0.11400, 0.50000, -0.08131]])
     
    yuv = numpy.dot(imgdata, m)
    yuv[:, :, 1:] += 128
    
    return yuv.astype(numpy.uint8)

def YUV_RGB(imgdata):
    m = numpy.array([[1.0, 1.0, 1.0],
                     [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                     [ 1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])
    
    rgb = numpy.dot(imgdata, m)
    
    rgb[:, :, 0] -= 179.45477266423404
    rgb[: ,:, 1] += 135.45870971679688
    rgb[:, :, 2] -= 226.8183044444304
    
    return rgb.astype(numpy.uint8)

class MImg:
    def __init__(self, flags, w, h, pil_img, chroma_scale=0.5):
        self.flags = flags
        self.w = w
        self.h = h
        self.pil_img = pil_img
        self.chroma_scale = chroma_scale

    @staticmethod
    def from_pil(pil_img, flags=FLAGS_ZLIB | FLAGS_YUV):
        return MImg(flags, pil_img.width, pil_img.height, pil_img)

    @staticmethod
    def from_bytes(b):
        return MImg.from_file(BytesIO(b))

    @staticmethod
    def from_file(f):    
        sig = f.read(FILE_SIG_L)
        if sig != FILE_SIG:
            raise Exception("Signature mismatch!")

        flags = f.read(1)[0]
        w,h,chroma_w,chroma_h = struct.unpack("IIII", f.read(16))

        print(chroma_w, chroma_h)

        pixel_data = []
        for i in range(3):
            l = struct.unpack("I", f.read(4))[0]
            data = f.read(l)
            
            if flags & FLAGS_ZLIB:
                data = zlib.decompress(data)
                
            print(len(data))
            pixel_data.append(data)

        imgdata = numpy.ndarray((h, w, 3), dtype=numpy.uint8)
        pil_image = None
        if flags & FLAGS_YUV:
            print("yuv mode (reading)")
            y1 = numpy.frombuffer(pixel_data[0], dtype=numpy.uint8)
            y1 = numpy.reshape(y1, (h, w))

            imgdata[:, :, 0] = y1
            
            for i in range(1, 3):
                chroma = numpy.frombuffer(pixel_data[i], dtype=numpy.uint8)
                chroma = numpy.reshape(chroma, (chroma_h, chroma_w))
                chroma = Image.frombytes("L", (chroma_w, chroma_h), chroma)
                chroma = chroma.resize((w, h))
                chroma = numpy.array(chroma)

                imgdata[:, :, i] = chroma

            imgdata = YUV_RGB(imgdata)
            pil_image = Image.frombytes("RGB", (w, h), imgdata)            
        else:
            print("rgb mode (reading)")
            for i in range(0, 3):
                imgdata[:, :, i] = numpy.reshape(numpy.frombuffer(pixel_data[i], dtype=numpy.uint8), (h, w))
            print(imgdata)
            pil_image = Image.frombytes("RGB", (w, h), imgdata)

        return MImg(flags, w, h, pil_image)

    def to_bytes(self):
        b = b""
        b += FILE_SIG
        b += bytes([self.flags])

        b += struct.pack("II", self.w, self.h)

        chroma_w = math.floor(self.w*self.chroma_scale)
        chroma_h = math.floor(self.h*self.chroma_scale)

        b += struct.pack("II", chroma_w, chroma_h)
    
        imgdata = numpy.array(self.pil_img)
        pixel_data = []
        if self.flags & FLAGS_YUV:
            print("yuv model")
            imgdata = RGB_YUV(imgdata)

            pixel_data.append(imgdata[:, :, 0].tobytes())
            ## chroma subsample
            for i in range(1, 3):
                chroma = Image.fromarray(numpy.reshape(imgdata[:, :, i], (self.h, self.w)) , 'L')
                chroma = chroma.resize((chroma_w, chroma_h))
                chroma.save("test_chroma_{0}.jpg".format(i))
                chroma = numpy.array(chroma).astype("uint8")

                pixel_data.append(chroma.tobytes())
        else:
            print("rgb model")
            print(imgdata)
            for i in range(0, 3):
                pixel_data.append(imgdata[:, :, i].tobytes())

        ## ZLIB COMPRESS
        if self.flags & FLAGS_ZLIB:
            for i,data in enumerate(pixel_data):
                pixel_data[i] = zlib.compress(data)

        ## PIXEL DATA
        for data in pixel_data:
            b += struct.pack("I", len(data))
            b += data

        return b

    def to_file(self, fn):
        f = open(fn, "wb")
        f.write(self.to_bytes())
        f.close()

    def save_pil(self, name):
        self.pil_img.save(name)

def convert_test_images():
    test_dir = "test"
    chroma_scale = 1.0
    files = os.listdir(test_dir)
    
    for f in files:
        if not "_converted" in f and not ".buf" in f:
            print(f)
            r = MImg.from_pil(Image.open(os.path.join(test_dir, f)))
            r.chroma_scale = chroma_scale
            out_fn = os.path.join(test_dir, os.path.splitext(f)[0] + ".buf")
            r.to_file(out_fn)
            b = MImg.from_file(open(out_fn, "rb"))
            b.save_pil(os.path.join(test_dir, os.path.splitext(f)[0] + "_converted.jpg"))

convert_test_images()

