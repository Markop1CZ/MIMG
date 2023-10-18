from compression import ImageCompression, rgb_ycocg, ycocg_rgb
from PIL import Image
from urllib import request
import numpy
import os

def format_size(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def calculate_compression_stats(pil_im, compressed):
    pass

def get_debug_channel_composite(debug_channels):
    empty_chan = numpy.empty((*debug_channels[0].shape, 3), dtype=numpy.single)
    empty_chan[:,:,0] = numpy.full(debug_channels[0].shape, 0.5, dtype=numpy.single)

    im_size = debug_channels[0].shape[::-1]
    output_img = Image.new("RGB", (im_size[0]*3, im_size[1]))
    
    for i,debug_channel in enumerate(debug_channels):
        channel_data = numpy.copy(empty_chan)
        channel_data[:,:,i] = debug_channel
        channel_pixels = ycocg_rgb(channel_data)
        
        channel_img = Image.fromarray(channel_pixels, mode="RGB")
        output_img.paste(channel_img, (i*im_size[0], 0))

    return output_img

## return decompressed image after compression
## and compressed size
def test_image(pil_img, subsample):
    img = ImageCompression(pil_img, subsample)

    compressed, debug_channels = img.compress()
    compressed_len = len(compressed)
    output_image, debug_channels = img.decompress(compressed)

    return compressed_len, output_image, debug_channels

def test_images_convert_folder(input_folder, output_folder, subsample=(0.5, 0.25), debug_yuv=True):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for fn in os.listdir(output_folder):
        f = os.path.join(output_folder, fn)
        if os.path.isfile(f):
            os.remove(f)

    for file in os.listdir(input_folder):
        print("Converting {0}:".format(file))
        
        img_path = os.path.join(input_folder, file)
        img_name = os.path.splitext(file)[0]

        pil_img = Image.open(img_path).convert("RGB")
        
        compressed_size, output_image, debug_channels = test_image(pil_img, subsample)

        output_image.save(os.path.join(output_folder, "{0}-color.png".format(img_name)))

        if debug_yuv:
            yuv_image = get_debug_channel_composite(debug_channels)
            yuv_image.save(os.path.join(output_folder, "{0}-debug.png".format(img_name)))
        
        print(img_name, format_size(compressed_size))

def test_random_images(num):
    pass

if __name__ == "__main__":
    test_images_convert_folder("test-images", "test-images-output")
