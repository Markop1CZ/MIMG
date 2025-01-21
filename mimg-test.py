import os
import time
import traceback
import json
import hashlib
import cProfile
import pstats
from mimg import MImg
from PIL import Image
from io import BytesIO

INPUT_DIR = "test-images"
OUTPUT_DIR = "test-images-output"
STATS_OUTPUT_DIR = "test-images-stats"
JPEG_QUALITY = 35

def format_size(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def test_get_image_png_jpeg_comparison(input_filepath, pil_img):
    comparison_file = os.path.join(STATS_OUTPUT_DIR, "!images.json")

    comparison = {}
    if os.path.exists(comparison_file):
        f = open(comparison_file, "r")
        comparison = json.loads(f.read())
        f.close()

    sha1 = hashlib.sha1()
    with open(input_filepath, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha1.update(data)

    file_hash = sha1.hexdigest()

    if file_hash in comparison:
        result = comparison[file_hash]
    else:
        png_buf = BytesIO()
        pil_img.save(png_buf, format="png")
        png_size = png_buf.getbuffer().nbytes

        jpeg_buf = BytesIO()
        pil_img.save(jpeg_buf, format="jpeg", quality=JPEG_QUALITY)
        jpeg_size = jpeg_buf.getbuffer().nbytes

        result = [png_size, jpeg_size]
        comparison[file_hash] = result

    f = open(comparison_file, "w")
    f.write(json.dumps(comparison))
    f.close()

    return result

def test_compress_image(input_filepath, output_dir):
    kb = 1024
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    stats = {}

    input_filename = os.path.basename(input_filepath)
    name,ext = os.path.splitext(input_filename)

    output_filename = "{0}.buf".format(name)
    output_filepath = os.path.join(output_dir, output_filename)
    pil_img = Image.open(input_filepath).convert("RGB")

    png_size, jpeg_size = test_get_image_png_jpeg_comparison(input_filepath, pil_img)

    t0 = time.time()
    try:
        img = MImg(pil_img)
        img.save(output_filepath)
        mimg_size = os.stat(output_filepath).st_size
    except Exception as e:
        print("Error compressing {0}:".format(input_filename))
        traceback.print_exc()
        return 
    
    t1 = time.time()
    try:
        img2 = MImg.open(output_filepath)
        img2._image.save(os.path.join(output_dir, "{0}.png".format(name)))
    except Exception as e:
        print("Error decompressing {0}:".format(output_filename))
        traceback.print_exc()
        return
    t2 = time.time()

    stats["name"] = name
    stats["width"], stats["height"] = pil_img.size
    stats["size_raw"] = (3 * pil_img.size[0] * pil_img.size[1]) / kb
    stats["size_mimg"] = mimg_size / kb
    stats["size_png"] = png_size / kb
    stats["size_jpeg"] = jpeg_size / kb
    stats["time_com"] = t1 - t0
    stats["time_dec"] = t2 - t1

    return stats

def test_compress_images(input_dir, output_dir):
    stats = []

    print("Testing compression in {0}".format(input_dir))
    
    for file in os.listdir(input_dir):
        input_filepath = os.path.join(input_dir, file)
        
        compression_stats = test_compress_image(input_filepath, output_dir)
        stats.append(compression_stats)
        
        print("file={name} compression={time_com:.1f}s decompression={time_dec:.1f}s compressed={size_mimg:.0f}KiB jpeg={size_jpeg:.0f}KiB png={size_png:.0f}KiB".format(**compression_stats))
    
    heading = list(stats[0].keys())

    stats_filename = "{0}.csv".format(time.strftime("%Y-%m-%d_%H-%M-%S"))

    f = open(os.path.join(STATS_OUTPUT_DIR, stats_filename), "w")
    f.write(";".join(heading) + "\n")

    for s in stats:
        values = [str(k).replace(".", ",") if type(k) == int or type(k) == float else str(k) for k in s.values()]

        f.write(";".join(values))
        f.write("\n")
    f.close()

def test_cprofile():
    pr = cProfile.Profile()
    pr.enable()
    pr.runcall(test_compress_image, "test-images/render-03.png", "test-images-output")
    pr.disable()

    p = pstats.Stats(pr)
    p.strip_dirs().sort_stats("cumtime").print_stats()
    p.print_callers("_wrapfunc") ##asarray, _r2rn, iscomplexobj

if __name__ == "__main__":
    test_compress_images(INPUT_DIR, OUTPUT_DIR)