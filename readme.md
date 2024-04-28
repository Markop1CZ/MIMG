# imgf
Experimental image compressor similar to JPEG/JPEG2000 written in Python using Numpy, Scipy, Pillow and cv2 libraries.
Works in [YCoCg](https://en.wikipedia.org/wiki/YCoCg) color space, supports chroma subsampling and DCT.
## Test image
![Test image](https://files.markop1.cz/imgf/photo-02.png)
## Test image compressed using imgf
![Test image compressed](https://files.markop1.cz/imgf/photo-02-converted.png)
## Test image statistics
![mimg statistics](https://files.markop1.cz/imgf/mimg-stats.png)
## Comparison
Note: ignore the imgf file size as the result of the compressor was converted back to PNG.
![mimg comparison](https://files.markop1.cz/imgf/mimg-comparison.png)