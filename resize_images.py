import time
from pathlib import Path

import cv2
import numpy as np

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def resize_image(filename, img_size=640):
    filename = Path(filename)
    with open(filename, "rb") as f:
        byte_array = f.read()
    file_bytes = np.asarray(bytearray(byte_array), dtype=np.uint8)
    t1 = time.perf_counter()
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = letterbox(img, (img_size, img_size), stride=img_size)[0]  # padded resize
    t2 = time.perf_counter()
    output_name = filename.with_stem(f"{filename.stem}-resized")
    cv2.imwrite(str(output_name), img)
    return t2-t1

def main():
    import sys
    if len(sys.argv)<2:
        print(f"Usage: {sys.argv[0]} [image_filename ...]")
        sys.exit(1)
    t1 = time.perf_counter()
    for f in sys.argv[1:]:
        t = resize_image(f)
        print(f"{f} resized in {t}s (excluding IO)")
    t2 = time.perf_counter()
    print(f"Total time for all images (including IO): {t2-t1}s")

if __name__ == "__main__":
    main()



