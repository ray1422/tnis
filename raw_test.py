import matplotlib.pyplot as plt
import numpy as np
import rawpy
import cv2

f = "./dataset/origin/DSC05420.ARW"
f = "./dataset/edited/DSC02969-LR.JPG"
try:
    with rawpy.imread(f) as raw:
        img = raw.postprocess(use_camera_wb=True, output_bps=16)
        img = np.float32(img)/(2.**16-1)
except rawpy._rawpy.LibRawFileUnsupportedError:
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 255.0
except Exception as e:
    raise e

print(np.min(img))
print(np.max(img))

plt.imshow(img)
plt.show()