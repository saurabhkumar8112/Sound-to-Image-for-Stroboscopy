import matplotlib.pyplot as plt
import matplotlib._png as png
import numpy as np 
path = "/home/mak/Desktop/Stroboscopy/AllImages/3.png"
img = png.read_png_int(path)
print img.dtype
print img[0:10,0:10,0]
plt.imshow(img)
plt.show()