# https://pillow.readthedocs.io/en/3.0.x/handbook/tutorial.html

from PIL import Image
from math import log10

im = Image.open('neural_net_zoo.png')
print(im.format, im.size, im.mode)
for x in range(im.size[0]):
    for y in range(im.size[1]):
        pix = im.getpixel((x, y))
        if (pix[3] > 0 and pix[0] == pix[1] and pix[1] == pix[2]):
            im.putpixel((x, y), (255,)*3)
    print(f"Row {str(x).zfill(int(log10(im.size[0])))} of {im.size[0]}{('.'*(x%3)).ljust(3)} {int(x/im.size[0]*100)}%", end='\r')

im.save('flip_gray_out.png')

