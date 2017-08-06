from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# defines block class
class Block:
    def __init__(self, loc, theta):
        self.loc = loc
        self.theta = theta

def extractGradients(im):
    blockarr=[]
    w, h = im.size
    imarray = np.array(im.getdata()).astype(np.float32).reshape((im.size[0],im.size[1]))
    #print(imarray)
    [dx, dy] = np.gradient(imarray)
    #print(np.min(dx), np.min(dy))
    numblocks=[(h//16),(w//16)]
    boxsize = [16, 16]
    for i in range(0, numblocks[0]):
        for n in range(0, numblocks[0]):
            box = [(boxsize[0] * i), (boxsize[1] * n), (boxsize[0] * (i + 1) + 1), (boxsize[1] * (n + 1)) + 1]
            vertw=np.mean(dy[box[0]:box[2], box[1]:box[3]])
            horzw=np.mean(dx[box[0]:box[2], box[1]:box[3]])
            theta=[horzw, vertw]
            blockarr.append(Block(box,theta))
    return blockarr


import os
dir=os.listdir('imgs')
path1="face.jpg"
im1 = Image.open(path1).convert('L')
blockarr=extractGradients(im1)
X, Y = np.meshgrid(np.arange(0, im1.size[0], 16), np.arange(0, im1.size[1], 16))
U = [blockarr[i].theta[0] for i in range(len(blockarr))]
V = [blockarr[i].theta[1] for i in range(len(blockarr))]
plt.figure()
Q = plt.quiver(X, Y, U, V, units='xy')
plt.ylim([0, im1.size[1]])
plt.xlim([0, im1.size[0]])
im = plt.imread(path1)
plot = plt.imshow(im, origin='lower')
plt.gca().invert_yaxis()
plt.show()


