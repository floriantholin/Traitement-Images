#ex1 : Re-focalisation  d’une  image  à  partir  d’une  séquence
#%% imports
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2

img_1 = Image.open("../Images_TP/Refocus_1.png")
img_2 = Image.open("../Images_TP/Refocus_2.png")
img_3 = Image.open("../Images_TP/Refocus_3.png")

im_1 = np.array(img_1)[:,:,:3] #convert to RGBa
im_2 = np.array(img_2)[:,:,:3]
im_3 = np.array(img_3)[:,:,:3]

im_1_gray = np.array(img_1.convert('L')) # convert to gray
im_2_gray = np.array(img_2.convert('L'))
im_3_gray = np.array(img_3.convert('L'))

plt.figure(figsize=(12,5))
plt.subplot(1,3,1); plt.imshow(im_1); plt.title("image 1"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(im_2); plt.title("image 2"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(im_3); plt.title("image 3"); plt.axis('off')
plt.show()

#%% Laplacian filter

laplacien_1 = cv2.convertScaleAbs(cv2.Laplacian(im_1_gray, cv2.CV_64F, ksize=3))
laplacien_2 = cv2.convertScaleAbs(cv2.Laplacian(im_2_gray, cv2.CV_64F, ksize=3))
laplacien_3 = cv2.convertScaleAbs(cv2.Laplacian(im_3_gray, cv2.CV_64F, ksize=3))

plt.figure(figsize=(12,5))
plt.subplot(1,3,1); plt.imshow(laplacien_1); plt.title("image 1"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(laplacien_2); plt.title("image 2"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(laplacien_3); plt.title("image 3"); plt.axis('off')
plt.show()

#%% 
#On mesure la variance du Laplacien par lignes (blocs de n lignes)
n = 3
bord = n//2
im_ret = np.zeros_like(im_1)
height, width, channel = im_ret.shape

images = [im_1,im_2,im_3]
for y in range(bord,height-bord):
    values = [laplacien_1[y-bord:y+bord+1,:].var(),
                laplacien_2[y-bord:y+bord+1,:].var(),
                laplacien_3[y-bord:y+bord+1,:].var()]
    argsort = np.argsort(values)
    argmax = argsort[-1] ;argmin= argsort[0]
    mean = np.mean(values)
    im_ret[y] = images[argmax][y] #on prend l'image avec variance Laplace max


plt.imshow(im_ret)


