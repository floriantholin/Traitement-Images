#ex1 : Re-focalisation  d’une  image  à  partir  d’une  séquence
#%% imports
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2

def show3(im1,im2,im3):
    plt.figure(figsize=(12,5))
    plt.subplot(1,3,1); plt.imshow(im1); plt.title("image 1"); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(im2); plt.title("image 2"); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(im3); plt.title("image 3"); plt.axis('off')
    plt.show()

im_1 = np.array(Image.open("../Images_TP/Refocus_1.png").convert('RGB'))
im_2 = np.array(Image.open("../Images_TP/Refocus_2.png").convert('RGB'))
im_3 = np.array(Image.open("../Images_TP/Refocus_3.png").convert('RGB'))

show3(im_1,im_2,im_3)

#%% 

def to_gray(img):
    return  .299*img[:, :, 0] + .587*img[:, :, 1] + .114*img[:, :, 2]

def laplace_magn(im_gray):
    return cv2.convertScaleAbs(cv2.Laplacian(im_gray, cv2.CV_64F, ksize=3))

def remove_focus(im_1,im_2,im_3):
    #convertion en gris
    im_1_gray = to_gray(im_1)
    im_2_gray = to_gray(im_2)
    im_3_gray = to_gray(im_3)

    #calcul Filtre de Laplace
    laplacien_1 = laplace_magn(im_1_gray)
    laplacien_2 = laplace_magn(im_2_gray)
    laplacien_3 = laplace_magn(im_3_gray)
    show3(laplacien_1,laplacien_2,laplacien_3)

    #On mesure la variance du filtre ed Laplace par lignes (blocs de n lignes)
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
    return im_ret

im_ret = remove_focus(im_1,im_2,im_3)
plt.imshow(im_ret)
plt.show()


