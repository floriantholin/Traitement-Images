from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from random import randrange as rand


# Image.fromarray(im_dungeon).show()
def show2(im_ref,im,cmap= 'viridis',cmap_ref='viridis',title1='Before',title2='After',size=9):
    if size < 11 :
        f, axarr = plt.subplots(1,2)    #subplot(r,c) -> r:nb_rows, c:nb_columns
    else:
        f, axarr = plt.subplots(2,1)    #subplot(r,c) -> r:nb_rows, c:nb_columns
    axarr[0].set_title(title1)
    axarr[0].imshow(im_ref,cmap=cmap_ref)
    axarr[1].set_title(title2)
    axarr[1].imshow(im,cmap=cmap)
    f.set_figheight(size)
    f.set_figwidth(size)

def show3(im_ref,im1, im2,cmap= 'viridis',cmap_ref='viridis',title1='Step 1',title2='Step 2',size=10):
    if size < 11 :
        f, axarr = plt.subplots(1,3)    #subplot(r,c) -> r:nb_rows, c:nb_columns
    else:
        f, axarr = plt.subplots(3,1)    #subplot(r,c) -> r:nb_rows, c:nb_columns
    axarr[0].set_title('Original')
    axarr[0].imshow(im_ref,cmap=cmap_ref)
    axarr[1].set_title(title1)
    axarr[1].imshow(im1,cmap=cmap)
    axarr[2].set_title(title2)
    axarr[2].imshow(im2,cmap=cmap)
    f.set_figheight(size)
    f.set_figwidth(size)

def show_histo(hist,title='Histogramme'):
    plt.figure(figsize=(8, 2))
    plt.title(title)
    plt.bar(range(len(hist)), hist, color='gray')

def show1_grey(im,hist=True,title='Original',size=10):
    #affiche l'image de gris, l'histo et l'histocumulé si True
    if hist:
        if size < 11 :
            f, axarr = plt.subplots(1,3,figsize=(size, size//4))    #subplot(r,c) -> rows, columns
        else:
            f, axarr = plt.subplots(3,1,figsize=(size, size//4))    #subplot(r,c) -> rows, columns
        axarr[0].set_title(title)
        axarr[0].imshow(im,cmap='grey')
        histo = build_histo(im)
        axarr[1].set_title("Histogramme")
        axarr[1].bar(range(len(histo)), histo)
        histocumul = build_histo_cumul(histo)
        axarr[2].set_title('Histogramme cumulé')
        axarr[2].bar(range(len(histocumul)), histocumul)
    else :
        plt.figure(figsize =(size,size))
        plt.imshow(im,cmap='grey')
        plt.title(title)


def build_histo(im,dtype=int):
    assert(len(im.shape)==2)    #ne supporte qu'une couleur
    assert(im.dtype == np.uint8)    #ne supporte que des valeurs entre 0 et 255
    nb_classes = 256
    histo = np.zeros([nb_classes],dtype=dtype) #histo = [0]*nb_classes
    for i in im.ravel():
        histo[int(i)] += 1
    return histo

def build_histo_cumul(histo):
    histo_cumul = np.empty_like(histo)
    temp = 0
    for i,h in enumerate(histo):
        temp += h 
        histo_cumul[i] = temp
    return histo_cumul

def grey(im):
    R, G, B = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    return (0.2989 * R + 0.5870 * G + 0.1140 * B).astype(np.uint8)


def linear_filter_RGB(im,filtre):
    height, width, nb_channels = im.shape
    size_filtre = filtre.shape[0]
    assert(nb_channels < 4)        #besoin image RGB
    assert(size_filtre%2 == 1) #filtre carré de taille impaire
    bord = size_filtre//2
    img = np.zeros_like(im)
    for y in range(bord,height-bord):
        for x in range(bord,width-bord):
            for c in range(0,nb_channels):
                temp = np.sum(im[y-bord:y+bord+1,x-bord:x+bord+1,c]*filtre)    #temp : float64
                img[y,x,c] = temp*(temp>=0) if (temp<255) else 255    #saturation
    return img


def linear_filter_grey(im,filtre):
    height, width = im.shape
    size_filtre = filtre.shape[0]
    assert(size_filtre%2 == 1) #filtre carré de taille impaire
    bord = size_filtre//2
    img = np.zeros_like(im)
    for y in range(bord,height-bord):
        for x in range(bord,width-bord):
            temp = np.sum(im[y-bord:y+bord+1,x-bord:x+bord+1]*filtre)    #temp : float64
            img[y,x] = temp*(temp>=0) if (temp<255) else 255    #saturation
    return img

def sharpen(im):
    filtre = np.array([[0,-0.5,0],
                       [-0.5,3,-0.5],
                       [0,-0.5,0]])
    return linear_filter_RGB(im,filtre)

def sharpen_grey(im):
    filtre = np.array([[0,-0.5,0],
                       [-0.5,3,-0.5],
                       [0,-0.5,0]])
    return linear_filter_grey(im,filtre)

def lisse_grey(im,size_filtre): #mean filter
    assert(size_filtre%2 == 1)      #filtre de taille impaire
    filtre = np.array([[1/(size_filtre**2)]*size_filtre]*size_filtre)
    return linear_filter_grey(im,filtre)


def median_filter_grey(im,size_filtre):
    #size_filtre : nb de pixels sur un côté
    assert(len(im.shape) == 2)        #besoin image grey
    height, width = im.shape
    assert(size_filtre%2 == 1)      #filtre de taille impaire
    bord = size_filtre//2
    img = np.zeros_like(im)
    for y in range(bord,height-bord):
        for x in range(bord,width-bord):
                input = im[y-bord:y+bord+1,x-bord:x+bord+1]
                img[y,x] = np.median(input,axis=None)
    return img

def filtre_nagao_RGB(im):
    height, width, nb_channels = im.shape
    bord = 5//2 ; bord3 = 3//2
    img = np.zeros_like(im)
    for y in range(bord,height-bord):
        for x in range(bord,width-bord):
            for c in range(0,nb_channels):
                means = []; stds = []
                for xi in range (-1,2):
                    for yi in range(-1,2):
                        window1 = im[y-bord3+yi:y+bord3+1+yi,x-bord3+xi:x+bord3+1+xi,c]
                        means.append(np.mean(window1))
                        stds.append(np.mean(window1))
                img[y,x,c] = means[np.argmin(stds)]
    return img

def filtre_nagao_grey(im):
    height, width = im.shape
    bord = 5//2 ; bord3 = 3//2
    img = np.zeros_like(im)
    for y in range(bord,height-bord):
        for x in range(bord,width-bord):
            means = []; stds = []
            for xi in range (-1,2):
                for yi in range(-1,2):
                    window1 = im[y-bord3+yi:y+bord3+1+yi,x-bord3+xi:x+bord3+1+xi]
                    means.append(np.mean(window1))
                    stds.append(np.mean(window1))
            img[y,x] = means[np.argmin(stds)]
    return img

def gauss_filter3_grey(im):
    filtre = np.array([[1,2,1],
                       [1,4,1],
                       [1,2,1]])
    filtre = filtre/14
    return linear_filter_grey(im,filtre)

def binomial_filter5_grey(im):
    filtre = np.array([[1,4,6,4,1],
                       [4,16,24,16,4],
                       [6,24,36,24,6],
                       [4,16,24,16,4],
                       [1,4,6,4,1]])
    filtre = filtre/256
    return linear_filter_grey(im,filtre)

def gauss_filter5_grey(im):
    filtre = np.array([[2,4,5,4,2],
                       [4,9,12,9,4],
                       [5,12,15,12,5],
                       [4,9,12,9,4],
                       [2,4,5,4,2]])
    filtre = filtre/159
    return linear_filter_grey(im,filtre)

def gauss_filter_grey(im,size_filtre):
    assert(size_filtre%2 == 1)
    std = (size_filtre - 1)/6
    filtre = np.empty(shape=(size_filtre,size_filtre))
    for x in range(11):
        for y in range(11):
            filtre[y,x]=np.exp(-np.pi*(x**2+y**2)/std**2)
    filtre /= np.sum(filtre)
    return linear_filter_grey(im,filtre)

def gauss_filter2_grey(im,size_filtre,std):
    assert(size_filtre%2 == 1)
    filtre = np.empty(shape=(size_filtre,size_filtre))
    #print(filtre)
    for x in range(11):
        for y in range(11):
            filtre[y,x]=np.exp(-np.pi*(x**2+y**2)/std**2)
    filtre /= np.sum(filtre)
    return linear_filter_grey(im,filtre)