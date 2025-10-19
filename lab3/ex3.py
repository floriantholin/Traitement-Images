#ex3 : Seam-carving
#%%
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
#from scipy.ndimage import sobel
import cv2

path = "../IMAGES_TP/Resize.png"
img = np.array(Image.open(path).convert("RGB"))
#%%

def compute_energy(image):
    #re-calcule du gris est plus rapide que carve_seam(gray)
    gray = .299*img[:, :, 0] + .587*img[:, :, 1] + .114*img[:, :, 2]
    #dx = sobel(gray, axis=1)
    #dy = sobel(gray, axis=0)
    #energy = np.hypot(dx, dy) #np.sqrt(grad_x**2 + grad_y**2)
    dy, dx = np.gradient(gray)  #plus rapide
    energy = cv2.magnitude(dx, dy)  #plus rapide
    return energy

def find_vertical_seam(energy):
    h, w = energy.shape
    cost = energy.copy()
    backtrack = np.zeros_like(cost, dtype=np.int32)
    
    for i in range(1, h):
        for j in range(w):
            left = max(j-1, 0)
            right = min(j+1, w-1)
            idx_min = np.argmin(cost[i-1, left:right+1]) + left
            cost[i, j] += cost[i-1, idx_min]
            backtrack[i, j] = idx_min
            
    seam = []
    j = np.argmin(cost[-1])
    for i in reversed(range(h)):
        seam.append((i, j))
        j = backtrack[i, j]
    return seam[::-1]

def carve_vertical_seam(image, seam):
    h, w, _ = image.shape
    new_image = np.zeros((h, w-1, 3), dtype=image.dtype)
    for i, j in seam:
        new_image[i, :, :] = np.delete(image[i, :, :], j, axis=0)
    return new_image

def show_seam(image,seam,n,num_seams):
    for i, j in seam:
        image[i, j] = [255,0,0]
    plt.title(f'seam : {n+1} /{num_seams}')
    plt.imshow(image)
    plt.show()


image_w = img.shape[1]

num_seams = image_w//4  # nombre de seams à supprimer
for n in range(num_seams):
    energy = compute_energy(img)
    seam = find_vertical_seam(energy)
    show_seam(img,seam,n,num_seams) #affichage de l'éclair
    img = carve_vertical_seam(img, seam)

plt.title('image finale')
plt.imshow(img)
plt.show()

