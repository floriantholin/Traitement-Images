#%% imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

path = "../IMAGES_TP/Profondeur.png"
img = np.array(Image.open(path).convert("RGB"))
h, w, _ = img.shape

#%% génération de la carte de profondeur (depth)
# Dimensions
h, w = img.shape[:2]

#Dégradé global progressif (haut gauche : loin = 1, bas droit : proche = 0)
yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
depth = 1 - (xx * 0.6 + yy * 0.4)

#Rectangles de premier plan (zones proches = 0)
mask_rect = np.ones_like(depth)
#Bas noir
y_start = int(h * 0.7)
mask_rect[y_start:, :] = 0.0
#Camion
y_start_camion = int(h * 0.17)
x_start_camion = int(w * 0.55)
mask_rect[y_start_camion:, x_start_camion:] = 0.0

#Fusion : rapprocher ces zones
depth *=  mask_rect #(1 - 0.6 * (1 - mask_rect))

#Normalisation entre 0 et 1
depth -= depth.min()
depth /= depth.max()


plt.figure(figsize=(8, 6))
plt.imshow(depth, cmap='gray')
plt.title("Carte des profondeurs")
plt.show()

# max de la depth autour du camion : seuil pour le floutage
max_depth_camion = np.max(depth[y_start_camion:, x_start_camion:])


#%% Floutage en fonction du focus et de la depth

def apply_depth_blur(img, depth_map, focus_depth=0.0, max_blur=15):
    #Applique un flou dépendant de la profondeur.
    #depth_map  : tableau numpy 2D (valeurs normalisées 0-1)
    #focus_depth : profondeur du plan net (0 = proche, 1 = loin)
    #max_blur : intensité maximale du flou (pixels)

    # Calcul de la différence à la profondeur de focus
    focus_diff = np.abs(depth_map - focus_depth)

    # Normalisation de la différence du flou
    blur_strength = (focus_diff / focus_diff.max()) * max_blur
    blur_strength = blur_strength.astype(np.float32)

    # Application du flou par paliers (Gaussien: masque plus gros si blur_strength plus grand)
    result = np.zeros_like(img)
    step = 2; bord = step//2
    for i in range(0, max_blur + 1, step):
        mask = (i - bord <= blur_strength) & (blur_strength <= i + bord)
        if np.any(mask):
            blurred = cv2.GaussianBlur(img, (max(1, i*2+1), max(1, i*2+1)), 0)
            result[mask] = blurred[mask]

    # Seuillage : zones de focus_diff très faibles restent nettes
    near_focus = focus_diff < max_depth_camion
    result[near_focus] = img[near_focus]

    return result



result = apply_depth_blur(img, depth, focus_depth=0)

plt.figure(figsize=(10, 6))
plt.imshow(result)
plt.title(f"Focus sur le camion (premier plan)")
plt.show()
