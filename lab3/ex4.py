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

# 1. Dégradé global progressif (haut gauche → loin, bas droit → proche)
yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
depth = 1 - (xx * 0.6 + yy * 0.4)  # clair = loin, noir = proche

# 2. Rectangles de premier plan (zones proches)
mask_rect = np.ones_like(depth)

# a) Bas (zone noire du sol)
mask_rect[int(h * 0.7):, :] = 0.0

# b) Camion (droite inférieure)
y_camion_start = int(h * 0.2)
mask_rect[y_camion_start:, int(w * 0.55):] = 0.0

# 3. Fusion : rapprocher ces zones
depth *= (1 - 0.6 * (1 - mask_rect))

# 4. Normalisation entre 0 et 1
depth -= depth.min()
depth /= depth.max()

# 5. Visualisation
plt.figure(figsize=(8, 6))
plt.imshow(depth, cmap='gray')
plt.title("Carte de profondeur : fond global + camion limité en hauteur")
plt.axis('off')
plt.show()

# max de la depth autour du camion sert au seuil pour le floutage)
x_start, x_end = int(0.7*w), w   # camion à droite
y_start, y_end = int(0.3*h), int(0.8*h)
max_depth_camion = np.max(depth[y_camion_start:, int(w * 0.55):])


#%% Floutage en fonction du focus et de la depth

def apply_depth_blur(img, depth_map, focus_depth=0.0, max_blur=15):
    """
    Applique un flou dépendant de la profondeur.
    
    Arguments :
      image_path : chemin vers l'image originale
      depth_map  : tableau numpy 2D (valeurs normalisées 0-1)
      focus_depth : profondeur du plan net (0 = proche, 1 = loin)
      max_blur : intensité maximale du flou (pixels)
    """
    # Dimentions de l'image
    h, w, _ = img.shape

    # Calcul de la différence à la profondeur de focus
    focus_diff = np.abs(depth - focus_depth)

    # Normalisation de la différence du flou
    blur_strength = (focus_diff / focus_diff.max()) * max_blur
    blur_strength = blur_strength.astype(np.float32)

    # Création d'une image résultat
    result = np.zeros_like(img)

    # Application du flou adaptatif par paliers
    # (pour éviter un flou pixel-par-pixel trop lent)
    step = 2; bord = step//2
    for i in range(0, max_blur + 1, step):
        mask = (i - bord <= blur_strength) & (blur_strength <= i + bord)
        if np.any(mask):
            blurred = cv2.GaussianBlur(img, (max(1, i*2+1), max(1, i*2+1)), 0)
            result[mask] = blurred[mask]
    
    # Les zones de focus_diff très faibles restent nettes
    near_focus = focus_diff < max_depth_camion
    result[near_focus] = img[near_focus]

    # Affichage du résultat
    plt.figure(figsize=(10, 6))
    plt.imshow(result)
    plt.title(f"Flou de profondeur (focus={focus_depth:.2f})")
    plt.axis('off')
    plt.show()

    return result

#test
result = apply_depth_blur(img, depth, focus_depth=0.0)
