#%% ex5 : on refait comme l'ex4 mais avec un focus la profondeur max

#%% imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

path = "../Images_TP/tour-eiffel.jpg"
image = np.array(Image.open(path).convert("RGB"))
plt.imshow(image);plt.show()
height,width,_ = image.shape

#%% Construction de la depth map (1=arrière plan, 0==premier plan)

#Grille normalisée [0..1]
x = np.linspace(0.0, 1.0, width)
y = np.linspace(0.0, 1.0, height)
X, Y = np.meshgrid(x, y)   # shape (height, width)

# Composante 'loin' centrée horizontalement (Tour Eiffel au milieu)
#    Pic au centre X=0.5, décroît latéralement
center_k = 12.0
far_center = np.exp(-((X - 0.5) ** 2) * center_k)  # proches de 1 au centre, -> 0 sur les côtés

# côtés et le bas (immeubles + route/voitures)
# a) side_near : 0 au centre, augmente vers les côtés
side_near = (np.abs(X - 0.5)) ** 1.5   # l'exposant rend la montée non linéaire

# b) bottom_near : 0 en haut, 1 en bas (renforce la proximité en bas de l'image)
bottom_near = Y ** 4 -0.3

# Combinaison des éléments premier plan (valeurs ~0..1)
near_score = np.clip(side_near + bottom_near, 0.0, 1.0)

# Application
#    alpha contrôle combien le near_score rapproche l'arrière plan'plan (0=no effect, 1=fort)
alpha = 0.95
far_after_penalty = far_center * (1.0 - alpha * near_score)

# Normalisation finale (assure [0,1] et convention 1=loin, 0=près)
depth_map = (far_after_penalty - far_after_penalty.min()) / (far_after_penalty.max() - far_after_penalty.min())

#encart en bas à droite : même plan que la tour eiffel (far)
depth_map[490:-10,660:-10] = 1

# Visualisation
plt.imshow(depth_map, cmap='gray', vmin=0, vmax=1)
plt.title("Depth map (1=arrière-plan, 0=premier-plan)")
plt.axis('off')
plt.show()





#%% Floutage en fonction du focus (arrière-plan = 1) et de la depth

def apply_depth_blur(img, depth, focus_depth=0.0, max_blur=15):
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
    near_focus = focus_diff < 0.18
    result[near_focus] = img[near_focus]

    # Affichage du résultat
    plt.figure(figsize=(10, 6))
    plt.imshow(result)
    plt.title(f"Flou de profondeur (focus={focus_depth:.2f})")
    plt.axis('off')
    plt.show()

    return result

#test
result = apply_depth_blur(image, depth_map, focus_depth=1.0, max_blur=8)
