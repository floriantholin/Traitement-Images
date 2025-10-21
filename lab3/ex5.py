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
X, Y = np.meshgrid(x, y)

# Tour Eiffel au milieu : depth = 1
#    Pic au centre X=0.5, décroît latéralement
center_k = 12.0
far_center = np.exp(-((X - 0.5) ** 2) * center_k)  # proches de 1 au centre, 0 sur les côtés

# immeubles + route/voitures (side,bottom)
#side_near : 0 au centre, augmente vers les côtés
side_near = (np.abs(X - 0.5)) ** 1.5   #exposant pour montée non linéaire

#bottom_near : 0 en haut, 1 en bas
bottom_near = Y ** 4 -0.3

# Combinaison des éléments premier plan (proche 0)
near_score = np.clip(side_near + bottom_near, 0.0, 1.0)

# Application
alpha = 0.95 #rapproche l'arrière plan (0=no effect, 1=fort)
far_after_penalty = far_center * (1.0 - alpha * near_score)

# Normalisation [0,1]
depth_map = (far_after_penalty - far_after_penalty.min()) / (far_after_penalty.max() - far_after_penalty.min())

#en-cart en bas à droite : même plan que la tour eiffel (1)
depth_map[490:-10,660:-10] = 1

# Visualisation
plt.imshow(depth_map, cmap='gray', vmin=0, vmax=1)
plt.title("Depth map (1=arrière-plan, 0=premier-plan)")
plt.axis('off')
plt.show()


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
    near_focus = focus_diff < 0.18
    result[near_focus] = img[near_focus]

    return result



result = apply_depth_blur(image, depth_map, focus_depth=1,max_blur=8)

plt.figure(figsize=(10, 6))
plt.imshow(result)
plt.title(f"Focus sur la tour Eiffel")
plt.show()