#ex2 : Mosaïc
#%% imports
from matplotlib import pyplot as plt
import cv2
import numpy as np

img1 = cv2.imread('../Images_TP/Mosaic_1.png')  #référence (non transformée)
img2 = cv2.imread('../Images_TP/Mosaic_2.png')[2:,3:,:]  #crop le bord blanc
#%%
def match_and_place(im1,im2):
    # Détection des points remarquables (utilise ORB)
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    #match
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Points pour estimateAffinePartial2D (img2 -> img1)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estime la transformation rigide (rotation + translation) img2 par rapport à img1
    M, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)

    # taille de l'image finale (canevas)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas_w = w1 + w2
    canvas_h = max(h1, h2)

    # Transformation img2 => translation et rotation
    warped_img2 = cv2.warpAffine(
        img2, M, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Masque : la rotation par warpAffine créé des pixels bizarres autour de img2 -> on résout avec un masque
    src_mask = np.ones((h2, w2), dtype=np.uint8) * 255
    warped_mask = cv2.warpAffine(
        src_mask, M, (canvas_w, canvas_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    mask_bool = warped_mask > 0   # True aux endroits où img2 a un vrai pixel après warp

    # résultat : img1 en base, img2 au-dessus (uniquement où mask True)
    result = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    # Coller img1 (référence) d'abord
    result[0:h1, 0:w1] = img1

    # Copier img2 transformée seulement là où warped_mask indique présence
    result[mask_bool] = warped_img2[mask_bool]
    return result

result = match_and_place(img1,img2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show()
