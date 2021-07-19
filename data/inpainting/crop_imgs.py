import numpy as np
import cv2

for i in range(6):
    img = cv2.imread(f"hair_{i}.png")

    size = np.min(np.array(img.shape[:2]))
    center = np.array(img.shape[:2]) / 2
    x = center[1] - size/2
    y = center[0] - size/2

    fx = img.shape[0] / 256
    newsize = img.shape[1] * (1/fx)
    newsize = int(newsize - (newsize % 32))
    oldsize = int(fx * newsize)

    cr_w = int(0.5 * (img.shape[1] - oldsize))
    crop_img = img[:,cr_w:oldsize+cr_w]
    #crop_img = img[int(y):int(y+size), int(x):int(x+size)]
    crop_img = cv2.resize(crop_img, (newsize, 256))
    # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(f"hair_{i}_res.png", crop_img)
    cv2.imshow("img", crop_img)
    cv2.waitKey(0)
