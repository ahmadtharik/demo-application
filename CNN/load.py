import os
import cv2
from PIL import Image
import numpy as np
# 

# 
data=[]
labels=[]
# 
# ----------------
# LABELS
# Cat 0
# Dog 1
# Monkey 2
# Parrot 3
# Elephant 4
# Bear 5
# ----------------

# Cat 0
cats = os.listdir(os.getcwd() + "/CNN/data/cats")
for x in cats:
    imag=cv2.imread(os.getcwd() + "/CNN/data/cats/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(0)

# Dog 1
dogs = os.listdir(os.getcwd() + "/CNN/data/dogs/")
for x in dogs:
    imag=cv2.imread(os.getcwd() + "/CNN/data/dogs/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)



animals=np.array(data)
labels=np.array(labels)
# 
np.save("animals",animals)
np.save("labels",labels)