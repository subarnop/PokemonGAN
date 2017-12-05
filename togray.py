#to convert
import cv2
import os
import glob

if not os.path.exists('Pokemon_Grey/'):
    os.makedirs('Pokemon_Grey/')
    
count =000
for img in glob.glob("Pokemon/*.jpg"):
    image = cv2.imread(img)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out = str('Pokemon_Grey/pic_'+str(count)+'.png')
    cv2.imwrite(out,gray_image)
    count += 1
