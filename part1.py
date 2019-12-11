import cv2
import glob
from moviepy.editor import *
import os


############################################# Resize The Image ################################################
def resize_image(image, dimension):
    return cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)

###############################################################################################################

############################################# Setting Initial Assingments #####################################
image_sequence_name = "walking"

width = 960
height = 540
dim = (width, height)

###############################################################################################################

file_names_images = sorted(glob.glob("JPEGImages/" + image_sequence_name + "/*.jpg"))
image_list = []                           # not segmented images
for i in range(len(file_names_images)):
    img = cv2.imread(file_names_images[i])
    img = resize_image(img, dim)
    image_list.append(img)

file_names_segmented = sorted(glob.glob("Annotations/" + image_sequence_name + "/*.png"))

segmented_image_list = []                 # segmented images
for i in range(len(file_names_segmented)):
    img = cv2.imread(file_names_segmented[i], cv2.IMREAD_GRAYSCALE)
    img = resize_image(img, dim)
    segmented_image_list.append(img)

for x in range(len(file_names_images)):
    for i in range(width):
        for j in range(height):
            colorDetected = segmented_image_list[x][j, i]
            if colorDetected == 38:                     # detected red object
                color = image_list[x][j, i]
                # color[0] = 255
                # color[1] = 255
                # color[2] = 255
                # color[0] = color[0] + 90
                color[1] = color[1] - ((color[1] * 75) / 100);      # decrease green channel
                color[2] = color[2] - ((color[2] * 75) / 100);      # decrease red   channel
                image_list[x][j, i] = color
    image_list[x] = cv2.cvtColor(image_list[x], cv2.COLOR_BGR2RGB)

#for i in range(len(image_list)):
#    cv2.imshow("deneme", image_list[i]);
#    cv2.waitKey(200)

height, width, layers = image_list[0].shape

cv2.destroyAllWindows()
clip = ImageSequenceClip(image_list, fps=25)

try:
    clip.write_videofile('part1Results/part1_' + image_sequence_name + '.mp4', codec="mpeg4")
except:
    os.mkdir('part1Results')
    clip.write_videofile('part1Results/part1_' + image_sequence_name + '.mp4', codec="mpeg4")
