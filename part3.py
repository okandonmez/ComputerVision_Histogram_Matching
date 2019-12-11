import glob
import os

import cv2
import numpy as np
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


def resize_image(image, dimension):
    return cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)

def hist_match(imsrc, imtint, nbr_bins = 255):
    if len(imsrc.shape) < 3:
        imsrc = imsrc[:, :, np.newaxis]
        imtint = imtint[:, :, np.newaxis]

    imres = imsrc.copy()
    for d in range(imsrc.shape[2]):
        imhist, bins = np.histogram(imsrc[:, :, d].flatten(), nbr_bins)
        tinthist, bins = np.histogram(imtint[:, :, d].flatten(), nbr_bins)

        cdfsrc = imhist.cumsum()  # cumulative distribution function
        cdfsrc = (255 * cdfsrc / cdfsrc[-1]).astype(np.uint8)  # normalize

        cdftint = tinthist.cumsum()  # cumulative distribution function
        cdftint = (255 * cdftint / cdftint[-1]).astype(np.uint8)  # normalize

        im2 = np.interp(imsrc[:, :, d].flatten(), bins[:-1], cdfsrc)

        im3 = np.interp(im2, cdftint, bins[:-1])

        imres[:, :, d] = im3.reshape((imsrc.shape[0], imsrc.shape[1]))

    try:
        return imres
    except:
        return imres.reshape((imsrc.shape[0], imsrc.shape[1]))

def show_image(img, delayTime=0):
    cv2.imshow("window", img)
    cv2.waitKey(delayTime)


height = 960
width = 540
dim = (height, width)

image_sequence_name = "car-turn"

file_names_images = sorted(glob.glob("JPEGImages/" + image_sequence_name + "/*.jpg"))
image_list = []
for i in range(len(file_names_images)):
    img = cv2.imread(file_names_images[i])
    img = resize_image(img, dim)
    image_list.append(img)


file_names_segmented = sorted(glob.glob("Annotations/" + image_sequence_name + "/*.png"))
segmented_image_list = []
for i in range(len(file_names_segmented)):
    img = cv2.imread(file_names_segmented[i], cv2.IMREAD_GRAYSCALE)
    img = resize_image(img, dim)
    segmented_image_list.append(img)

#imgSource = cv2.imread("JPEGImages/walking/00000.jpg")
#imgSource = cv2.resize(imgSource, dim)

#imgSeg = cv2.imread("Annotations/walking/00000.png", cv2.IMREAD_GRAYSCALE)
#imgSeg = cv2.resize(imgSeg, dim)


'''
lastPixValue = -1
colorCounter = 0
uniqueColorList = []

for i in range(width):
    for j in range(height):
        pixVal = imgSeg[i, j]
        if pixVal == 0 or pixVal == 38 or pixVal == 75:
            if pixVal not in uniqueColorList:
                colorCounter += 1
                lastPixValue = pixVal
                uniqueColorList.append(pixVal)

explain_text = "In the frame, there are/is " + str(colorCounter - 1) + " object"
print(explain_text)
'''

imgTargetRed = cv2.imread("targetImages/target_yellow.jpg")
imgTargetYellow = cv2.imread("targetImages/target_purple.jpg")
imgTargetBlue = cv2.imread("targetImages/target_red.jpg")

result_images = []
for index in range(len(file_names_images)):
    black_coordinates = []
    red_coordinates = []
    green_coordinates = []

    for i in range(width):
        for j in range(height):
            pixVal = segmented_image_list[index][i, j]
            if pixVal == 0:
                black_coordinates.append((i, j))
            if pixVal == 38 or pixVal == 8:
                red_coordinates.append((i, j))
            if pixVal == 75:
                green_coordinates.append((i, j))

    blank_image_black = 255 * np.ones(shape=[width, height, 3], dtype=np.uint8)
    blank_image_red = 255 * np.ones(shape=[width, height, 3], dtype=np.uint8)
    blank_image_green = 255 * np.ones(shape=[width, height, 3], dtype=np.uint8)

    for i in range(len(black_coordinates)):
        blank_image_black[(black_coordinates[i][0]), (black_coordinates[i][1])] = image_list[index][(black_coordinates[i][0]), (black_coordinates[i][1])]

    for i in range(len(red_coordinates)):
        blank_image_red[(red_coordinates[i][0]), (red_coordinates[i][1])] = image_list[index][(red_coordinates[i][0]), (red_coordinates[i][1])]

    for i in range(len(green_coordinates)):
        blank_image_green[(green_coordinates[i][0]), (green_coordinates[i][1])] = image_list[index][(green_coordinates[i][0]), (green_coordinates[i][1])]


    histRed = hist_match(blank_image_red, imgTargetRed)
    histYellow = hist_match(blank_image_black, imgTargetYellow)
    histBlue = hist_match(blank_image_green, imgTargetBlue)

    blank_image_result = 255 * np.ones(shape=[width, height, 3], dtype=np.uint8)

    for i in range(len(black_coordinates)):
        blank_image_result[(black_coordinates[i][0]), (black_coordinates[i][1])] = histYellow[(black_coordinates[i][0]), (black_coordinates[i][1])]

    for i in range(len(red_coordinates)):
        blank_image_result[(red_coordinates[i][0]), (red_coordinates[i][1])] = histRed[(red_coordinates[i][0]), (red_coordinates[i][1])]

    for i in range(len(green_coordinates)):
        blank_image_result[(green_coordinates[i][0]), (green_coordinates[i][1])] = histBlue[(green_coordinates[i][0]), (green_coordinates[i][1])]

    blank_image_result = cv2.cvtColor(blank_image_result, cv2.COLOR_BGR2RGB)
    result_images.append(blank_image_result)


height, width, layers = image_list[0].shape

cv2.destroyAllWindows()
clip = ImageSequenceClip(result_images, fps=25)

try:
    clip.write_videofile('part3Results/part3_' + image_sequence_name + '.mp4', codec="mpeg4")
except:
    os.mkdir('part3Results')
    clip.write_videofile('part3Results/part3_' + image_sequence_name + '.mp4', codec="mpeg4")




