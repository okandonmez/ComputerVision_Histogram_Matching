import glob
import os
import copy
import numpy as np
import cv2
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

############################################# Resize The Image ################################################
def resize_image(image, dimension):
    return cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)

############################################# Concatenation ###################################################
def get_concat_horizontal(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_for_huge(image_sequence_name, dim):
    file_names_images = sorted(glob.glob("JPEGImages/" + image_sequence_name + "/*.jpg"))

    image_list = []
    for i in range(len(file_names_images)):
        img = Image.open(file_names_images[i])
        img = img.resize(dim)
        image_list.append(img)

    im_huge = image_list[0]
    for i in range(len(file_names_images)-1):
        im_huge = get_concat_horizontal(im_huge, image_list[i+1])

    im_huge.save("huge_image.jpg")
    return im_huge

########################################### Histogram Mathcing ################################################
def hist_match(imsrc, imtint, huge_image, nbr_bins = 255):
    if len(imsrc.shape) < 3:
        imsrc = imsrc[:, :, np.newaxis]
        imtint = imtint[:, :, np.newaxis]

    imres = imsrc.copy()
    for d in range(imsrc.shape[2]):
        imhist, bins = np.histogram(huge_image[:, :, d].flatten(), nbr_bins)
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

###############################################################################################################

image_sequence_name = "car-turn"  # setting the variables

width = 960  # Setting the global variables
height = 540
dim = (width, height)

nbr_bins = 255

#get_concat_for_huge(image_sequence_name, dim)

file_names_images = sorted(glob.glob("JPEGImages/" + image_sequence_name + "/*.jpg"))

image_list = []
for i in range(len(file_names_images)):
    img = cv2.imread(file_names_images[i])
    img = resize_image(img, dim)
    image_list.append(img)

#imtarget = cv2.imread("target_blue.jpg")
imtarget = cv2.imread("targetImages/target_purple.jpg")
huge_image = cv2.imread("huge_image.jpg")

imtint = cv2.resize(imtarget, dim)

result_list = []
for image in image_list:
    result = hist_match(image, imtint, huge_image, nbr_bins)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result_list.append(result)
    #cv2.imshow("deneme", result)
    #cv2.waitKey(100)

height, width, layers = result_list[0].shape

cv2.destroyAllWindows()
clip = ImageSequenceClip(result_list, fps=25)

try:
    clip.write_videofile('part2Results/part2_' + image_sequence_name + '.mp4', codec="mpeg4")
except:
    os.mkdir('part2Results')
    clip.write_videofile('part2Results/part2_' + image_sequence_name + '.mp4', codec="mpeg4")
