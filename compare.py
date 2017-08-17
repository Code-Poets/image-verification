from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from PIL import Image, ImageChops

def findCropWind(imageA, imageB):

    smallImage1 = cv2.imread(imageA)
    smallImage = cv2.cvtColor(smallImage1, cv2.COLOR_BGR2GRAY)
    smallImage = cv2.Canny(smallImage, 50, 200)
    (imgH, imgW) = smallImage.shape[:2]
    cv2.imshow("Image", smallImage)

    bigImage = cv2.imread(imageB)
    bigImage = cv2.cvtColor(bigImage, cv2.COLOR_BGR2GRAY)


    edged = cv2.Canny(bigImage, 50, 200)
    result = cv2.matchTemplate(edged, smallImage, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    clone = np.dstack([edged, edged, edged])
    cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                      (maxLoc[0] + imgW, maxLoc[1] + imgH), (0, 0, 255), 2)
    cv2.imshow("Visualize", clone)
    cv2.waitKey(0)

    found = (maxVal, maxLoc)

    (_, maxLoc) = found
    (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
    (endX, endY) = (int((maxLoc[0] + imgW)), int((maxLoc[1] + imgH)))


    crop_wind = Image.open(imageB)
    area = (startX, startY, endX, endY)
    cropped_img = crop_wind.crop(area)
    cropped_img.show()

    cv2.rectangle(bigImage, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Image", bigImage)
    cv2.waitKey(0)

    cropped_img = np.array(cropped_img)

    compare_images(cropped_img, smallImage1, "title")





def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_images(imageA, imageB, title):
    m = mse(rgb2gray(imageA), rgb2gray(imageB))
    s = ssim(rgb2gray(imageA), rgb2gray(imageB))

    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.10f" % (m, s))

    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(imageA)
    plt.axis("off")

    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(imageB)
    plt.axis("off")

    diff = ImageChops.difference(fromArrayToImage(imageA), fromArrayToImage(imageB))

    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(diff)
    plt.axis("off")

    plt.show()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def fromArrayToImage(img):
    return Image.fromarray(np.uint8(img))

imB = "cat.jpg"
imA = "cat2.jpg"

findCropWind(imA, imB)



