from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from PIL import Image, ImageChops

def findCropWind(imageA, imageB):

    smallImage1 = cv2.imread(imageA)
    smallImage = cv2.cvtColor(smallImage1, cv2.COLOR_BGR2GRAY)
    smallImage = cv2.Canny(smallImage, 0, 0)
    (imgH, imgW) = smallImage.shape[:2]

    bigImage = cv2.imread(imageB)
    bigImage = cv2.cvtColor(bigImage, cv2.COLOR_BGR2GRAY)


    edged = cv2.Canny(bigImage, 0, 0)
    cv2.imshow("edge", edged)
    result = cv2.matchTemplate(edged, smallImage, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    clone = np.dstack([edged, edged, edged])
    cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                      (maxLoc[0] + imgW, maxLoc[1] + imgH), (0, 0, 255), 2)

    found = (maxVal, maxLoc)

    (_, maxLoc) = found
    (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
    (endX, endY) = (int((maxLoc[0] + imgW)), int((maxLoc[1] + imgH)))


    crop_wind = cv2.imread(imageB)
    cropped_img = crop_wind[startY:endY, startX:endX]


    cv2.rectangle(bigImage, (startX, startY), (endX, endY), (0, 0, 255), 3)
    cv2.imshow("Image", bigImage)
    cv2.waitKey(0)

    sH, sW = smallImage1.shape[:2]
    cH, cW = cropped_img.shape[:2]
    print sH, sW
    print cH, cW
    cv2.imshow("image1", smallImage1)
    cv2.imshow("image", cropped_img)
    cv2.waitKey(0)
    compare_images(smallImage1, cropped_img, "title")





def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_images(imageA, imageB, title):
    m = mse(cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY))
    s = ssim(cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY))

    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.10f" % (m, s))

    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(imageA)
    plt.axis("off")

    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(imageB)
    plt.axis("off")

    diff = cv2.subtract(imageA, imageB)

    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(diff)
    plt.axis("off")

    plt.show()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def fromArrayToImage(img):
    return Image.fromarray(np.uint8(img))

imB = "cat.jpg"
imA = "cat3.jpg"

findCropWind(imA, imB)



