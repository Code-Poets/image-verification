from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from PIL import Image, ImageChops

def findCropWind(imageA, imageB):
    e1 = cv2.getTickCount()
    smallImage1 = cv2.imread(imageA, cv2.IMREAD_UNCHANGED)
    smallImage = cv2.cvtColor(smallImage1, cv2.COLOR_BGR2GRAY)
    smallImage = cv2.Canny(smallImage, 0, 0)
    (imgH, imgW) = smallImage.shape[:2]

    bigImage1 = cv2.imread(imageB, cv2.IMREAD_UNCHANGED)
    bigImage = cv2.cvtColor(bigImage1, cv2.COLOR_BGR2GRAY)


    edged = cv2.Canny(bigImage, 0, 0)
    cv2.imshow("edged", edged)
    result = cv2.matchTemplate(edged, smallImage, cv2.TM_CCOEFF)
    #print result
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    #clone = np.dstack([edged, edged, edged])
    cv2.rectangle(edged, (maxLoc[0], maxLoc[1]),
                      (maxLoc[0] + imgW, maxLoc[1] + imgH), (0, 0, 255), 2)

    found = (maxVal, maxLoc)

    (_, maxLoc) = found
    (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
    (endX, endY) = (int((maxLoc[0] + imgW)), int((maxLoc[1] + imgH)))


    crop_wind = cv2.imread(imageB, cv2.IMREAD_UNCHANGED)
    cropped_img = crop_wind[startY:endY, startX:endX]
    cv2.rectangle(bigImage1, (startX, startY), (endX, endY), (0, 0, 255), 3)
    cv2.imshow("Image", bigImage1)
    #cv2.imwrite('res1.png', bigImage1)
    cv2.waitKey(0)

    sH, sW = smallImage1.shape[:2]
    cH, cW = cropped_img.shape[:2]
    #print sH, sW
    #print cH, cW
    compare_images(cropped_img, smallImage1, "title")
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print time

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

    diff = cv2.cvtColor(cv2.subtract(imageA, imageB), cv2.COLOR_BGR2GRAY)

    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(diff)
    plt.axis("off")

    plt.show()

def findAll(imgA, imgB):

    img_rgb = cv2.imread(imgB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(imgA,0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    cv2.imwrite('img_gray.png', img_gray)
    cv2.imwrite('template.png', template)
    threshold = 0.9
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 5)

    cv2.imwrite('res.png', img_rgb)

imB = "images/ziemia.png"
imA = "images/z2.png"

findCropWind(imA, imB)

#findAll(imA, imB)