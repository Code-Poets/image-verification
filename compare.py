#!/usr/bin/env python

import numpy as np
import cv2
from skimage.measure import structural_similarity as ssim
import argparse

def findCropWindow(imageA, imageB):
    template = cv2.imread(imageA, cv2.IMREAD_UNCHANGED)
    templateToCompare = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    templateToCompare = cv2.Canny(templateToCompare, 0, 0)
    (imgH, imgW) = templateToCompare.shape[:2]
    if imgH*imgW < 50:
        print ("Too small image. Upload bigger image!")
        exit()
    bigImage = cv2.imread(imageB, cv2.IMREAD_UNCHANGED)
    bigImage = cv2.cvtColor(bigImage, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(bigImage, 0, 0)
    result = cv2.matchTemplate(edged, templateToCompare, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    print cv2.minMaxLoc(result)
    (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
    (endX, endY) = (int((maxLoc[0] + imgW)), int((maxLoc[1] + imgH)))
    crop_wind = cv2.imread(imageB, cv2.IMREAD_UNCHANGED)
    cropped_img = crop_wind[startY:endY, startX:endX]
    cv2.rectangle(bigImage, (startX, startY), (endX, endY), (0, 0, 255), 3)
    similarImages = findAll(imageA, imageB)
    structualSimilarity, meanSquaredError = compare_images(cropped_img, template)
    if ((structualSimilarity > 0.99 and similarImages > 0) or structualSimilarity == 1):
        print ("Image is ok. Structural Similarity is: %.3f. Mean Squared Error is: %.4f. "
               "Similar Images found in big Image: %.0f" % (structualSimilarity, meanSquaredError, similarImages))
    else:
        print ("Can't find similarity in this image. Upload bigger image")

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_images(imageA, imageB):
    meanSquaredError = mse(cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY))
    structualSimilarity = ssim(cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY))
    return structualSimilarity, meanSquaredError

def findAll(imgA, imgB):

    bigImage = cv2.imread(imgB)
    bigImage = cv2.cvtColor(bigImage, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(imgA,0)
    w, h = template.shape[::-1]
    result = cv2.matchTemplate(bigImage, template, cv2.TM_CCOEFF_NORMED)
    cv2.imwrite('img_gray.png', bigImage)
    cv2.imwrite('template.png', template)
    threshold = 0.95
    loc = np.where(result >= threshold)
    similarImages = 0
    for pt in zip(*loc[::-1]):
        cv2.rectangle(bigImage, pt, (pt[0] + w, pt[1] + h), (0,255,255), 1)
        similarImages +=1
        if similarImages > 1000:
            print("Too small image. Upload bigger image!")
            exit()
    cv2.imwrite('res.png', bigImage)
    return similarImages

parser = argparse.ArgumentParser(description="Image detection")
parser.add_argument("image", nargs=2, help="Upload images to compare")
args = parser.parse_args()

def main():
    findCropWindow(args.image[0], args.image[1])

if __name__ == '__main__':
    main()
