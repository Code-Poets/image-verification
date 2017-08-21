#!/usr/bin/env python

import numpy as np
import cv2
from skimage.measure import structural_similarity as ssim
import argparse
import random

def findCropWindow(imageA, imageB, cordH, cordW, sizeH, sizeW):
    templateToCompare = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    templateToCompare = cv2.Canny(templateToCompare, 0, 0)
    (imgH, imgW) = templateToCompare.shape[:2]
    if imgH*imgW < 50:
        print ("Too small image. Upload bigger image!")
        exit()

    #if we have big image and have to cut it from inside
    image = cv2.imread(imageB, cv2.IMREAD_UNCHANGED)
    cropped_img = image[cordH:cordH+sizeH, cordW:cordW+sizeW]
    #if we have exactly resolution of image and we dont have to cut it out
    #from bigger image
    #image = cv2.imread(imageB, cv.IMREAD_UNCHANGED)
    cv2.imshow("cropp", cropped_img)
    cv2.waitKey(0)
    imagesCorrelation = compare_histograms(cropped_img, imageA)
    structualSimilarity, meanSquaredError = compare_images(cropped_img, imageA)
    if (structualSimilarity > 0.99 and imagesCorrelation > 0.99):
        print ("Image is ok. Structural Similarity is: %.10f. Mean Squared Error is: %.10f. "
               "Images Correlation is: %.10f. " %
               (structualSimilarity, meanSquaredError, imagesCorrelation))
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


def compare_histograms(imageA, imageB):

    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    hist_item = 0
    hist_item1 = 0
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([imageA], [ch], None, [256], [0, 255])
        hist_item1 = cv2.calcHist([imageB], [ch], None, [256], [0, 255])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(hist_item1, hist_item1, 0, 255, cv2.NORM_MINMAX)
    result = cv2.compareHist(hist_item, hist_item1, cv2.HISTCMP_CORREL)
    return result

def findRandomWindow(imageA, sizeH, sizeW):
    original = cv2.imread(imageA, cv2.IMREAD_UNCHANGED)
    (imgH, imgW) = original.shape[:2]
    randomH = random.randrange(0, imgH-sizeH)
    randomW = random.randrange(0, imgW-sizeW)
    croppedwindow = original[randomH:randomH+sizeH, randomW:randomW+sizeW]
    return croppedwindow, randomH, randomW, sizeH, sizeW

photo, cordH, cordW, sizeH, sizeW = findRandomWindow("images/cat.png", 10, 10)
findCropWindow(photo, "images/cat.png", cordH, cordW, sizeH, sizeW)

#parser = argparse.ArgumentParser(description="Image detection")
#parser.add_argument("image", nargs=2, help="Upload images to compare")
#args = parser.parse_args()



#def main():
#    findCropWindow(args.image[0], args.image[1])
#
#if __name__ == '__main__':
#    main()