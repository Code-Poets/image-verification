#!/usr/bin/env python3

import sys
from pathlib import Path
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random


class InvalidOption(Exception):
    pass


def validate_options(options):
    if len(options) == 1:
        image1_path = input("Type path to the image You want to compare(rendered localy): ")
        image1 = Path(image1_path)
        if image1.is_file():
            image1 = cv2.imread(image1_path)
        else:
            sys.exit("No such file or directory!")
        image2_path = input("Type path to the image You want to compare with the first one: ")
        image2 = Path(image2_path)
        if image2.is_file():
            image2 = cv2.imread(image2_path)
        else:
            sys.exit("No such file or directory!")
        sizeH = int(input("Type the height of the crop image you want to compare: "))
        if sizeH == 0:
            sys.exit("You can't type 0 as a height")
        sizeW = int(input("Type the width of the crop image you want to compare: "))
        if sizeW == 0:
            sys.exit("You can't type 0 as a width")

        return(image1, image2, sizeH, sizeW)

    if (len(options) <= 4) or (len(options) >= 6):
        sys.exit("Type two directories and sizes of crop window or none")

    if len(options) == 5:
        image1 = Path(options[1])
        image2 = Path(options[2])
        sizeH = int(options[3])
        sizeW = int(options[4])


        if (image1.is_file()) and (image2.is_file()):
            image1 = cv2.imread(options[1])
            image2 = cv2.imread(options[2])
        else:
            sys.exit("One or both files does not exists!")
        return(image1, image2, sizeH, sizeW)


def yes_no(answer):
    yes = set(['yes','y', 'ye', ''])
    no = set(['no','n'])

    while True:
        choice = input(answer).lower()
        if choice in yes:
           return True
        elif choice in no:
           return False
        else:
           print ("Please respond with 'yes' or 'no'\n")


def mean_squared_error(imageA, imageB):
    mse = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    mse /= float(imageA.shape[0] * imageA.shape[1])
    return mse


def compare_images(imageA, imageB):
    meanSquaredError = mean_squared_error(cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY))
    structualSimilarity = ssim(cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY))
    return structualSimilarity, meanSquaredError

def findRandomWindow(imageA, imageB, sizeH, sizeW):
    (imgH, imgW) = imageA.shape[:2]
    if sizeH > imgH or sizeW > imgW:
        print("Too small image for this size of crop window")
        exit()
    randomH = random.randrange(0, imgH-sizeH)
    randomW = random.randrange(0, imgW-sizeW)
    croppedwindow = imageA[randomH:randomH+sizeH, randomW:randomW+sizeW]
    findCropWindow(croppedwindow, imageB, randomH, randomW, sizeH, sizeW)

def findCropWindow(imageA, imageB, cordH, cordW, sizeH, sizeW):
    templateToCompare = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    templateToCompare = cv2.Canny(templateToCompare, 0, 0)
    (imgH, imgW) = templateToCompare.shape[:2]
    if imgH*imgW < 50:
        print ("Too small image. Upload bigger image!")
        exit()

    cropped_img = imageB[cordH:cordH+sizeH, cordW:cordW+sizeW]
    imagesCorrelation = compare_histograms(cropped_img, imageA)
    structualSimilarity, meanSquaredError = compare_images(cropped_img, imageA)
    if (structualSimilarity > 0.99 and imagesCorrelation > 0.99):
        print ("Image is ok. Structural Similarity is: %.10f. Mean Squared Error is: %.10f. "
               "Images Correlation is: %.10f. " %
               (structualSimilarity, meanSquaredError, imagesCorrelation))
    else:
        print ("Can't find similarity in this image. Upload bigger image")

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



def main():
    input_option = validate_options(sys.argv)
    findRandomWindow(input_option[0], input_option[1], input_option[2], input_option[3])


if __name__ == '__main__':
    try:
        main()
    except InvalidOption as exception:
        sys.exit(str(exception))

