#!/usr/bin/env python3

import sys
from pathlib import Path
import random
import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
import csv

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

def mean_squared_error(imageA, imageB):
    mse = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    mse /= float(imageA.shape[0] * imageA.shape[1])
    return mse


def compare_images(imageA, imageB):
    structualSimilarity = 0
    meanSquaredError = mean_squared_error(cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY))
    structualSim = ssim(cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY))
    return structualSim, meanSquaredError

def compare_images_canny(imageA, imageB):
    structualSim = ssim(imageA, imageB)
    return structualSim


def write_in_file_ssim(ssim, count, cropWindowSize):
    with open('data/ssim.csv', 'a') as csvfile:
        fieldnames = [cropWindowSize]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if (count == 0):
            writer.writeheader()
        writer.writerow({cropWindowSize : ssim})

def write_in_file_ssim_canny(ssim, count, cropWindowSize):
    with open('data/ssim_canny.csv', 'a') as csvfile:
        fieldnames = [cropWindowSize]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if (count == 0):
            writer.writeheader()
        writer.writerow({cropWindowSize : ssim})

def write_in_file_mse(mse, count, cropWindowSize):
    with open('data/mse.csv', 'a') as csvfile:
        fieldnames = [cropWindowSize]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if (count == 0):
            writer.writeheader()
        writer.writerow({cropWindowSize : mse})

def write_in_file_mse_canny(mse, count, cropWindowSize):
    with open('data/mse_canny.csv', 'a') as csvfile:
        fieldnames = [cropWindowSize]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if (count == 0):
            writer.writeheader()
        writer.writerow({cropWindowSize : mse})        

def write_in_file_corr(corr, count, cropWindowSize):
    with open('data/corr.csv', 'a') as csvfile:
        fieldnames = [cropWindowSize]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if (count == 0):
            writer.writeheader()
        writer.writerow({cropWindowSize : corr})

def write_in_file_cord(height, width, count, cropWindowSize):
    with open('data/cord.csv', 'a') as csvfile:
        fieldnames = [cropWindowSize]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        cord = height, width
        if (count == 0):
            writer.writeheader()
        writer.writerow({cropWindowSize : cord})

def check_all_pixels(img1, img2, sizeH, sizeW):
    e1 = cv2.getTickCount()
    imageA = cv2.imread(img1)
    imageB = cv2.imread(img2)
    (imgAH, imgAW) = imageA.shape[:2]
    (imgBH, imgBW) = imageB.shape[:2]
    imageAa = cv2.Canny(imageA, 0, 0)
    imageBb = cv2.Canny(imageB, 0, 0)
    counter = 0
    cropWindowSize = "WIELKOSC OKNA: ", sizeH, sizeW
    ssim_norm_counter = 0
    ssim_canny_counter = 0
    for H in range(imgAH - sizeH):
        for W in range(imgAW - sizeW):
            cropWind = imageA[H:H + sizeH, W:W + sizeW]
            cropWind_canny = imageAa[H:H + sizeH, W:W + sizeW]
            cropWind2 = imageB[H:H + sizeH, W:W + sizeW]
            cropWind2_canny = imageBb[H:H + sizeH, W:W + sizeW]
            imgCorr = compare_histograms(cropWind, cropWind2)
            SSIM_normal, MSE_normal = compare_images(cropWind, cropWind2)
            ssim_norm_counter
            if SSIM_normal == 1:
                ssim_norm_counter += 1
            SSIM_canny = compare_images_canny(cropWind_canny, cropWind2_canny)
            if SSIM_canny == 1:
                ssim_canny_counter += 1
            e2 = cv2.getTickCount()
            time = (e2 - e1)/ cv2.getTickFrequency()
            print("Structural Similarity is: %.10f. Structural Similarity canny image is: %.10f  Time: %.5f Cropped Window checked: %.f SSIM_NORM: %.0f SSIM_CANNY %.0f" % (SSIM_normal, SSIM_canny, time, counter, ssim_norm_counter, ssim_canny_counter))
            write_in_file_ssim(SSIM_normal, counter, cropWindowSize)
            write_in_file_ssim_canny(SSIM_canny, counter, cropWindowSize)
            write_in_file_mse(MSE_normal, counter, cropWindowSize)
            write_in_file_corr(imgCorr, counter, cropWindowSize)
            write_in_file_cord(H, W, counter, cropWindowSize)
            counter += 1

def check_random_pic(img1, img2, sizeH, sizeW, number_of_times):
    e1 = cv2.getTickCount()
    imageA = cv2.imread(img1)
    imageB = cv2.imread(img2)
    (imgAH, imgAW) = imageA.shape[:2]
    (imgBH, imgBW) = imageB.shape[:2]
    imageAa = cv2.Canny(imageA, 0, 0)
    imageBb = cv2.Canny(imageB, 0, 0)
    counter = 0
    cropWindowSize = "WIELKOSC OKNA: ", sizeH, sizeW
    ssim_norm_counter = 0
    ssim_canny_counter = 0
    while counter < number_of_times:
        H = random.randrange(0, imgAH - sizeH)
        W = random.randrange(0, imgAW - sizeW)
        cropWind = imageA[H:H + sizeH, W:W + sizeW]
        cropWind_canny = imageAa[H:H + sizeH, W:W + sizeW]
        cropWind2 = imageB[H:H + sizeH, W:W + sizeW]
        cropWind2_canny = imageBb[H:H + sizeH, W:W + sizeW]
        imgCorr = compare_histograms(cropWind, cropWind2)
        SSIM_normal, MSE_normal = compare_images(cropWind, cropWind2)
        ssim_norm_counter
        if SSIM_normal == 1:
            ssim_norm_counter += 1
        SSIM_canny = compare_images_canny(cropWind_canny, cropWind2_canny)
        if SSIM_canny == 1:
            ssim_canny_counter += 1
        e2 = cv2.getTickCount()
        time = (e2 - e1)/ cv2.getTickFrequency()
        print("Structural Similarity is: %.10f. Structural Similarity canny image is: %.10f  Time: %.5f Cropped Window checked: %.f SSIM_NORM: %.0f SSIM_CANNY %.0f" % (SSIM_normal, SSIM_canny, time, counter, ssim_norm_counter, ssim_canny_counter))
        write_in_file_ssim(SSIM_normal, counter, cropWindowSize)
        write_in_file_ssim_canny(SSIM_canny, counter, cropWindowSize)
        write_in_file_mse(MSE_normal, counter, cropWindowSize)
        write_in_file_corr(imgCorr, counter, cropWindowSize)
        write_in_file_cord(H, W, counter, cropWindowSize)
        counter += 1
