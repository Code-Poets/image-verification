#!/usr/bin/env python3

import sys
from pathlib import Path
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2


class InvalidOption(Exception):
    pass


def validate_options(options):
    if len(options) == 1:
        image1_path = input("Type path to the image You want to compare(rendered localy): ")
        image1 = Path(image1_path)
        print(image1, type(image1))
        print(image1_path, type(image1_path))

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
        return(image1, image2)

    if (len(options) == 2) or (len(options) >= 4):
        sys.exit("Type two directories or none")

    if len(options) == 3:
        image1 = Path(options[1])
        image2 = Path(options[2])
        if (image1.is_file()) and (image2.is_file()):
            image1 = cv2.imread(options[1])
            image2 = cv2.imread(options[2])
        else:
            sys.exit("One or both files does not exists!")
        return(image1, image2)


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


def compare_images(imageA, imageB, title):
    mse = mean_squared_error(cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY))
    structural_similarity = ssim(cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY))
    print("Mean Squared Error between two images: %.2f" % (mse))
    print("Structural Similarity Index between two images: %.10f" % (structural_similarity))

    figure = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.10f" % (mse, structural_similarity))

    ax = figure.add_subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    ax = figure.add_subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    diffrence = cv2.subtract(imageA, imageB)

    ax = figure.add_subplot(1, 3, 3)
    plt.imshow(diffrence)
    plt.axis("off")

    show_comparison = yes_no('Do You want to see images difference?[y/n]')
    if show_comparison:
        plt.show()


def main():
    input_option = validate_options(sys.argv)
    compare_images(input_option[0], input_option[1], "images comparison")


if __name__ == '__main__':
    try:
        main()
    except InvalidOption as exception:
        sys.exit(str(exception))
