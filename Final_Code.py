import numpy as np
import cv2
import csv
import os
from sklearn.cluster import KMeans
from Color_Blindness_Simulation import *
from Color_Difference_Checker import *

# color blind accessibility checker
def cbaChecker(img, imgPath):
    # finds the name of img
    nameOfImg = os.path.basename(imgPath)
    nameOfImg = nameOfImg[:nameOfImg.find(".")]
    # returns all three color blind simulated images in one array, to be parsed later
    threeColorBlindImages = simulation(img)
    print("{}'s results:".format(nameOfImg))

    # charNet is the color difference checker
    # checkResult makes the result more understandable than a single True/False
    protanopiaResult = charNet(img, nameOfImg, threeColorBlindImages[0])
    protanopiaResult = checkResult(protanopiaResult, "Protanopia")

    deuteranopiaResult = charNet(img, nameOfImg, threeColorBlindImages[1])
    deuteranopiaResult = checkResult(deuteranopiaResult, "Deuteranopia")

    tritanopiaResult = charNet(img, nameOfImg, threeColorBlindImages[2])
    tritanopiaResult = checkResult(tritanopiaResult, "Tritanopia")

    # checks in case there no words even detected
    # in that case, it checks if the first one, Protanopia, returns no words detected
    # because if so, there's no use printing "no words detected" 3 times
    if protanopiaResult == "No words detected":
        print(protanopiaResult)
    # else print out the 3 results
    else:
        print(protanopiaResult)
        print(deuteranopiaResult)
        print(tritanopiaResult)

def checkResult(result, type):
    if result == False:
        return "Please increase accessibility for {}". format(type)
    if result == True:
        return "{} filter passed".format(type)
    else:
        return result

# color blindness simulation
for filename in os.listdir("Signs"):
    if filename.endswith("jpg"):
        imgPath = "Signs/{}".format(filename)
        print(filename)
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cbaChecker(img, imgPath)
print("Complete")
#
# imgPath = "Signs/Signs.jpg"
# img = cv2.imread(imgPath)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cbaChecker(img, imgPath)