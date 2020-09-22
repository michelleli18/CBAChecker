import numpy as np
import cv2

def gammaCorrection(pictureArray, type):
    if type == "remove":
        threshold = 0.04045 * 255
    if type == "apply":
        threshold = 0.0031308

    # applies changes to all the smaller than or equal to threshold values
    lessOrEqual = pictureArray[pictureArray <= threshold]
    lessOrEqual = lessOrEqual.astype(float)
    # length of LE as in lessOrEqual
    lengthOfLE = len(lessOrEqual)
    for i in range(lengthOfLE):
        if type == "remove":
            lessOrEqual[i] = (float(lessOrEqual[i]) / float(255))/12.92
        if type == "apply":
            lessOrEqual[i] = (float(lessOrEqual[i]) * 12.92) * 255

    # applies formula to all the greater than values
    greater = pictureArray[pictureArray > threshold]
    greater = greater.astype(float)
    # length of greater
    lengthOfG = len(greater)
    for i in range(lengthOfG):
        if type == "remove":
            greater[i] = (((float(greater[i]) / float(255))+0.055)/1.055)**2.4
        if type == "apply":
            greater[i] = (((float(greater[i]) ** 0.41666) * 1.055) - 0.055) * 255

    # combines all the changes into the original array
    # the order is swapped because when removing, you're turning values smaller
    # so the smaller threshold must be applied first, or else everything would be below threshold
    # since all values will finish with values between 0 and 1, and the threshold is around 10
    if type == "remove":
        pictureArray[pictureArray <= threshold] = lessOrEqual
        pictureArray[pictureArray > threshold] = greater
    # on the opposite hand, the greater array must be applied first when reapplying
    # because the threshold is 0.003 something, if less is applied first, everything will be greater than threshold
    if type == "apply":
        pictureArray[pictureArray > threshold] = greater
        pictureArray[pictureArray <= threshold] = lessOrEqual

    return pictureArray

def rgbToLms(imgArray):
    # rgbToLms is T
    rgbToLms = np.array([[0.31399022, 0.63951294, 0.04649755],
                         [0.15537241, 0.75789446, 0.08670142],
                         [0.01775239, 0.10944209, 0.87256922]])
    # aka return T[r(c), g(c), b(c)] = [l(c), m(c), s(c)]
    # lms
    return rgbToLms @ imgArray

def lmsToProtanopia(lms):
    # protanopiaMatrix is a form of S
    protanopiaMatrix = np.array([[0, 1.05118294, -0.05116099],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    # returns S*lms
    return protanopiaMatrix @ lms

def lmsToDeuteranopia(lms):
    deuteranopiaMatrix = np.array([[1, 0, 0],
                                   [0.9513092, 0, 0.04866992],
                                   [0, 0, 1]])
    return deuteranopiaMatrix @ lms

def lmsToTritanopia(lms):
    tritanopiaMatrix = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [-0.86744736, 1.86727089, 0]])
    return tritanopiaMatrix @ lms

def lmsToRgb(lmsArray):
    # lmsArray is S*lms
    # lmsToRgb is inverse T
    lmsToRgb = np.array([[5.47221206, -4.6419601, 0.16963708],
                          [-1.1252419, 2.29317094, -0.1678952],
                          [0.02980165, -0.19318073, 1.16364789]])
    # returns inverseT*(S*lms)
    return lmsToRgb @ lmsArray

def typeSimulation(lms, type):
    # apply S to lms
    if type == "protanopia":
        simulationArray = np.apply_along_axis(lmsToProtanopia, 2, lms)
        # print("lms to protanopia complete")
    if type == "deuteranopia":
        simulationArray = np.apply_along_axis(lmsToDeuteranopia, 2, lms)
        # print("lms to deuteranopia complete")
    if type == "tritanopia":
        simulationArray = np.apply_along_axis(lmsToTritanopia, 2, lms)
        # print("lms to tritanopia complete")

    # apply inverse T to S*lms
    rgbArray = np.apply_along_axis(lmsToRgb, 2, simulationArray)
    # print("simulation to rgb complete")

    # gamma correct
    gammaCorrectedArray = gammaCorrection(rgbArray, "apply")
    # print("rgb gamma applied complete")

    # rescale
    rescaledRGB = cv2.convertScaleAbs(gammaCorrectedArray)
    # print("rescaled complete")

    # convert color
    finalImg = cv2.cvtColor(rescaledRGB, cv2.COLOR_RGB2BGR)
    return finalImg

def simulation(img):
    # make to floats, or else all values would turn to 0
    imgArray = img.astype(float)

    # remove gamma correction
    gammaRemovedArray = gammaCorrection(imgArray, "remove")
    # print("gamma removed complete")

    # turn into lms
    lmsArray = np.apply_along_axis(rgbToLms, 2, gammaRemovedArray)
    # print("gamma removed to lms complete")

    # protanopia simulation - complete as of 8.24.2020
    # simulation step
    protanopia = typeSimulation(lmsArray, "protanopia")
    # print("Protanopia simulation complete")
    deuteranopia = typeSimulation(lmsArray, "deuteranopia")
    # print("Deuteranopia simulation complete")
    tritanopia = typeSimulation(lmsArray, "tritanopia")
    # print("Tritanopia simulation complete")
    print("Color blind image simulation complete")

    # # show original image and three simulated images
    # cv2.imshow("protanopia", protanopia)
    # cv2.imshow("deuteranopia", deuteranopia)
    # cv2.imshow("tritanopia", tritanopia)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imshow("original", img)
    # cv2.waitKey(0)
    return (protanopia, deuteranopia, tritanopia)
