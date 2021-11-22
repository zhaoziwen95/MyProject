import cv2  # open CV包
def thresholdBGR(img):
    """find the water
        return a picture"""
    img = img
    b, g, r = cv2.split(img)
    # cv2.imshow("b", b)
    # cv2.imshow("g", g)
    # cv2.imshow("r", r)
    # cv2.waitKey(0)

    #########################################################   b_Min < b < b_Max
    ret, b_min = cv2.threshold(b, 60, 255, cv2.THRESH_BINARY)
    # cv2.imshow('b_min', b_min)
    # cv2.waitKey(0)
    ret, b_max = cv2.threshold(b, 120, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('b_max', b_max)
    # cv2.waitKey(0)
    b_min_max = cv2.bitwise_and(b_min, b_min, mask=b_max)
    # cv2.imshow('b_min_max', b_min_max)
    # cv2.waitKey(0)

    #########################################################   g_Min < g < g_Max
    ret, g_min = cv2.threshold(b, 60, 255, cv2.THRESH_BINARY)
    # cv2.imshow('g_min', g_min)
    # cv2.waitKey(0)
    ret, g_max = cv2.threshold(b, 115, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('g_max', g_max)
    # cv2.waitKey(0)
    g_min_max = cv2.bitwise_and(g_min, g_min, mask=g_max)
    # cv2.imshow('g_min_max', g_min_max)
    # cv2.waitKey(0)

    #########################################################   r_Min < r < r_Max

    ret, r_min = cv2.threshold(r, -1, 255, cv2.THRESH_BINARY)
    # cv2.imshow('r_min', r_min)
    # cv2.waitKey(0)
    ret, r_max = cv2.threshold(r, 40, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('r_max', r_max)
    # cv2.waitKey(0)
    r_min_max = cv2.bitwise_and(r_min, r_min, mask=r_max)
    # cv2.imshow('r_min_max', r_min_max)
    # cv2.waitKey(0)

    #########################################################   b and g and r
    bitwiseAnd = cv2.bitwise_and(b_min_max, b_min_max, mask=g_min_max)
    # cv2.imshow('bitwiseAnd', bitwiseAnd)
    # cv2.waitKey(0)
    bitwiseAnd = cv2.bitwise_and(bitwiseAnd, bitwiseAnd, mask=r_min_max)
    # cv2.imshow('bitwiseAnd', bitwiseAnd)
    # cv2.waitKey(1)
    return bitwiseAnd