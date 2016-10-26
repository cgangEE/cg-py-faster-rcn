#!/usr/bin/env python
import matplotlib.pyplot as plt
import cv2
import os

def getFloat(s):
    if s.find('(') != -1:
        s = s.split('(')[1]
    return max(0.0 , float(s))

def getPoint(point):
    point = point.split(',')
    x = getFloat(point[0])
    y = getFloat(point[1])
    return x, y

def getBoxFromSeg(seg):
    points = seg.split(')')
    pointsNum = len(points)

    x = 1e5
    y = 1e5
    x1 = 0.0
    y1 = 0.0
    
    for i in xrange(pointsNum - 1):
        xi, yi = getPoint(points[i])
        if xi < x:
            x = xi
        if yi < y:
            y = yi
        if xi > x1:
            x1 = xi
        if yi > y1:
            y1 = yi
    return x, y, x1, y1

def getBoundingBox(fileName):
    f = open(fileName, 'r')

    ret = []
    for line in f:
        line = line.split('[')
        boxNum = len(line)

        boxes = []
        for i in xrange(1, boxNum):
            x, y, x1, y1 = getBoxFromSeg(line[i])
            if x1 - x >= 0 and y1 - y >= 0:
                boxes.append((x, y, x1, y1))

        ret.append((line[0], boxes))

    return ret


if __name__ == '__main__':
    getBoundingBox('tattoo_annotations.txt')

