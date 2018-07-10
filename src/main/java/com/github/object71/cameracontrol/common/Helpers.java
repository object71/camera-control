/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol.common;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;

/**
 *
 * @author hristo
 */
public class Helpers {

    public static boolean isRectangleInMatrix(Rect rectangle, Mat matrix) {
        throw new UnsupportedOperationException();
    }

    public static boolean isPointInMatrix(Point rectangle, Mat matrix) {
        throw new UnsupportedOperationException();
    }

    public static Mat getMatrixMagnitude(Mat matrixX, Mat matrixY) {
        throw new UnsupportedOperationException();
    }

    public static double computeDynamicTreshold(Mat mat, double standardDeviationFactor) {
        throw new UnsupportedOperationException();
    }

    public static Point centerOfRect(Rect rectangle) {
        return new Point(rectangle.x + (rectangle.width / 2.0), rectangle.y + (rectangle.height / 2.0));
    }

    public static double distanceBetweenPoints(Point a, Point b) {
        return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
    }
}
