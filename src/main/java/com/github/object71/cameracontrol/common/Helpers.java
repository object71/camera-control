/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol.common;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Point;
import org.opencv.core.Rect;

/**
 *
 * @author hristo
 */
public class Helpers {

    public static boolean isRectangleInMatrix(Rect rectangle, Mat matrix) {
        return rectangle.x > 0 && rectangle.y > 0 && rectangle.x + rectangle.width < matrix.cols()
                && rectangle.y + rectangle.height < matrix.rows();
    }

    public static boolean isPointInMatrix(Point point, Mat matrix) {
        return point.x >= 0 && point.x < matrix.cols() && point.y >= 0 && point.y < matrix.rows();
    }

    public static boolean isPointInMatrix(Point point, int rows, int cols) {
        return point.x >= 0 && point.x < cols && point.y >= 0 && point.y < rows;
    }

    public static Mat getMatrixMagnitude(Mat matrixX, Mat matrixY) {
        Mat magnitudeMatrix = new Mat(matrixX.rows(), matrixX.cols(), CvType.CV_64F);
        for (int y = 0; y < matrixX.rows(); y++) {
            for (int x = 0; x < matrixX.cols(); x++) {
                double valueX = matrixX.get(y, x)[0];
                double valueY = matrixY.get(y, x)[0];
                double magnitudeValue = Math.sqrt(Math.pow(valueX, 2) + Math.pow(valueY, 2));
                magnitudeMatrix.put(y, x, magnitudeValue);
            }
        }
        return magnitudeMatrix;
    }

    public static double computeDynamicTreshold(Mat matrix, double standardDeviationFactor) {
        MatOfDouble standartMagnitudeGradient = new MatOfDouble();
        MatOfDouble meanMagnitudeGradient = new MatOfDouble();
        Core.meanStdDev(matrix, meanMagnitudeGradient, standartMagnitudeGradient);

        double standartDeviation = standartMagnitudeGradient.toArray()[0] / Math.sqrt(matrix.rows() * matrix.cols());
        double result = standartDeviation * standardDeviationFactor + meanMagnitudeGradient.toArray()[0];

        return result;
    }

    public static Point centerOfRect(Rect rectangle) {
        return new Point(rectangle.x + (rectangle.width / 2.0), rectangle.y + (rectangle.height / 2.0));
    }

    public static double centerOfRectXAxis(Rect rectangle) {
        return rectangle.x + (rectangle.width / 2.0);
    }

    public static double distanceBetweenPoints(Point a, Point b) {
        return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
    }

    public static double distanceBetweenValues(double a, double b) {
        return a >= b ? a - b : b - a;
    }

    public static Mat computeMatXGradient(Mat input) throws IllegalArgumentException {
        if (input.cols() < 3) {
            throw new IllegalArgumentException("Invalid matrix size");
        }
        Mat output = new Mat(input.rows(), input.cols(), CvType.CV_64F);

        for (int y = 0; y < input.rows(); y++) {
            // Mat inputRow = input.row(y);

            // set the border gradients
            output.put(y, 0, input.get(y, 1)[0] - input.get(y, 0)[0]);
            output.put(y, output.cols() - 1, input.get(y, input.cols() - 1)[0] - input.get(y, input.cols() - 2)[0]);

            for (int x = 1; x < input.cols() - 1; x++) {
                output.put(y, x, input.get(y, x + 1)[0] - input.get(y, x - 1)[0]);
            }
        }

        return output;
    }

    public static Mat computeMatYGradient(Mat input) throws IllegalArgumentException {
        if (input.rows() < 3) {
            throw new IllegalArgumentException("Invalid matrix size");
        }
        Mat output = new Mat(input.rows(), input.cols(), CvType.CV_64F);

        for (int x = 0; x < input.rows(); x++) {
            // Mat inputRow = input.row(y);

            // set the border gradients
            output.put(0, x, input.get(1, x)[0] - input.get(0, x)[0]);
            output.put(output.rows() - 1, x, input.get(input.rows() - 1, x)[0] - input.get(input.cols() - 2, x)[0]);

            for (int y = 1; y < input.cols() - 1; y++) {
                output.put(y, x, input.get(y + 1, x)[0] - input.get(y - 1, x)[0]);
            }
        }

        return output;
    }

    public static void bright(Mat matrix) {
        for(int y = 0; y < matrix.rows(); y++) {
            for(int x = 0; x < matrix.cols(); x++) {
                matrix.put(y, x, matrix.get(y, x)[0] + 50);
            }
        }
    }

    public static void possibleCenterFormula(int x, int y, Mat weight, double gx, double gy, Mat output) {

        for (int cy = 0; cy < output.rows(); cy++) {
            for (int cx = 0; cx < output.cols(); cx++) {
                if (x == cx && y == cy) {
                    continue;
                }

                double dx = x - cx;
                double dy = y - cy;
                double magnitude = Math.sqrt(Math.pow(dx, 2) + Math.pow(dy, 2));
                dx = dx / magnitude;
                dy = dy / magnitude;

                // 0 or positive
                double dotProduct = Math.max(0, dx * gx + dy * gy);

                double currentValue = output.get(cy, cx)[0];

                if (Constants.enableWeight) {
                    double weightValue = weight.get(cy, cx)[0];
                    output.put(cy, cx,
                            currentValue + (Math.pow(dotProduct, 2) * (weightValue / Constants.weightDivisor)));
                } else {
                    output.put(cy, cx, currentValue + Math.pow(dotProduct, 2));
                }
            }
        }
    }
}
