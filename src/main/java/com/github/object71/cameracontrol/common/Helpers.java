/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol.common;

import org.opencv.core.Rect2d;
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

    public static double computeDynamicTreshold(double[] matrix, double standardDeviationFactor, int rows, int cols) {
        Mat inputMatrix = new Mat(rows, cols, CvType.CV_64FC1);
        inputMatrix.put(0, 0, matrix);

        MatOfDouble standartMagnitudeGradient = new MatOfDouble();
        MatOfDouble meanMagnitudeGradient = new MatOfDouble();
        Core.meanStdDev(inputMatrix, meanMagnitudeGradient, standartMagnitudeGradient);

        double standartDeviation = standartMagnitudeGradient.toArray()[0] / Math.sqrt(rows * cols);
        double result = standartDeviation * standardDeviationFactor + meanMagnitudeGradient.toArray()[0];

        return result;
    }

    public static Point centerOfRect(Rect rectangle) {
        return new Point(rectangle.x + (rectangle.width / 2.0), rectangle.y + (rectangle.height / 2.0));
    }

    public static Point centerOfRect(Rect2d rectangle) {
        return new Point(rectangle.x + (rectangle.width / 2.0), rectangle.y + (rectangle.height / 2.0));
    }

    public static double centerOfRectXAxis(Rect rectangle) {
        return rectangle.x + (rectangle.width / 2.0);
    }

    public static double distanceBetweenPoints(Point a, Point b) {
        return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
    }

    public static Point averageBetweenPoints(Point a, Point b) {
        if (a == null) {
            return b;
        } else if (b == null) {
            return b;
        } else {
            return new Point((a.x + b.x) / 2, (a.y + b.y) / 2);
        }
    }

    public static double distanceBetweenValues(double a, double b) {
        return a >= b ? a - b : b - a;
    }

    public static GradientsModel computeGradient(double[] input, int rows, int cols) throws IllegalArgumentException {
        if (cols < 3) {
            throw new IllegalArgumentException("Invalid matrix size");
        }
        double[] gradientX = new double[rows * cols];
        double[] gradientY = new double[rows * cols];
        double[] magnitude = new double[rows * cols];
        double xA, xB, yA, yB, valueX, valueY;

        for (int y = 0; y < rows; y++) {
            for (int x = 1; x < cols; x++) {
                int coordinate = (y * cols) + x;

                xA = x == cols - 1 ? input[coordinate] : input[coordinate + 1];
                xB = x == 0 ? input[coordinate] : input[coordinate - 1];
                gradientX[coordinate] = valueX = xA - xB;

                yA = y == rows - 1 ? input[coordinate] : input[coordinate + cols];
                yB = y == 0 ? input[(y) + (x)] : input[coordinate - cols];
                gradientY[coordinate] = valueY = yA - yB;

                magnitude[coordinate] = Math.sqrt((valueX * valueX) + (valueY * valueY));
            }
        }

        double gradientTreshold = Helpers.computeDynamicTreshold(magnitude, Constants.gradientTreshold, rows,
                cols);

        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                int coordinate = (y * cols) + x;
                valueX = gradientX[coordinate];
                valueY = gradientY[coordinate];
                double magnitudeValue = magnitude[coordinate];
                if (magnitudeValue > gradientTreshold) {
                    gradientX[coordinate] = valueX / magnitudeValue;
                    gradientY[coordinate] = valueY / magnitudeValue;
                } else {
                    gradientX[coordinate] = 0;
                    gradientY[coordinate] = 0;
                }
            }
        }
        GradientsModel gradients = new GradientsModel();
        gradients.gradientX = gradientX;
        gradients.gradientY = gradientY;
        gradients.magnitude = magnitude;
        gradients.gradientTreshold = gradientTreshold;
        gradients.cols = cols;
        gradients.rows = rows;
        
        return gradients;
    }

    public static double[] getMatrixMagnitude(double[] matrixX, double[] matrixY, int rows, int cols) {
        double[] magnitudeMatrix = new double[rows * cols];
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                int coordinate = (y * cols) + x;
                double valueX = matrixX[coordinate];
                double valueY = matrixY[coordinate];
                double magnitudeValue = Math.sqrt((valueX * valueX) + (valueY * valueY));
                magnitudeMatrix[coordinate] = magnitudeValue;
            }
        }
        return magnitudeMatrix;
    }

    public static double[] matrixToArray(Mat matrix) {
        int rows = matrix.rows();
        int cols = matrix.cols();
        double[] result = new double[rows * cols];

        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                result[y * cols + x] = matrix.get(y, x)[0];

            }
        }

        return result;
    }    
    
    public static Mat arrayToMatrix(double[] matrix, int rows, int cols) {
        return arrayToMatrix(matrix, rows, cols, CvType.CV_64F);
    }

    public static Mat arrayToMatrix(double[] matrix, int rows, int cols, int type) {
        Mat result = new Mat(rows, cols, type);

        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                result.put(y, x, matrix[y * cols + x]);
            }
        }

        return result;
    }

    public static void bright(Mat matrix, double brightenBy) {
        for (int y = 0; y < matrix.rows(); y++) {
            for (int x = 0; x < matrix.cols(); x++) {
                double[] values = matrix.get(y, x);
                for (int v = 0; v < values.length; v++) {
                    double value = values[v] + brightenBy;
                    values[v] = value > 255 ? 255 : value;
                }
                matrix.put(y, x, values);
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
