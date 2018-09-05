package com.github.object71.cameracontrol.models;

import java.util.LinkedList;
import java.util.Queue;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.imgproc.Imgproc;

import com.github.object71.cameracontrol.common.Constants;
import com.github.object71.cameracontrol.common.GradientsModel;
import com.github.object71.cameracontrol.common.Helpers;

public class EyeHandler {
	
	public static Point getEyeCenter(Mat faceSubframe, Rect eyeRegion) {
        Mat eyeRegionSubframe = faceSubframe.submat(eyeRegion);
        
        int rows = eyeRegionSubframe.rows();
        int cols = eyeRegionSubframe.cols();
        double[] frameAsDoubles = Helpers.matrixToArray(eyeRegionSubframe);

        GradientsModel gradients = Helpers.computeGradient(frameAsDoubles, rows, cols);

        Mat weight = new Mat(rows, cols, CvType.CV_64F);
        Imgproc.GaussianBlur(eyeRegionSubframe, weight, new Size(Constants.weightBlurSize, Constants.weightBlurSize), 0,
                0);

        double[] sum = new double[rows * cols];
        double[] weightArray = Helpers.matrixToArray(weight);
        
        for (int y = 0; y < weight.rows(); y++) {
            for (int x = 0; x < weight.cols(); x++) {
                int coordinate = (y * cols) + x;

                double valueX = gradients.gradientX[coordinate];
                double valueY = gradients.gradientY[coordinate];
                if (valueX == 0.0 && valueY == 0.0) {
                    continue;
                }

                for (int cy = 0; cy < rows; cy++) {
                    for (int cx = 0; cx < cols; cx++) {
                        int coordinateC = (cy * cols) + cx;
                        //check all other than the current value
                        if (x == cx && y == cy) {
                            continue;
                        }

                        double dx = x - cx; // the distance to a certain point
                        double dy = y - cy; // the distance to a certain point
                        double magnitude = Math.sqrt((dx * dx) + (dy * dy)); // actual vector distance

                        dx = dx / magnitude;
                        dy = dy / magnitude;

                        // 0 or positive
                        double dotProduct = Math.max(0, dx * valueX + dy * valueY);

                        double currentValue = sum[coordinateC];

                        if (Constants.enableWeight) {
                            double weightValue = 255 - weightArray[coordinate];
                            sum[coordinateC] = currentValue + (Math.pow(dotProduct, 2) * (weightValue / Constants.weightDivisor));
                        } else {
                            sum[coordinateC] = currentValue + Math.pow(dotProduct, 2);
                        }
                    }
                }
            }
        }

        Mat out = Helpers.arrayToMatrix(sum, rows, cols, CvType.CV_32F);

        MinMaxLocResult result = Core.minMaxLoc(out, null);

        if (Constants.enablePostProcessing) {
            Mat floodClone = new Mat(rows, cols, CvType.CV_32F);
            double floodThresh = result.maxVal * Constants.postProcessingTreshold;
            Imgproc.threshold(out, floodClone, floodThresh, 0.0, Imgproc.THRESH_TOZERO);
            Mat mask = floodKillEdges(floodClone);

            MinMaxLocResult endResult = Core.minMaxLoc(out, mask);

            return endResult.maxLoc;
        }

        return result.maxLoc;
    }

    private static Mat floodKillEdges(Mat matrix) {
        Mat mask = new Mat(matrix.rows(), matrix.cols(), CvType.CV_8U, new Scalar(255, 255, 255, 255));
        Queue<Point> todo = new LinkedList<>();
        todo.add(new Point(0, 0));

        while (todo.size() > 0) {
            Point currentPoint = todo.peek();
            todo.poll();
            if (matrix.get((int) currentPoint.y, (int) currentPoint.x)[0] == 0.0) {
                continue;
            }
            // add in every direction
            Point nextPoint = new Point(currentPoint.x + 1, currentPoint.y); // right
            if (Helpers.isPointInMatrix(nextPoint, matrix.rows(), matrix.cols())) {
                todo.add(nextPoint);
            }
            
            nextPoint.x = currentPoint.x - 1;
            nextPoint.y = currentPoint.y; // left
            if (Helpers.isPointInMatrix(nextPoint, matrix.rows(), matrix.cols())) {
                todo.add(nextPoint);
            }
            
            nextPoint.x = currentPoint.x;
            nextPoint.y = currentPoint.y + 1; // down
            if (Helpers.isPointInMatrix(nextPoint, matrix.rows(), matrix.cols())) {
                todo.add(nextPoint);
            }
            
            nextPoint.x = currentPoint.x;
            nextPoint.y = currentPoint.y - 1; // up
            if (Helpers.isPointInMatrix(nextPoint, matrix.rows(), matrix.cols())) {
                todo.add(nextPoint);
            }
            // kill it
            matrix.put((int) currentPoint.y, (int) currentPoint.x, 0.0);
            mask.put((int) currentPoint.y, (int) currentPoint.x, 0.0);
        }
        return mask;
    }
}
