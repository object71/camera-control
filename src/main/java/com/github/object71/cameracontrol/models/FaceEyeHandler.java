/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol.models;

import java.util.LinkedList;
import java.util.Queue;

import com.github.object71.cameracontrol.common.Constants;
import com.github.object71.cameracontrol.common.GradientsModel;
import com.github.object71.cameracontrol.common.Helpers;
import com.github.object71.cameracontrol.common.PointHistoryCollection;

import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author hristo
 */
public class FaceEyeHandler {

    private final CascadeClassifier faceCascade;
    private final CascadeClassifier rightEyeCascade;
    private final CascadeClassifier leftEyeCascade;

    public PointHistoryCollection leftEye = new PointHistoryCollection(3);
    public PointHistoryCollection rightEye = new PointHistoryCollection(3);

    public FaceEyeHandler() {
        this.faceCascade = new CascadeClassifier();
        this.faceCascade.load("./src/main/resources/haarcascade_frontalface_alt.xml");

        leftEyeCascade = new CascadeClassifier();
        leftEyeCascade.load("./src/main/resources/haarcascade_lefteye_2splits.xml");

        rightEyeCascade = new CascadeClassifier();
        rightEyeCascade.load("./src/main/resources/haarcascade_righteye_2splits.xml");
    }

    public void initializeFrameInformation(Mat inputFrame) {
        Helpers.bright(inputFrame, 25);

        Mat frame = new Mat(inputFrame.rows(), inputFrame.cols(), CvType.CV_64F);

        Imgproc.cvtColor(inputFrame, frame, Imgproc.COLOR_BGR2GRAY);

        Rect faceLocation = this.getFaceLocation(frame);
        if (faceLocation == null) {
            return;
        }

        Mat faceSubframe = frame.submat(faceLocation);

        if (Constants.smoothFaceImage) {
            double sigma = Constants.smoothFaceFactor * faceSubframe.width();
            Imgproc.GaussianBlur(faceSubframe, faceSubframe, new Size(), sigma);
        }

        Rect[] detectionsLeft;
        double faceLeftSide = 0;
        MatOfRect leftEyeDetections = new MatOfRect();

        leftEyeCascade.detectMultiScale(faceSubframe, leftEyeDetections, 1.05, 0, Objdetect.CASCADE_DO_CANNY_PRUNING,
                new Size(), new Size(faceSubframe.width(), faceSubframe.height()));
        detectionsLeft = leftEyeDetections.toArray();

        Rect[] detectionsRight;
        double faceRightSide = faceSubframe.width();
        MatOfRect rightEyeDetections = new MatOfRect();

        rightEyeCascade.detectMultiScale(faceSubframe, rightEyeDetections, 1.05, 0, Objdetect.CASCADE_DO_CANNY_PRUNING, 
                new Size(), new Size(faceSubframe.width(), faceSubframe.height()));
        detectionsRight = rightEyeDetections.toArray();

        if (detectionsLeft.length > 0) {
            Rect leftEyeRegion = getEyeLocation(detectionsLeft, faceLeftSide, faceRightSide);
            if (leftEyeRegion != null) {
                Point leftEyeCenter = getEyeCenter(faceSubframe, leftEyeRegion);
                

                leftEyeCenter.x += faceLocation.x + leftEyeRegion.x;
                leftEyeCenter.y += faceLocation.y + leftEyeRegion.y;
                this.leftEye.insertNewPoint(leftEyeCenter);
            }
        }

        if (detectionsRight.length > 0) {
            Rect rightEyeRegion = getEyeLocation(detectionsRight, faceRightSide, faceLeftSide);

            if (rightEyeRegion != null) {
                Point rightEyeCenter = getEyeCenter(faceSubframe, rightEyeRegion);

                rightEyeCenter.x += faceLocation.x + rightEyeRegion.x;
                rightEyeCenter.y += faceLocation.y + rightEyeRegion.y;
                this.rightEye.insertNewPoint(rightEyeCenter);
            }
        }
    }

    private Rect getFaceLocation(Mat frame) {
        MatOfRect faces = new MatOfRect();

        faceCascade.detectMultiScale(frame, faces, 1.1, 2,
                Objdetect.CASCADE_DO_CANNY_PRUNING | Objdetect.CASCADE_FIND_BIGGEST_OBJECT, new Size(frame.size().width / 25, frame.size().height / 25), frame.size());

        double maxDistance = 0;
        Rect[] arrayFaces = faces.toArray();
        Rect faceLocation = null;
        if (arrayFaces.length == 0) {
            return null;
        }
        for (Rect face : arrayFaces) {
            Point faceCenter = Helpers.centerOfRect(face);
            Point imageCenter = Helpers.centerOfRect(new Rect(0, 0, frame.width(), frame.height()));
            double distance = Helpers.distanceBetweenPoints(faceCenter, imageCenter);
            if (distance > maxDistance) {
                maxDistance = distance;
                faceLocation = face;
            }
        }

        return faceLocation;
    }

    private Rect getEyeLocation(Rect[] detections, double faceDesiredSide, double faceUndesiredSide) {
        Rect rightEyeRegion = null;
        double minEyeToSideDistance = Double.MAX_VALUE;

        for (Rect eyeRegion : detections) {
            double eyeRegionX = Helpers.centerOfRectXAxis(eyeRegion);
            double distanceToDesired = Helpers.distanceBetweenValues(eyeRegionX, faceDesiredSide);
            double distanceToUndesired = Helpers.distanceBetweenValues(eyeRegionX, faceUndesiredSide);

            if (distanceToDesired < minEyeToSideDistance && distanceToDesired < distanceToUndesired) {
                minEyeToSideDistance = distanceToDesired;
                rightEyeRegion = eyeRegion;
            }
        }

        return rightEyeRegion;
    }

    protected Point getEyeCenter(Mat faceSubframe, Rect eyeRegion) {
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

        //double numGradients = (rows * cols);
        //Mat out = new Mat(rows, cols, CvType.CV_32F);
        //Helpers.arrayToMatrix(sum, rows, cols).convertTo(out, CvType.CV_32F, 1.0 / numGradients);
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
        // rectangle(matrix, new Rect(0,0,matrix.cols(), matrix.rows()),255);

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
            if (floodShouldPushPoint(nextPoint, matrix)) {
                todo.add(nextPoint);
            }
            nextPoint.x = currentPoint.x - 1;
            nextPoint.y = currentPoint.y; // left
            if (floodShouldPushPoint(nextPoint, matrix)) {
                todo.add(nextPoint);
            }
            nextPoint.x = currentPoint.x;
            nextPoint.y = currentPoint.y + 1; // down
            if (floodShouldPushPoint(nextPoint, matrix)) {
                todo.add(nextPoint);
            }
            nextPoint.x = currentPoint.x;
            nextPoint.y = currentPoint.y - 1; // up
            if (floodShouldPushPoint(nextPoint, matrix)) {
                todo.add(nextPoint);
            }
            // kill it
            matrix.put((int) currentPoint.y, (int) currentPoint.x, 0.0);
            mask.put((int) currentPoint.y, (int) currentPoint.x, 0.0);
        }
        return mask;
    }

    private static boolean floodShouldPushPoint(Point nextPoint, Mat matrix) {
        return Helpers.isPointInMatrix(nextPoint, matrix.rows(), matrix.cols());
    }

    private static Point unscalePoint(Point point, Rect originalSize) {
        double ratio = Constants.fastEyeWidth / originalSize.width;
        int x = (int) Math.round(point.x / ratio);
        int y = (int) Math.round(point.y / ratio);
        return new Point(x, y);
    }
}
