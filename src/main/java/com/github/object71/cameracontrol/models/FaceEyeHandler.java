/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol.models;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;

import com.github.object71.cameracontrol.common.Constants;
import com.github.object71.cameracontrol.common.Helpers;

import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.tracking.Tracker;
import org.opencv.tracking.TrackerBoosting;
import org.opencv.tracking.TrackerKCF;
import org.opencv.imgproc.Imgproc;
import org.opencv.tracking.TrackerTLD;

/**
 *
 * @author hristo
 */
public class FaceEyeHandler {

    private CascadeClassifier faceCascade;
    private CascadeClassifier rightEyeCascade;
    private CascadeClassifier leftEyeCascade;
    private Tracker leftEyeTracker = null;
    private Tracker rightEyeTracker = null;
    private long lastUpdate = System.currentTimeMillis();

    public Point leftEye = null;
    public Point rightEye = null;

    public FaceEyeHandler() {
        this.faceCascade = new CascadeClassifier();
        this.faceCascade.load("./src/main/resources/haarcascade_frontalface_alt.xml");

        leftEyeCascade = new CascadeClassifier();
        leftEyeCascade.load("./src/main/resources/haarcascade_lefteye_2splits.xml");

        rightEyeCascade = new CascadeClassifier();
        rightEyeCascade.load("./src/main/resources/haarcascade_righteye_2splits.xml");

        this.leftEyeTracker = TrackerTLD.create();
        this.rightEyeTracker = TrackerTLD.create();

    }

    public void initializeFrameInformation(Mat inputFrame) {
        boolean tracking = true;
        if ((System.currentTimeMillis() - lastUpdate > 80) || leftEye == null || rightEye == null) {
            tracking = false;
            this.lastUpdate = System.currentTimeMillis();
        }
        Helpers.bright(inputFrame, 0.35);

        Mat frame = new Mat(inputFrame.rows(), inputFrame.cols(), CvType.CV_64F);

        boolean leftTracked = tracking;
        boolean rightTracked = tracking;

        Imgproc.cvtColor(inputFrame, frame, Imgproc.COLOR_BGR2GRAY);

        if (tracking) {
            int size = frame.cols() / 100;
            Rect2d leftEyeLocation = new Rect2d(this.leftEye.x - size, this.leftEye.y - size, size * 2, size * 2);
            Rect2d rightEyeLocation = new Rect2d(this.rightEye.x - size, this.rightEye.y - size, size * 2, size * 2);

            leftTracked = this.leftEyeTracker.update(frame, leftEyeLocation);
            rightTracked = this.rightEyeTracker.update(frame, rightEyeLocation);

            if (leftTracked) {
                this.leftEye = Helpers.averageBetweenPoints(Helpers.centerOfRect(leftEyeLocation), this.leftEye);
            }
            if (rightTracked) {
                this.rightEye = Helpers.averageBetweenPoints(Helpers.centerOfRect(rightEyeLocation), this.rightEye);
            }
            if (leftTracked && rightTracked) {
                return;
            }
        }

        Rect faceLocation = this.getFaceLocation(frame);
        if (faceLocation == null) {
            return;
        }
        Mat faceSubframe = frame.submat(faceLocation);

        if (Constants.smoothFaceImage) {
            double sigma = Constants.smoothFaceFactor * faceSubframe.width();
            Imgproc.GaussianBlur(faceSubframe, faceSubframe, new Size(), sigma);
        }

        if (!leftTracked) {
            Rect[] detections = null;
            double faceLeftSide = faceLocation.x + faceLocation.width;
            MatOfRect leftEyeDetections = new MatOfRect();
            leftEyeCascade.detectMultiScale(faceSubframe, leftEyeDetections, 1.1, 3, 0 | Objdetect.CASCADE_SCALE_IMAGE,
                    new Size(), new Size(100, 100));
            detections = leftEyeDetections.toArray();
            boolean leftEyeDetected = detections.length > 0;
            if (leftEyeDetected) {
                Rect leftEyeRegion = getEyeLocation(detections, faceLeftSide);
                Point leftEyeCenter = getEyeCenter(faceSubframe, leftEyeRegion);

                leftEyeCenter.x += faceLocation.x + leftEyeRegion.x;
                leftEyeCenter.y += faceLocation.y + leftEyeRegion.y;
                this.leftEye = leftEyeCenter;
            }
        }

        if (!rightTracked) {
            Rect[] detections = null;
            double faceRightSide = faceLocation.x;
            MatOfRect rightEyeDetections = new MatOfRect();
            rightEyeCascade.detectMultiScale(faceSubframe, rightEyeDetections, 1.1, 3,
                    0 | Objdetect.CASCADE_SCALE_IMAGE, new Size(), new Size(100, 100));
            detections = rightEyeDetections.toArray();
            boolean rightEyeDetected = detections.length > 0;
            if (rightEyeDetected) {
                Rect rightEyeRegion = getEyeLocation(detections, faceRightSide);
                Point rightEyeCenter = getEyeCenter(faceSubframe, rightEyeRegion);

                rightEyeCenter.x += faceLocation.x + rightEyeRegion.x;
                rightEyeCenter.y += faceLocation.y + rightEyeRegion.y;
                this.rightEye = rightEyeCenter;
            }
        }
    }

    private Rect getFaceLocation(Mat frame) {
        MatOfRect faces = new MatOfRect();

        faceCascade.detectMultiScale(frame, faces, 1.1, 2,
                0 | Objdetect.CASCADE_DO_CANNY_PRUNING | Objdetect.CASCADE_DO_ROUGH_SEARCH, new Size(), frame.size());

        double maxDistance = 0;
        Rect[] arrayFaces = faces.toArray();
        Rect faceLocation = null;
        if (arrayFaces.length == 0) {
            return null;
        }
        for (int i = 0; i < arrayFaces.length; i++) {
            Rect face = arrayFaces[i];
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

    private Rect getEyeLocation(Rect[] detections, double faceSide) {
        Rect rightEyeRegion = null;
        double minEyeToSideDistance = Double.MAX_VALUE;

        for (int i = 0; i < detections.length; i++) {
            Rect eyeRegion = detections[i];
            double eyeRegionX = Helpers.centerOfRectXAxis(eyeRegion);
            double distance = Helpers.distanceBetweenValues(eyeRegionX, faceSide);
            if (distance < minEyeToSideDistance) {
                minEyeToSideDistance = distance;
                rightEyeRegion = eyeRegion;
            }
        }

        return rightEyeRegion;
    }

    protected Point getEyeCenter(Mat faceSubframe, Rect eyeRegion) {
        // Mat eyeRegionSubframeUnscaled = faceSubframe.submat(eyeRegion);
        // Size scaleTo = new Size(Constants.fastEyeWidth, Constants.fastEyeWidth);
        // Mat eyeRegionSubframe = new Mat(scaleTo, eyeRegionSubframeUnscaled.type());
        // Imgproc.resize(eyeRegionSubframeUnscaled, eyeRegionSubframe, scaleTo);
        Mat eyeRegionSubframe = faceSubframe.submat(eyeRegion);

        int rows = eyeRegionSubframe.rows();
        int cols = eyeRegionSubframe.cols();
        double[] frameAsDoubles = Helpers.matrixToArray(eyeRegionSubframe);

        double[] gradientXMatrix = Helpers.computeMatXGradient(frameAsDoubles, rows, cols);
        double[] gradientYMatrix = Helpers.computeMatYGradient(frameAsDoubles, rows, cols);
        double[] magnitudeMatrix = Helpers.getMatrixMagnitude(gradientXMatrix, gradientYMatrix, rows, cols);

        double gradientTreshold = Helpers.computeDynamicTreshold(magnitudeMatrix, Constants.gradientTreshold, rows,
                cols);

        for (int y = 0; y < eyeRegionSubframe.rows(); y++) {
            for (int x = 0; x < eyeRegionSubframe.cols(); x++) {
                int coordinate = (y * cols) + x;
                double valueX = gradientXMatrix[coordinate];
                double valueY = gradientYMatrix[coordinate];
                double magnitude = magnitudeMatrix[coordinate];
                if (magnitude > gradientTreshold) {
                    gradientXMatrix[coordinate] = valueX / magnitude;
                    gradientYMatrix[coordinate] = valueY / magnitude;
                } else {
                    gradientXMatrix[coordinate] = 0;
                    gradientYMatrix[coordinate] = 0;
                }
            }
        }

        Mat weight = new Mat(rows, cols, CvType.CV_64F);
        Imgproc.GaussianBlur(eyeRegionSubframe, weight, new Size(Constants.weightBlurSize, Constants.weightBlurSize), 0,
                0);
        for (int y = 0; y < weight.rows(); y++) {
            for (int x = 0; x < weight.cols(); x++) {
                weight.put(y, x, 255 - weight.get(y, x)[0]);
            }
        }

        Mat sum = new Mat(rows, cols, CvType.CV_64F);
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                int coordinate = (y * cols) + x;
                double valueX = gradientXMatrix[coordinate];
                double valueY = gradientYMatrix[coordinate];
                if (valueX == 0.0 && valueY == 0.0) {
                    continue;
                }

                Helpers.possibleCenterFormula(x, y, weight, valueX, valueY, sum);
            }
        }

        double numGradients = (rows * cols);
        Mat out = new Mat(rows, cols, CvType.CV_32F);
        sum.convertTo(out, CvType.CV_32F, 1.0 / numGradients);

        MinMaxLocResult result = Core.minMaxLoc(out, null);

        if (Constants.enablePostProcessing) {
            Mat floodClone = new Mat(rows, cols, CvType.CV_32F);
            double floodThresh = result.maxVal * Constants.postProcessingTreshold;
            Imgproc.threshold(out, floodClone, floodThresh, 0.0, Imgproc.THRESH_TOZERO);
            Mat mask = floodKillEdges(floodClone);

            MinMaxLocResult endResult = Core.minMaxLoc(out, mask);

            return endResult.maxLoc;
            // return unscalePoint(endResult.maxLoc, eyeRegion);
        }

        // return unscalePoint(result.maxLoc, eyeRegion);
        return result.maxLoc;
    }

    private static Mat floodKillEdges(Mat matrix) {
        // rectangle(matrix, new Rect(0,0,matrix.cols(), matrix.rows()),255);

        Mat mask = new Mat(matrix.rows(), matrix.cols(), CvType.CV_8U, new Scalar(255, 255, 255, 255));
        Queue<Point> todo = new LinkedList<Point>();
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
