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

    private CascadeClassifier faceCascade;
    private CascadeClassifier rightEyeCascade;
    private CascadeClassifier leftEyeCascade;

    public FaceEyeHandler() {
        this.faceCascade = new CascadeClassifier();
        this.faceCascade.load("./src/main/resources/haarcascade_frontalface_alt.xml");

        leftEyeCascade = new CascadeClassifier();
        leftEyeCascade.load("./src/main/resources/haarcascade_lefteye_2splits.xml");

        rightEyeCascade = new CascadeClassifier();
        rightEyeCascade.load("./src/main/resources/haarcascade_righteye_2splits.xml");

    }

    public void initializeFrameInformation(Mat inputFrame) {
        Mat frame = new Mat(inputFrame.rows(), inputFrame.cols(), CvType.CV_64FC1);

        Imgproc.cvtColor(inputFrame, frame, Imgproc.COLOR_BGR2GRAY);

        Rect faceLocation = this.getFaceLocation(frame);
        if(faceLocation == null) {
            return;
        }
        Mat faceSubframe = frame.submat(faceLocation);

        if (Constants.smoothFaceImage) {
            double sigma = Constants.smoothFaceFactor * faceSubframe.width();
            Imgproc.GaussianBlur(faceSubframe, faceSubframe, new Size(), sigma);
        }

        MatOfRect leftEyeDetections = new MatOfRect();
        leftEyeCascade.detectMultiScale(faceSubframe, leftEyeDetections, 1.1, 3, 0 | Objdetect.CASCADE_SCALE_IMAGE,
                new Size(), new Size(100, 100));

        MatOfRect rightEyeDetections = new MatOfRect();
        rightEyeCascade.detectMultiScale(faceSubframe, rightEyeDetections, 1.1, 3, 0 | Objdetect.CASCADE_SCALE_IMAGE,
                new Size(), new Size(100, 100));

        Rect[] detections = null;
        double faceLeftSide = faceLocation.x + faceLocation.width;
        double faceRightSide = faceLocation.x;

        detections = leftEyeDetections.toArray();
        boolean leftEyeDetected = detections.length > 0;
        if (leftEyeDetected) {
            Rect leftEyeRegion = getEyeLocation(detections, faceLeftSide);
            Point leftEyeCenter = getEyeCenter(faceSubframe, leftEyeRegion);

            leftEyeCenter.x += faceLocation.x + leftEyeRegion.x;
            leftEyeCenter.y += faceLocation.y + leftEyeRegion.y;
        }

        detections = rightEyeDetections.toArray();
        boolean rightEyeDetected = detections.length > 0;
        if (rightEyeDetected) {
            Rect rightEyeRegion = getEyeLocation(detections, faceRightSide);
            Point rightEyeCenter = getEyeCenter(faceSubframe, rightEyeRegion);

            rightEyeCenter.x += faceLocation.x + rightEyeRegion.x;
            rightEyeCenter.y += faceLocation.y + rightEyeRegion.y;
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
        Mat eyeRegionSubframeUnscaled = faceSubframe.submat(eyeRegion);
        Size scaleTo = new Size(Constants.fastEyeWidth, Constants.fastEyeWidth);
        Mat eyeRegionSubframe = new Mat((int) Math.round(scaleTo.width), (int) Math.round(scaleTo.height),
                eyeRegionSubframeUnscaled.type());

        Imgproc.resize(eyeRegionSubframeUnscaled, eyeRegionSubframe, scaleTo);

        // y is calculated with the same func - matrix is rotated
        Mat gradientXMatrix = Helpers.computeMatXGradient(eyeRegionSubframe);
        Mat gradientYMatrix = Helpers.computeMatXGradient(eyeRegionSubframe.t()).t();
        Mat magnitudeMatrix = Helpers.getMatrixMagnitude(gradientXMatrix, gradientYMatrix);

        double gradientTreshold = Helpers.computeDynamicTreshold(magnitudeMatrix, Constants.gradientTreshold);

        for (int y = 0; y < eyeRegionSubframe.rows(); y++) {
            for (int x = 0; x < eyeRegionSubframe.cols(); x++) {
                double valueX = gradientXMatrix.get(y, x)[0];
                double valueY = gradientYMatrix.get(y, x)[0];
                double magnitude = magnitudeMatrix.get(y, x)[0];
                if (magnitude > gradientTreshold) {
                    gradientXMatrix.put(y, x, valueX / magnitude);
                    gradientYMatrix.put(y, x, valueY / magnitude);
                } else {
                    gradientXMatrix.put(y, x, 0);
                    gradientYMatrix.put(y, x, 0);
                }
            }
        }

        Mat weight = new Mat(eyeRegionSubframe.rows(), eyeRegionSubframe.cols(), CvType.CV_64F);
        Imgproc.GaussianBlur(eyeRegionSubframe, weight, new Size(Constants.weightBlurSize, Constants.weightBlurSize), 0,
                0);
        for (int y = 0; y < weight.rows(); y++) {
            for (int x = 0; x < weight.cols(); x++) {
                weight.put(y, x, 255 - weight.get(y, x)[0]);
            }
        }

        Mat sum = new Mat(eyeRegionSubframe.rows(), eyeRegionSubframe.cols(), CvType.CV_64F);
        for (int y = 0; y < weight.rows(); y++) {
            for (int x = 0; x < weight.cols(); x++) {
                double valueX = gradientXMatrix.get(y, x)[0];
                double valueY = gradientYMatrix.get(y, x)[0];
                if (valueX == 0.0 && valueY == 0.0) {
                    continue;
                }

                Helpers.possibleCenterFormula(x, y, weight, valueX, valueY, sum);
            }
        }

        double numGradients = (weight.rows() * weight.cols());
        Mat out = new Mat(sum.rows(), sum.cols(), CvType.CV_32F);
        sum.convertTo(out, CvType.CV_32F, 1.0 / numGradients);

        MinMaxLocResult result = Core.minMaxLoc(out, null);

        if (Constants.enablePostProcessing) {
            Mat floodClone = new Mat(out.rows(), out.cols(), CvType.CV_32F);
            double floodThresh = result.maxVal * Constants.postProcessingTreshold;
            Imgproc.threshold(out, floodClone, floodThresh, 0.0, Imgproc.THRESH_TOZERO);
            Mat mask = floodKillEdges(floodClone);

            MinMaxLocResult endResult = Core.minMaxLoc(out, mask);

            return unscalePoint(endResult.maxLoc, eyeRegion);
        }

        return unscalePoint(result.maxLoc, eyeRegion);
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
            if (floodShouldPushPoint(nextPoint, matrix))
                todo.add(nextPoint);
            nextPoint.x = currentPoint.x - 1;
            nextPoint.y = currentPoint.y; // left
            if (floodShouldPushPoint(nextPoint, matrix))
                todo.add(nextPoint);
            nextPoint.x = currentPoint.x;
            nextPoint.y = currentPoint.y + 1; // down
            if (floodShouldPushPoint(nextPoint, matrix))
                todo.add(nextPoint);
            nextPoint.x = currentPoint.x;
            nextPoint.y = currentPoint.y - 1; // up
            if (floodShouldPushPoint(nextPoint, matrix))
                todo.add(nextPoint);
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
