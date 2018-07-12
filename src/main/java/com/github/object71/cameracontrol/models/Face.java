/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol.models;

import java.util.ArrayList;

import com.github.object71.cameracontrol.common.Constants;
import com.github.object71.cameracontrol.common.Helpers;

import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author hristo
 */
public class Face {
    private Eye leftEye;
    private Eye rightEye;
    private CascadeClassifier faceCascade;
    private CascadeClassifier rightEyeCascade;
    private CascadeClassifier leftEyeCascade;

    protected Mat frame = null;
    protected Mat debugFrame = null;
    protected Rect faceLocation;

    public Mat getDebugFrame() {
        return this.debugFrame;
    }

    public Face() {
        this.faceCascade = new CascadeClassifier();
        this.faceCascade.load("./src/main/resources/haarcascade_frontalface_alt.xml");

        leftEyeCascade = new CascadeClassifier();
        leftEyeCascade.load("./src/main/resources/haarcascade_lefteye_2splits.xml");

        rightEyeCascade = new CascadeClassifier();
        rightEyeCascade.load("./src/main/resources/haarcascade_righteye_2splits.xml");

        this.leftEye = new Eye(this);
        this.rightEye = new Eye(this);

    }

    public void initializeFrameInformation(Mat frame) {
        MatOfRect faces = new MatOfRect();

        if (this.frame == null) {
            this.frame = new Mat(frame.rows(), frame.cols(), frame.type());
            this.debugFrame = new Mat(frame.rows(), frame.cols(), frame.type());
        }
        frame.copyTo(this.frame);
        frame.copyTo(this.debugFrame);

        ArrayList<Mat> rgbChannels = new ArrayList<Mat>(3);
        Core.split(frame, rgbChannels);
        Mat grayFrame = rgbChannels.get(2);

        try {
            faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2,
                    0 | Objdetect.CASCADE_DO_CANNY_PRUNING | Objdetect.CASCADE_DO_ROUGH_SEARCH, new Size(),
                    frame.size());
        } catch (Exception e) {
            return;
        }

        double maxDistance = 0;
        Rect[] arrayFaces = faces.toArray();
        if (arrayFaces.length == 0) {
            return;
        }
        for (int i = 0; i < arrayFaces.length; i++) {
            Rect face = arrayFaces[i];
            Point faceCenter = Helpers.centerOfRect(face);
            Point imageCenter = Helpers.centerOfRect(new Rect(0, 0, frame.width(), frame.height()));
            double distance = Helpers.distanceBetweenPoints(faceCenter, imageCenter);
            if (distance > maxDistance) {
                maxDistance = distance;
                this.faceLocation = face;
                Imgproc.rectangle(debugFrame, new Point(face.x, face.y),
                        new Point(face.x + face.width, face.y + face.height), new Scalar(0, 255, 0, 255));
            } else {
                Imgproc.rectangle(debugFrame, new Point(face.x, face.y),
                        new Point(face.x + face.width, face.y + face.height), new Scalar(255, 255, 255, 255));
            }
        }

        Mat faceSubframe = grayFrame.submat(this.faceLocation);

        if (Constants.smoothFaceImage) {
            double sigma = Constants.smoothFaceFactor * faceSubframe.width();
            Imgproc.GaussianBlur(faceSubframe, faceSubframe, new Size(), sigma);
        }

        MatOfRect leftEyeDetections = new MatOfRect();
        leftEyeCascade.detectMultiScale(faceSubframe, leftEyeDetections, 1.1, 3,
                0 | Objdetect.CASCADE_SCALE_IMAGE, new Size(),
                new Size(100, 100));

        MatOfRect rightEyeDetections = new MatOfRect();
        rightEyeCascade.detectMultiScale(faceSubframe, rightEyeDetections, 1.1, 3,
                0 | Objdetect.CASCADE_SCALE_IMAGE, new Size(),
                new Size(100, 100));
        
        Rect[] detections = null;
        
        
        detections = leftEyeDetections.toArray();
        boolean leftEyeDetected = detections.length > 0;
        if (leftEyeDetected) {
            Rect leftEyeRegion = detections[0];
            // Imgproc.rectangle(debugFrame, new Point(leftEyeRegion.x + faceLocation.x, leftEyeRegion.y + faceLocation.y),
            //         new Point(leftEyeRegion.x + faceLocation.x + leftEyeRegion.width, leftEyeRegion.y + faceLocation.y + leftEyeRegion.height),
            //         new Scalar(255, 255, 255, 255));
            Point leftEyeCenter = leftEye.getEyeCenter(faceSubframe, leftEyeRegion);

            leftEyeCenter.x += faceLocation.x + leftEyeRegion.x;
            leftEyeCenter.y += faceLocation.y + leftEyeRegion.y;
            Imgproc.circle(debugFrame, leftEyeCenter, 3, new Scalar(255, 0, 0, 255));
        }

        detections = rightEyeDetections.toArray();
        boolean rightEyeDetected = detections.length > 0;        
        if (rightEyeDetected) {
            Rect rightEyeRegion = detections[0];
            // Imgproc.rectangle(debugFrame,
            //         new Point(rightEyeRegion.x + faceLocation.x, rightEyeRegion.y + faceLocation.y),
            //         new Point(rightEyeRegion.x + faceLocation.x + rightEyeRegion.width, rightEyeRegion.y + faceLocation.y + rightEyeRegion.height),
            //         new Scalar(255, 255, 255, 255));
            Point rightEyeCenter = rightEye.getEyeCenter(faceSubframe, rightEyeRegion);

            rightEyeCenter.x += faceLocation.x + rightEyeRegion.x;
            rightEyeCenter.y += faceLocation.y + rightEyeRegion.y;
            Imgproc.circle(debugFrame, rightEyeCenter, 3, new Scalar(255, 0, 0, 255));
        }
    }
}
