/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol.models;

import com.github.object71.cameracontrol.common.Helpers;
import com.github.object71.cameracontrol.common.MarkPoint;
import com.github.object71.cameracontrol.common.PointHistoryCollection;

import org.opencv.core.Mat;
import org.apache.commons.lang.time.StopWatch;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.flandmark;
import org.bytedeco.javacpp.flandmark.FLANDMARK_Model;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Size;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.tracking.Tracker;
import org.opencv.tracking.TrackerBoosting;
import org.opencv.tracking.Tracking;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author hristo
 */
public class FaceHandler {

    private static final CascadeClassifier faceCascade;
    private static final FLANDMARK_Model model;

    private final double[] faceMarks;
    private final Point[] faceMarkPoints;
    private Tracker faceTracker;
    private Rect2d trackedFace;

    public PointHistoryCollection leftEye = new PointHistoryCollection(3);
    public PointHistoryCollection rightEye = new PointHistoryCollection(3);

    public PointHistoryCollection eyeGazeCoordinate = new PointHistoryCollection(3);
    public double coordinateSystemSide = 1;

    static {
        faceCascade = new CascadeClassifier();
        faceCascade.load("./src/main/resources/haarcascade_frontalface_alt.xml");

        model = flandmark.flandmark_init("./src/main/resources/flandmark_model.dat");
        if (model == null) {
            System.out.println("Structure model wasn't created. Corrupted file flandmark_model.dat?");
            System.exit(1);
        }

    }

    public FaceHandler() {
        this.faceMarks = new double[2 * model.data().options().M()];
        this.faceMarkPoints = new Point[model.data().options().M()];

    }

    public void initializeFrameInformation(Mat inputFrame) {

        if (inputFrame == null || inputFrame.empty()) {
            return;
        }

        Size size = inputFrame.size();
        Size commonSize = new Size(640, 480);
        if (size.width != commonSize.width || size.height != commonSize.height) {
            Mat resizedFrame = new Mat(commonSize, inputFrame.type());
            Imgproc.resize(inputFrame, resizedFrame, commonSize);
            Mat old = inputFrame;
            inputFrame = resizedFrame;
            old.release();
        }

        Core.flip(inputFrame, inputFrame, 1);

        Mat frame = new Mat(inputFrame.rows(), inputFrame.cols(), CvType.CV_64F);

        Imgproc.cvtColor(inputFrame, frame, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(frame, frame);

        if (frame.empty()) {
            return;
        }
        
        Rect faceLocation = null;
        if (faceTracker == null) {
            faceLocation = this.getFaceLocation(frame);
            if (faceLocation == null) {
                return;
            }

            trackedFace = Helpers.RectToRect2d(faceLocation);

            faceTracker = TrackerBoosting.create();
            faceTracker.init(inputFrame, trackedFace);
        } else {
            if (faceTracker.update(inputFrame, trackedFace)) {
                faceLocation = Helpers.Rect2dToRect(trackedFace);
            }
        }

        if (faceLocation == null) {
            return;
        }

        try (Pointer framePtr = Helpers.getPointer(frame.clone().getNativeObjAddr());
                org.bytedeco.javacpp.opencv_core.Mat img_grayscale_mat = new org.bytedeco.javacpp.opencv_core.Mat(
                        framePtr);
                org.bytedeco.javacpp.opencv_core.IplImage img_grayscale = new org.bytedeco.javacpp.opencv_core.IplImage(
                        img_grayscale_mat);) {

            int[] bbox = getBoundindBox(faceLocation);
            if (flandmark.flandmark_detect(img_grayscale, bbox, model, faceMarks) != 0) {
                return;
            }

            this.marksToPoints();

        } catch (Exception e) {
            return;
        }

        Point leftEyeLeftCorner = this.getFaceMark(MarkPoint.LeftEyeLeftCorner);
        Point leftEyeRightCorner = this.getFaceMark(MarkPoint.LeftEyeRightCorder);
        Point rightEyeLeftCorner = this.getFaceMark(MarkPoint.RightEyeLeftCorner);
        Point rightEyeRightCorner = this.getFaceMark(MarkPoint.RightEyeRightCorner);

        double distanceLeftCorners = Helpers.distanceBetweenPoints(leftEyeLeftCorner, leftEyeRightCorner);
        double distanceRightCorners = Helpers.distanceBetweenPoints(rightEyeLeftCorner, rightEyeRightCorner);

        int commonDistance = (int) (distanceLeftCorners + distanceRightCorners) / 2;

        Rect leftEyeBall = new Rect((int) leftEyeLeftCorner.x, (int) leftEyeLeftCorner.y - commonDistance / 2,
                commonDistance, commonDistance);

        Rect rightEyeBall = new Rect((int) rightEyeLeftCorner.x, (int) rightEyeLeftCorner.y - commonDistance / 2,
                commonDistance, commonDistance);

        Point leftEyeCenter = EyeHandler.getEyeCenter(frame.submat(leftEyeBall));

        leftEyeCenter.x += leftEyeBall.x;
        leftEyeCenter.y += leftEyeBall.y;
        this.leftEye.insertNewPoint(leftEyeCenter);

        Point rightEyeCenter = EyeHandler.getEyeCenter(frame.submat(rightEyeBall));

        rightEyeCenter.x += rightEyeBall.x;
        rightEyeCenter.y += rightEyeBall.y;
        this.rightEye.insertNewPoint(rightEyeCenter);

        this.recalculateCoordinates();

        frame.release();
    }

    public Point getFaceMark(MarkPoint mark) {
        switch (mark) {
            case Center:
                return faceMarkPoints[0];
            case LeftEyeRightCorder:
                return faceMarkPoints[1];
            case RightEyeLeftCorner:
                return faceMarkPoints[2];
            case MouthLeftCorner:
                return faceMarkPoints[3];
            case MouthRightCorner:
                return faceMarkPoints[4];
            case LeftEyeLeftCorner:
                return faceMarkPoints[5];
            case RightEyeRightCorner:
                return faceMarkPoints[6];
            case Nose:
                return faceMarkPoints[7];
            default:
                return faceMarkPoints[0];
        }
    }

    private void marksToPoints() {
        int x = 0;
        for (int i = 0; i < faceMarks.length; i += 2, x++) {
            faceMarkPoints[x] = new Point(faceMarks[i], faceMarks[i + 1]);
        }
    }

    private static int[] getBoundindBox(Rect rectangle) {
        int[] bbox = new int[]{rectangle.x, rectangle.y, rectangle.x + rectangle.width,
            rectangle.y + rectangle.height};
        return bbox;
    }

    private Rect getFaceLocation(Mat frame) {

        if (faceCascade.empty()) {
            throw new RuntimeException("Face cascade is empty.");
        }
        MatOfRect faces = new MatOfRect();

        faceCascade.detectMultiScale(frame, faces, 1.1, 2,
                Objdetect.CASCADE_DO_CANNY_PRUNING | Objdetect.CASCADE_FIND_BIGGEST_OBJECT, new Size(), frame.size());

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

    public void recalculateCoordinates() {
        Point leftEyeLeftCorner = this.getFaceMark(MarkPoint.LeftEyeLeftCorner);
        Point leftEyeRightCorner = this.getFaceMark(MarkPoint.LeftEyeRightCorder);
        Point rightEyeLeftCorner = this.getFaceMark(MarkPoint.RightEyeLeftCorner);
        Point rightEyeRightCorner = this.getFaceMark(MarkPoint.RightEyeRightCorner);

        double distanceLeftCorners = Helpers.distanceBetweenPoints(leftEyeLeftCorner, leftEyeRightCorner);
        double distanceRightCorners = Helpers.distanceBetweenPoints(rightEyeLeftCorner, rightEyeRightCorner);

        int commonDistance = (int) (distanceLeftCorners + distanceRightCorners) / 2;

        Rect leftEyeBall = new Rect((int) leftEyeLeftCorner.x, (int) leftEyeLeftCorner.y - commonDistance / 2,
                commonDistance, commonDistance);

        Rect rightEyeBall = new Rect((int) rightEyeLeftCorner.x, (int) rightEyeLeftCorner.y - commonDistance / 2,
                commonDistance, commonDistance);

        this.coordinateSystemSide = commonDistance;

        Point coordinateLeft = null;
        Point coordinateRight = null;
        Point leftEyeLocal = this.leftEye.getAveragePoint();
        Point rightEyeLocal = this.rightEye.getAveragePoint();

        if (leftEyeBall.contains(leftEyeLocal)) {
            coordinateLeft = new Point(leftEyeLocal.x - leftEyeBall.x, leftEyeLocal.y - leftEyeBall.y);
        }

        if (rightEyeBall.contains(this.rightEye.getAveragePoint())) {
            coordinateRight = new Point(rightEyeLocal.x - rightEyeBall.x, rightEyeLocal.y - rightEyeBall.y);
        }

        if (coordinateLeft == null && coordinateRight == null) {
            eyeGazeCoordinate.insertNewPoint(null);
        } else {
            eyeGazeCoordinate.insertNewPoint(Helpers.centerOfPoints(coordinateLeft, coordinateRight));
        }

    }
}
