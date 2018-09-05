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
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.tracking.Tracker;
import org.opencv.tracking.TrackerBoosting;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author hristo
 */
public class FaceHandler {

	private static final CascadeClassifier faceCascade;
	private static final CascadeClassifier rightEyeCascade;
	private static final CascadeClassifier leftEyeCascade;
	private static final CascadeClassifier mouthCascade;
	private static final CascadeClassifier noseCascade;

	private static Tracker noseTracker;
	private static Tracker mouthTracker;

	public PointHistoryCollection leftEye = new PointHistoryCollection(3);
	public PointHistoryCollection rightEye = new PointHistoryCollection(3);
	public PointHistoryCollection mouth = new PointHistoryCollection(3);
	public PointHistoryCollection nose = new PointHistoryCollection(3);

	static {
		faceCascade = new CascadeClassifier();
		faceCascade.load("./src/main/resources/haarcascade_frontalface_alt.xml");

		leftEyeCascade = new CascadeClassifier();
		leftEyeCascade.load("./src/main/resources/haarcascade_lefteye_2splits.xml");

		rightEyeCascade = new CascadeClassifier();
		rightEyeCascade.load("./src/main/resources/haarcascade_righteye_2splits.xml");

		mouthCascade = new CascadeClassifier();
		mouthCascade.load("./src/main/resources/haarcascade_mouth.xml");

		noseCascade = new CascadeClassifier();
		noseCascade.load("./src/main/resources/haarcascade_nose.xml");

		noseTracker = null;
		mouthTracker = null;
	}

	public FaceHandler() {

	}

	public void initializeFrameInformation(Mat inputFrame) {
		Helpers.bright(inputFrame, 25);

		Mat frame = new Mat(inputFrame.rows(), inputFrame.cols(), CvType.CV_64F);

		Imgproc.cvtColor(inputFrame, frame, Imgproc.COLOR_BGR2GRAY);
		Imgproc.equalizeHist(frame, frame);

		Rect faceLocation = this.getFaceLocation(frame);
		if (faceLocation == null) {
			return;
		}

		Mat faceSubframe = frame.submat(faceLocation);

		if (Constants.smoothFaceImage) {
			double sigma = Constants.smoothFaceFactor * faceSubframe.width();
			Imgproc.GaussianBlur(faceSubframe, faceSubframe, new Size(), sigma);
		}

		// Initialise variables
		double faceLeftSide = 0;
		MatOfRect leftEyeDetections = new MatOfRect();

		double faceRightSide = faceSubframe.width();
		MatOfRect rightEyeDetections = new MatOfRect();

		// Detect eyes
		leftEyeCascade.detectMultiScale(faceSubframe, leftEyeDetections, 1.05, 0, Objdetect.CASCADE_DO_CANNY_PRUNING,
				new Size(), new Size(faceSubframe.width(), faceSubframe.height()));

		rightEyeCascade.detectMultiScale(faceSubframe, rightEyeDetections, 1.05, 0, Objdetect.CASCADE_DO_CANNY_PRUNING,
				new Size(), new Size(faceSubframe.width(), faceSubframe.height()));

		// Get eye centres
		if (!leftEyeDetections.empty()) {
			Rect leftEyeRegion = getEyeLocation(leftEyeDetections.toArray(), faceLeftSide, faceRightSide);
			if (leftEyeRegion != null) {
				Point leftEyeCenter = EyeHandler.getEyeCenter(faceSubframe, leftEyeRegion);

				leftEyeCenter.x += faceLocation.x + leftEyeRegion.x;
				leftEyeCenter.y += faceLocation.y + leftEyeRegion.y;
				this.leftEye.insertNewPoint(leftEyeCenter);
			}
		}

		if (!rightEyeDetections.empty()) {
			Rect rightEyeRegion = getEyeLocation(rightEyeDetections.toArray(), faceRightSide, faceLeftSide);

			if (rightEyeRegion != null) {
				Point rightEyeCenter = EyeHandler.getEyeCenter(faceSubframe, rightEyeRegion);

				rightEyeCenter.x += faceLocation.x + rightEyeRegion.x;
				rightEyeCenter.y += faceLocation.y + rightEyeRegion.y;
				this.rightEye.insertNewPoint(rightEyeCenter);
			}
		}

		MatOfRect mouthDetections = new MatOfRect();
		mouthCascade.detectMultiScale(faceSubframe, mouthDetections, 1.05, 3, Objdetect.CASCADE_DO_CANNY_PRUNING | Objdetect.CASCADE_DO_ROUGH_SEARCH, new Size(),
				new Size(faceSubframe.width(), faceSubframe.height()));

		if (!mouthDetections.empty()) {
			Point location = Helpers.centerOfRect(mouthDetections.toArray()[0]);
			
			location.x += faceLocation.x;
			location.y += faceLocation.y;
			
			this.mouth.insertNewPoint(location);
		}

		MatOfRect noseDetections = new MatOfRect();
		noseCascade.detectMultiScale(faceSubframe, noseDetections, 1.05, 3, Objdetect.CASCADE_DO_CANNY_PRUNING | Objdetect.CASCADE_DO_ROUGH_SEARCH, new Size(),
				new Size(faceSubframe.width(), faceSubframe.height()));

		if (!noseDetections.empty()) {
			Point location = Helpers.centerOfRect(noseDetections.toArray()[0]);
			
			location.x += faceLocation.x;
			location.y += faceLocation.y;
			
			this.nose.insertNewPoint(location);
		}

	}

	private Rect getFaceLocation(Mat frame) {
		MatOfRect faces = new MatOfRect();

		faceCascade.detectMultiScale(frame, faces, 1.1, 2,
				Objdetect.CASCADE_DO_CANNY_PRUNING | Objdetect.CASCADE_FIND_BIGGEST_OBJECT,
				new Size(frame.size().width / 25, frame.size().height / 25), frame.size());

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

}
