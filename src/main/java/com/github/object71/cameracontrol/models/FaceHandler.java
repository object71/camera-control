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
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.flandmark;
import org.bytedeco.javacpp.flandmark.FLANDMARK_Model;
import org.opencv.core.CvType;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author hristo
 */
public class FaceHandler {

	private static final CascadeClassifier faceCascade;
	private static final FLANDMARK_Model model;

	private double[] faceMarks;
	private Point[] faceMarkPoints;

	public PointHistoryCollection leftEye = new PointHistoryCollection(3);
	public PointHistoryCollection rightEye = new PointHistoryCollection(3);

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
		this.faceMarkPoints = new Point[model.data().options().M() + 3];
	}

	public void initializeFrameInformation(Mat inputFrame) {
		Mat frame = new Mat(inputFrame.rows(), inputFrame.cols(), CvType.CV_64F);

		Imgproc.cvtColor(inputFrame, frame, Imgproc.COLOR_BGR2GRAY);
		Imgproc.equalizeHist(frame, frame);

		Rect faceLocation = this.getFaceLocation(frame);
		if (faceLocation == null) {
			return;
		}

		try (Pointer framePtr = Helpers.getPointer(frame.getNativeObjAddr())) {
			int[] bbox = getBoundindBox(faceLocation);
			org.bytedeco.javacpp.opencv_core.Mat img_grayscale_mat = new org.bytedeco.javacpp.opencv_core.Mat(framePtr);
			org.bytedeco.javacpp.opencv_core.IplImage img_grayscale = new org.bytedeco.javacpp.opencv_core.IplImage(
					img_grayscale_mat);

			if (flandmark.flandmark_detect(img_grayscale, bbox, model, faceMarks) != 0) {
				return;
			}

			this.marksToPoints();

		} catch (Exception e) {
			return;
		}

		Point leftEyeBioCenter = this.getFaceMark(MarkPoint.LeftEyeBioCenter);
		Point rightEyeBioCenter = this.getFaceMark(MarkPoint.RightEyeBioCenter);

		double leftEyeRadius = Helpers.distanceBetweenPoints(this.getFaceMark(MarkPoint.LeftEyeLeftCorner),
				this.getFaceMark(MarkPoint.LeftEyeRightCorder)) / 2;
		double rightEyeRadius = Helpers.distanceBetweenPoints(this.getFaceMark(MarkPoint.RightEyeLeftCorner),
				this.getFaceMark(MarkPoint.RightEyeRightCorner)) / 2;

		Rect leftEyeRegion = new Rect((int) (leftEyeBioCenter.x - leftEyeRadius),
				(int) (leftEyeBioCenter.y - leftEyeRadius), (int) leftEyeRadius * 2, (int) leftEyeRadius * 2);
		Rect rightEyeRegion = new Rect((int) (rightEyeBioCenter.x - rightEyeRadius),
				(int) (rightEyeBioCenter.y - rightEyeRadius), (int) rightEyeRadius * 2, (int) rightEyeRadius * 2);

		Point leftEyeCenter = EyeHandler.getEyeCenter(frame.submat(leftEyeRegion));

		leftEyeCenter.x += leftEyeRegion.x;
		leftEyeCenter.y += leftEyeRegion.y;
		this.leftEye.insertNewPoint(leftEyeCenter);

		Point rightEyeCenter = EyeHandler.getEyeCenter(frame.submat(rightEyeRegion));

		rightEyeCenter.x += rightEyeRegion.x;
		rightEyeCenter.y += rightEyeRegion.y;
		this.rightEye.insertNewPoint(rightEyeCenter);
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
		case LeftEyeBioCenter:
			return faceMarkPoints[8];
		case RightEyeBioCenter:
			return faceMarkPoints[9];
		case MouthCenter:
			return faceMarkPoints[10];
		default:
			return faceMarkPoints[0];
		}
	}

	private void marksToPoints() {
		int x = 0;
		for (int i = 0; i < faceMarks.length; i += 2, x++) {
			faceMarkPoints[x] = new Point(faceMarks[i], faceMarks[i + 1]);
		}

		faceMarkPoints[x++] = Helpers.centerOfPoints(faceMarkPoints[5], faceMarkPoints[1]);
		faceMarkPoints[x++] = Helpers.centerOfPoints(faceMarkPoints[2], faceMarkPoints[6]);
		faceMarkPoints[x++] = Helpers.centerOfPoints(faceMarkPoints[3], faceMarkPoints[4]);
	}

	private static int[] getBoundindBox(Rect rectangle) {
		int[] bbox = new int[] { rectangle.x, rectangle.y, rectangle.x + rectangle.width,
				rectangle.y + rectangle.height };
		return bbox;
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
}
