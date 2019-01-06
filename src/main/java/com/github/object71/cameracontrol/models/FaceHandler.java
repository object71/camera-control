/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol.models;

import com.github.object71.cameracontrol.common.Helpers;
import com.github.object71.cameracontrol.common.ImageProcessedListener;
import com.github.object71.cameracontrol.common.MarkPoint;
import com.github.object71.cameracontrol.common.PointHistoryCollection;

import org.opencv.core.Mat;

import java.awt.AWTException;
import java.awt.Dimension;
import java.awt.Image;
import java.awt.Robot;
import java.awt.Toolkit;
import java.util.ArrayList;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.flandmark;
import org.bytedeco.javacpp.flandmark.FLANDMARK_Model;
import org.jnativehook.GlobalScreen;
import org.jnativehook.NativeHookException;
import org.jnativehook.mouse.NativeMouseEvent;
import org.jnativehook.mouse.NativeMouseListener;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.tracking.Tracker;
import org.opencv.tracking.TrackerBoosting;
import org.opencv.videoio.VideoCapture;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author hristo
 */
public class FaceHandler implements Runnable {

	private static CascadeClassifier faceCascade;
	private static FLANDMARK_Model model;
	private static Toolkit toolkit = Toolkit.getDefaultToolkit();
	private static Robot robot;

	private final double[] faceMarks;
	private final Point[] faceMarkPoints;
	private Tracker faceTracker;
	private Rect2d trackedFace;

	public PointHistoryCollection leftEye = new PointHistoryCollection(3);
	public PointHistoryCollection rightEye = new PointHistoryCollection(3);

	public PointHistoryCollection eyeGazeCoordinate = new PointHistoryCollection(2);
	public double coordinateSystemSide = 80;

	public Thread currentThread;
	public boolean process;
	public boolean calibrateOnMouseClick;
	public boolean controlMouse;
	public boolean closing = false;
	
	public double leftBound = coordinateSystemSide;
	public double rightBound = 0;
	public double topBound = coordinateSystemSide;
	public double bottomBound = 0;

	private ArrayList<ImageProcessedListener> imageProcessedListeners = new ArrayList<ImageProcessedListener>();

	public void registerImageProcessedListener(ImageProcessedListener listener) {
		imageProcessedListeners.add(listener);
	}

	private void triggerImageProcessed(Image image) {
		for (ImageProcessedListener listener : imageProcessedListeners) {
			listener.onImageProcessed(image);
		}
	}

	public FaceHandler() {
		faceCascade = new CascadeClassifier();
		faceCascade.load(getClass().getClassLoader().getResource("data/haarcascade_frontalface_alt.xml").toString()
				.replace("file:/", ""));

		model = flandmark.flandmark_init(
				getClass().getClassLoader().getResource("data/flandmark_model.dat").toString().replace("file:/", ""));
		if (model == null) {
			System.out.println("Structure model wasn't created. Corrupted file flandmark_model.dat?");
			System.exit(1);
		}

		this.faceMarks = new double[2 * model.data().options().M()];
		this.faceMarkPoints = new Point[model.data().options().M()];

		currentThread = null;
		process = false;
		
		try {
			GlobalScreen.registerNativeHook();
		} catch (NativeHookException e1) {
			e1.printStackTrace();
		}
		
		GlobalScreen.addNativeMouseListener(new NativeMouseListener() {
			
			@Override
			public void nativeMouseClicked(NativeMouseEvent e) {
				Point currentGazePoint = eyeGazeCoordinate.getAveragePoint();
				
				if (currentGazePoint == null) {
					return;
				}
				
				if(currentGazePoint.x < leftBound) {
					leftBound = currentGazePoint.x;
				}
				
				if(currentGazePoint.x > rightBound) {
					rightBound = currentGazePoint.x;
				}
				
				if(currentGazePoint.y < topBound) {
					topBound = currentGazePoint.y;
				}
				
				if(currentGazePoint.y > bottomBound) {
					bottomBound = currentGazePoint.y;
				}
			}

			@Override
			public void nativeMousePressed(NativeMouseEvent e) {
				// TODO Auto-generated method stub
				
			}

			@Override
			public void nativeMouseReleased(NativeMouseEvent e) {
				// TODO Auto-generated method stub
				
			}
			
		});

		try {
			robot = new Robot();
		} catch (AWTException e) {
			e.printStackTrace();
		}
	}

	public void initializeFrameInformation(Mat inputFrame) {

		if (inputFrame == null || inputFrame.empty()) {
			leftEye.insertNewPoint(null);
			rightEye.insertNewPoint(null);
			eyeGazeCoordinate.insertNewPoint(null);

			inputFrame.release();
			return;
		}

		Size size = inputFrame.size();
		Size commonSize = new Size(1280, 960);
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
			leftEye.insertNewPoint(null);
			rightEye.insertNewPoint(null);
			eyeGazeCoordinate.insertNewPoint(null);

			inputFrame.release();
			return;
		}

		Rect faceLocation = null;
		if (faceTracker == null) {
			faceLocation = this.getFaceLocation(frame);
			if (faceLocation == null) {
				leftEye.insertNewPoint(null);
				rightEye.insertNewPoint(null);
				eyeGazeCoordinate.insertNewPoint(null);

				inputFrame.release();
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
			leftEye.insertNewPoint(null);
			rightEye.insertNewPoint(null);
			eyeGazeCoordinate.insertNewPoint(null);

			inputFrame.release();
			return;
		}

		try (Pointer framePtr = Helpers.getPointer(frame.clone().getNativeObjAddr());
				org.bytedeco.javacpp.opencv_core.Mat img_grayscale_mat = new org.bytedeco.javacpp.opencv_core.Mat(
						framePtr);
				org.bytedeco.javacpp.opencv_core.IplImage img_grayscale = new org.bytedeco.javacpp.opencv_core.IplImage(
						img_grayscale_mat);) {

			int[] bbox = getBoundindBox(faceLocation);
			if (flandmark.flandmark_detect(img_grayscale, bbox, model, faceMarks) != 0) {
				
				inputFrame.release();
				return;
			}

			this.marksToPoints();

		} catch (Exception e) {
			inputFrame.release();
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
		this.leftEye.insertNewPoint(leftEyeCenter);

		Point rightEyeCenter = EyeHandler.getEyeCenter(frame.submat(rightEyeBall));
		this.rightEye.insertNewPoint(rightEyeCenter);

		this.recalculateCoordinates();

		inputFrame.release();
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
		int[] bbox = new int[] { rectangle.x, rectangle.y, rectangle.x + rectangle.width,
				rectangle.y + rectangle.height };
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
//		Point leftEyeLeftCorner = this.getFaceMark(MarkPoint.LeftEyeLeftCorner);
//		Point leftEyeRightCorner = this.getFaceMark(MarkPoint.LeftEyeRightCorder);
//		Point rightEyeLeftCorner = this.getFaceMark(MarkPoint.RightEyeLeftCorner);
//		Point rightEyeRightCorner = this.getFaceMark(MarkPoint.RightEyeRightCorner);
//
//		double distanceLeftCorners = Helpers.distanceBetweenPoints(leftEyeLeftCorner, leftEyeRightCorner);
//		double distanceRightCorners = Helpers.distanceBetweenPoints(rightEyeLeftCorner, rightEyeRightCorner);
//
//		int commonDistance = (int) (distanceLeftCorners + distanceRightCorners) / 2;
//
//		Rect leftEyeBall = new Rect((int) leftEyeLeftCorner.x, (int) leftEyeLeftCorner.y - commonDistance / 2,
//				commonDistance, commonDistance);
//
//		Rect rightEyeBall = new Rect((int) rightEyeLeftCorner.x, (int) rightEyeLeftCorner.y - commonDistance / 2,
//				commonDistance, commonDistance);
//
//		this.coordinateSystemSide = commonDistance;
//
//		Point coordinateLeft = null;
//		Point coordinateRight = null;
//		Point leftEyeLocal = this.leftEye.getAveragePoint();
//		Point rightEyeLocal = this.rightEye.getAveragePoint();
//
//		if (leftEyeBall.contains(leftEyeLocal)) {
//			coordinateLeft = new Point(leftEyeLocal.x, leftEyeLocal.y);
//		}
//
//		if (rightEyeBall.contains(this.rightEye.getAveragePoint())) {
//			coordinateRight = new Point(rightEyeLocal.x, rightEyeLocal.y);
//		}
//
//		if (coordinateLeft == null && coordinateRight == null) {
//			eyeGazeCoordinate.insertNewPoint(null);
//		} else {
//			eyeGazeCoordinate.insertNewPoint(Helpers.centerOfPoints(coordinateLeft, coordinateRight));
//		}
		
		eyeGazeCoordinate.insertNewPoint(Helpers.centerOfPoints(this.leftEye.getAveragePoint(), this.rightEye.getAveragePoint()));

	}

	@Override
	public void run() {
		int cameraDevice = 0;
		VideoCapture capture = new VideoCapture(cameraDevice);

		if (!capture.isOpened()) {
			System.err.println("--(!)Error opening video capture");
			System.exit(0);
		}

		Mat frame = new Mat();

		boolean running = capture.read(frame);
		while (running && !closing) {
			if (process) {

				if (frame.empty()) {
					System.err.println("--(!) No captured frame -- Break!");
					break;
				}

				if (frame.cols() > 0 || frame.rows() > 0) {

					try {
						this.triggerImageProcessed(HighGui.toBufferedImage(frame));
						this.initializeFrameInformation(frame);
					} catch (Exception e) {
						System.out.println(e.getMessage());
					}

				}
				frame.release();
				System.gc();

				if (controlMouse) {
					this.moveMouse();
				}

				try {
					running = capture.read(frame);
				} catch (Exception e) {

				}
			}

			try {
				Thread.sleep(10);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}

		capture.release();
	}

	public void moveMouse() {
		
		Dimension screenDimensions = toolkit.getScreenSize();
		
		double screenWidth = screenDimensions.getWidth();
		double screenHeight = screenDimensions.getHeight();
		
		double kX = (rightBound - leftBound) / screenWidth;
		double kY = (bottomBound - topBound) / screenHeight;
		
		int x = 0;
		int y = 0;
		
		Point coord = this.eyeGazeCoordinate.getAveragePoint();

		if (coord == null) {
			return;
		}
		
		if(coord.x > leftBound && coord.x < rightBound && coord.y > topBound && coord.y < bottomBound) {
			x = (int) ((coord.x - leftBound) / kX);
			y = (int) ((coord.y - topBound) / kY);
		}
		
		if(coord.x > rightBound) {
			x = (int) ((rightBound - leftBound) / kX) - 1;
		}
		
		if(coord.y > bottomBound) {
			y = (int) ((bottomBound - topBound) / kY) - 1;
		}
		
		Point anchored = anchorOnRect(screenWidth, screenHeight, x, y, 4, 4);
		robot.mouseMove((int) anchored.x, (int) anchored.y);
	}
	
	private Point anchorOnRect(double width, double height, double x, double y, int rows, int cols) {
		
		double quadrantWidth = width / cols;
		double quadrantHeight = height / rows;
		
		int currentColumn = (int) Math.floor(x / quadrantWidth);
		int currentRow = (int) Math.floor(y / quadrantHeight);
		
		double anchoredX = (currentColumn * quadrantWidth) + (quadrantWidth / 2);
		double anchoredY = (currentRow * quadrantHeight) + (quadrantHeight / 2);
		
		return new Point(anchoredX, anchoredY);
	}

	public void startThread() {
		if (currentThread != null) {
			System.out.println("Thread already running");
			return;
		}

		currentThread = new Thread(this);
		currentThread.start();
	}
}
