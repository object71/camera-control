/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol.models;

import com.github.object71.cameracontrol.common.Helpers;
import com.github.object71.cameracontrol.common.ImageProcessedListener;
import com.github.object71.cameracontrol.common.PointHistoryCollection;

import java.awt.AWTException;
import java.awt.Dimension;
import java.awt.Image;
import java.awt.Robot;
import java.awt.Toolkit;
import java.awt.event.InputEvent;
import java.io.UnsupportedEncodingException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Rect2d;
import org.bytedeco.javacpp.opencv_face.Facemark;
import org.bytedeco.javacpp.opencv_face.FacemarkKazemi;
import org.bytedeco.javacpp.opencv_face.FacemarkLBF;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacpp.opencv_tracking.Tracker;
import org.bytedeco.javacpp.opencv_tracking.TrackerBoosting;
import org.bytedeco.javacpp.opencv_videoio.VideoCapture;
import org.jnativehook.GlobalScreen;
import org.jnativehook.NativeHookException;
import org.jnativehook.mouse.NativeMouseEvent;
import org.jnativehook.mouse.NativeMouseListener;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;
import static org.bytedeco.javacpp.opencv_highgui.*;

/**
 *
 * @author hristo
 */
public class FaceHandler implements Runnable {

	private final static Logger LOGGER = Logger.getLogger(FaceHandler.class.getName());

	private static CascadeClassifier faceCascade;
	private static Facemark facemark;
//	private static FLANDMARK_Model model;
	private static Toolkit toolkit = Toolkit.getDefaultToolkit();
	private static Robot robot;

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
	public double brightnessValue = 0;

	public void registerImageProcessedListener(ImageProcessedListener listener) {
		imageProcessedListeners.add(listener);
	}

	private void triggerImageProcessed(Image image) {
		for (ImageProcessedListener listener : imageProcessedListeners) {
			listener.onImageProcessed(image);
		}
	}

	static {
		faceCascade = new CascadeClassifier();
		faceCascade.load(FaceHandler.class.getClassLoader().getResource("data/haarcascade_frontalface_alt.xml")
				.toString().replace("file:/", ""));

		facemark = FacemarkKazemi.create();
		//facemark = FacemarkLBF.create();
		facemark.loadModel(FaceHandler.class.getClassLoader().getResource("data/face_landmark_model.dat").toString().replace("file:/", ""));
	}

	public FaceHandler() {

		currentThread = null;
		process = false;

		try {
			// Get the logger for "org.jnativehook" and set the level to off.
			Logger logger = Logger.getLogger(GlobalScreen.class.getPackage().getName());
			logger.setLevel(Level.OFF);

			// Don't forget to disable the parent handlers.
			logger.setUseParentHandlers(false);

			GlobalScreen.registerNativeHook();
		} catch (NativeHookException e1) {
			e1.printStackTrace();
		}

		GlobalScreen.addNativeMouseListener(new NativeMouseListener() {

			@Override
			public void nativeMouseClicked(NativeMouseEvent e) {
				if (calibrateOnMouseClick) {
					Point currentGazePoint = eyeGazeCoordinate.getAveragePoint();

					if (currentGazePoint == null) {
						return;
					}

					if (currentGazePoint.x() < leftBound) {
						leftBound = currentGazePoint.x();
					}

					if (currentGazePoint.x() > rightBound) {
						rightBound = currentGazePoint.x();
					}

					if (currentGazePoint.y() < topBound) {
						topBound = currentGazePoint.y();
					}

					if (currentGazePoint.y() > bottomBound) {
						bottomBound = currentGazePoint.y();
					}
				}
			}

			@Override
			public void nativeMousePressed(NativeMouseEvent e) {
			}

			@Override
			public void nativeMouseReleased(NativeMouseEvent e) {
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

		flip(inputFrame, inputFrame, 1);

		Mat frame = new Mat(inputFrame.rows(), inputFrame.cols(), CV_64F);

		cvtColor(inputFrame, frame, COLOR_BGR2GRAY);
		equalizeHist(frame, frame);

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
			} else {
				faceLocation = this.getFaceLocation(frame);
				if (faceLocation == null) {
					leftEye.insertNewPoint(null);
					rightEye.insertNewPoint(null);
					eyeGazeCoordinate.insertNewPoint(null);

					inputFrame.release();
					return;
				}

				trackedFace = Helpers.RectToRect2d(faceLocation);
				faceTracker.init(inputFrame, trackedFace);
			}
		}

		if (faceLocation == null) {
			leftEye.insertNewPoint(null);
			rightEye.insertNewPoint(null);
			eyeGazeCoordinate.insertNewPoint(null);

			inputFrame.release();
			return;
		}

		Point2fVectorVector faceMarks = new Point2fVectorVector();
		if (facemark.fit(frame, new RectVector(faceLocation), faceMarks)) {

			Point2fVector firstFace = faceMarks.get(0);
			Point[] points = new Point[(int) firstFace.size()];
			for (int i = 0; i < firstFace.size(); i++) {
				points[i] = new Point((int) firstFace.get(i).x(), (int) firstFace.get(i).y());
			}

			// left eye interest points
			Point leftEyeLeftCorner = points[36];
			Point leftEyeTopCorner = Helpers.centerOfPoints(points[37], points[38]);
			Point leftEyeRightCorner = points[39];
			Point leftEyeBottomCorner = Helpers.centerOfPoints(points[40], points[41]);

			// right eye interest points
			Point rightEyeLeftCorner = points[42];
			Point rightEyeTopCorner = Helpers.centerOfPoints(points[43], points[44]);
			Point rightEyeRightCorner = points[45];
			Point rightEyeBottomCorner = Helpers.centerOfPoints(points[46], points[47]);

			double distanceLeftCorners = Helpers.distanceBetweenPoints(leftEyeLeftCorner, leftEyeRightCorner);
			double distanceLeftEyeLids = Helpers.distanceBetweenPoints(leftEyeTopCorner, leftEyeBottomCorner);
			double distanceRightCorners = Helpers.distanceBetweenPoints(rightEyeLeftCorner, rightEyeRightCorner);
			double distanceRightEyeLids = Helpers.distanceBetweenPoints(rightEyeTopCorner, rightEyeBottomCorner);

			Rect leftEyeBall = new Rect((int) leftEyeLeftCorner.x(), (int) leftEyeTopCorner.y(),
					(int) distanceLeftCorners, (int) distanceLeftEyeLids);

			Rect rightEyeBall = new Rect((int) rightEyeLeftCorner.x(), (int) rightEyeTopCorner.y(),
					(int) distanceRightCorners, (int) distanceRightEyeLids);

			Point leftEyeResult = EyeHandler.getEyeCenter(frame.apply(leftEyeBall));
			Point rightEyeResult = EyeHandler.getEyeCenter(frame.apply(rightEyeBall));

			boolean isLeftEyeBlinking = false;
			boolean isRightEyeBlinking = false;

			Point leftEyeCenter = null;
			if (!isLeftEyeBlinking) {
				leftEyeCenter = leftEyeResult;
			}
			this.leftEye.insertNewPoint(leftEyeCenter);

			Point rightEyeCenter = null;
			if (!isRightEyeBlinking) {
				rightEyeCenter = rightEyeResult;
			}
			this.rightEye.insertNewPoint(rightEyeCenter);

			if (controlMouse && isLeftEyeBlinking && !isRightEyeBlinking) {
				int defaultAutoEvent = robot.getAutoDelay();
				robot.setAutoDelay(200);
				robot.mousePress(InputEvent.BUTTON1_MASK);
				robot.mouseRelease(InputEvent.BUTTON1_MASK);
				robot.setAutoDelay(defaultAutoEvent);
			}

			if (controlMouse && isRightEyeBlinking && !isLeftEyeBlinking) {
				int defaultAutoEvent = robot.getAutoDelay();
				robot.setAutoDelay(200);
				robot.mousePress(InputEvent.BUTTON2_MASK);
				robot.mouseRelease(InputEvent.BUTTON2_MASK);
				robot.setAutoDelay(defaultAutoEvent);
			}

			this.recalculateCoordinates();
		}
		inputFrame.release();
		frame.release();
	}

	private static int[] getBoundindBox(Rect rectangle) {
		int[] bbox = new int[] { rectangle.x(), rectangle.y(), rectangle.x() + rectangle.width(),
				rectangle.y() + rectangle.height() };
		return bbox;
	}

	private Rect getFaceLocation(Mat frame) {

		if (faceCascade.empty()) {
			throw new RuntimeException("Face cascade is empty.");
		}
		RectVector faces = new RectVector();

		faceCascade.detectMultiScale(frame, faces, 1.1, 2, CASCADE_DO_CANNY_PRUNING | CASCADE_FIND_BIGGEST_OBJECT,
				new Size(), frame.size());

		double maxDistance = 0;
		Rect faceLocation = null;
		if (faces.size() == 0) {
			return null;
		}

		for (int i = 0; i < faces.size(); i++) {
			Rect face = faces.get(i);
			Point faceCenter = Helpers.centerOfRect(face);
			Point imageCenter = Helpers.centerOfRect(new Rect(0, 0, frame.cols(), frame.rows()));
			double distance = Helpers.distanceBetweenPoints(faceCenter, imageCenter);
			if (distance > maxDistance) {
				maxDistance = distance;
				faceLocation = face;
			}
		}

		return faceLocation;
	}

	public void recalculateCoordinates() {
		eyeGazeCoordinate.insertNewPoint(
				Helpers.centerOfPoints(this.leftEye.getAveragePoint(), this.rightEye.getAveragePoint()));
	}

	@Override
	public void run() {
		int cameraDevice = 0;
		try (VideoCapture capture = new VideoCapture(cameraDevice)) {

			if (!capture.isOpened()) {
				LOGGER.log(Level.SEVERE, "--(!)Error opening video capture");
				System.exit(0);
			}

			Mat frame = new Mat();

			boolean running = capture.read(frame);
			while (running && !closing) {
				if (process) {

					if (frame.empty()) {
						LOGGER.log(Level.SEVERE, "--(!) No captured frame -- Break!");
						break;
					}

					int frameWidth = frame.cols();
					int frameHeight = frame.rows();

					if (frameWidth > 0 || frameHeight > 0) {

						Helpers.bright(frame, brightnessValue);

						Size size = new Size(480, 320);
						if (frameWidth > size.width()) {
							int x = (int) (frameWidth - size.width()) / 2;
							int y = (int) (frameHeight - size.height()) / 2;
							Mat forDelete = frame;
							frame = frame.apply(new Rect(x, y, (int) size.width(), (int) size.height()));
							forDelete.release();
						} else if (frameWidth < size.width()) {
							Mat resizedFrame = new Mat(size, frame.type());
							resize(frame, resizedFrame, size);
							frame.release();
							frame = resizedFrame;
						}

						try {
//							this.triggerImageProcessed(toBufferedImage(frame));
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
		}
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

		if (coord.x() > leftBound && coord.x() < rightBound && coord.y() > topBound && coord.y() < bottomBound) {
			x = (int) ((coord.x() - leftBound) / kX);
			y = (int) ((coord.y() - topBound) / kY);
		}

		if (coord.x() > rightBound) {
			x = (int) ((rightBound - leftBound) / kX) - 1;
		}

		if (coord.y() > bottomBound) {
			y = (int) ((bottomBound - topBound) / kY) - 1;
		}

		Point anchored = anchorOnRect(screenWidth, screenHeight, x, y, 3, 3);
		robot.mouseMove((int) anchored.x(), (int) anchored.y());
	}

	private Point anchorOnRect(double width, double height, double x, double y, int rows, int cols) {

		double quadrantWidth = width / cols;
		double quadrantHeight = height / rows;

		int currentColumn = (int) Math.floor(x / quadrantWidth);
		int currentRow = (int) Math.floor(y / quadrantHeight);

		double anchoredX = (currentColumn * quadrantWidth) + (quadrantWidth / 2);
		double anchoredY = (currentRow * quadrantHeight) + (quadrantHeight / 2);

		return new Point((int) anchoredX, (int) anchoredY);
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
