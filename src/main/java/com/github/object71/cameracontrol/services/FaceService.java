/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol.services;

import com.github.object71.cameracontrol.common.Helpers;
import com.github.object71.cameracontrol.models.EyeModel;
import com.github.object71.cameracontrol.common.ImageProcessedListener;
import com.github.object71.cameracontrol.common.PointHistoryCollection;

import java.awt.AWTException;
import java.awt.Dimension;
import java.awt.Image;
import java.awt.Robot;
import java.awt.Toolkit;
import java.awt.event.InputEvent;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Rect2d;
import org.bytedeco.javacpp.opencv_face.Facemark;
import org.bytedeco.javacpp.opencv_face.FacemarkKazemi;
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

/**
 *
 * @author hristo
 */
public class FaceService implements Runnable {

	private final static Logger LOGGER = Logger.getLogger(FaceService.class.getName());

	private static CascadeClassifier faceCascade;
	private static Facemark facemark;
	private static Toolkit toolkit = Toolkit.getDefaultToolkit();
	private static Robot robot;

	private Tracker faceTracker;
	private Rect2d trackedFace;

	public PointHistoryCollection historyLEP = new PointHistoryCollection(3);
	public PointHistoryCollection historyREP = new PointHistoryCollection(3);

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
		faceCascade.load(FaceService.class.getClassLoader().getResource("data/haarcascade_frontalface_alt.xml")
				.toString().replace("file:", ""));

		facemark = FacemarkKazemi.create();
		// facemark = FacemarkLBF.create();
		facemark.loadModel(FaceService.class.getClassLoader().getResource("data/face_landmark_model.dat").toString()
				.replace("file:", ""));
	}

	public FaceService() {

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

	public void processFrame(Mat inputFrame) {

		if (inputFrame == null || inputFrame.empty()) {
			breakProcessCleanup(inputFrame);
			return;
		}

		Mat frame = modifyFrame(inputFrame);

		if (frame.empty()) {
			breakProcessCleanup(inputFrame);
			return;
		}

		Rect faceLocation = locateFace(frame);

		if (faceLocation == null) {
			breakProcessCleanup(inputFrame);
			return;
		}

		Point2fVectorVector faceMarks = new Point2fVectorVector();
		if (facemark.fit(frame, new RectVector(faceLocation), faceMarks)) {
			processEyes(frame, faceMarks);
		}
		
		inputFrame.release();
		frame.release();
	}

	private Mat modifyFrame(Mat inputFrame) {
		// flip(inputFrame, inputFrame, 1);

		Mat frame = new Mat(inputFrame.rows(), inputFrame.cols(), CV_64F);

		cvtColor(inputFrame, frame, COLOR_BGR2GRAY);
		equalizeHist(frame, frame);
		return frame;
	}

	private void breakProcessCleanup(Mat inputFrame) {
		historyLEP.insertNewPoint(null);
		historyREP.insertNewPoint(null);
		eyeGazeCoordinate.insertNewPoint(null);

		inputFrame.release();
	}

	private Rect locateFace(Mat frame) {
		Rect faceLocation = null;
		if (faceTracker == null) {
			faceLocation = this.getFaceLocation(frame);
			if (faceLocation == null) {
				return faceLocation;
			}

			trackedFace = Helpers.RectToRect2d(faceLocation);

			faceTracker = TrackerBoosting.create();
			faceTracker.init(frame, trackedFace);
		} else {
			if (faceTracker.update(frame, trackedFace)) {
				faceLocation = Helpers.Rect2dToRect(trackedFace);
			} else {
				faceLocation = this.getFaceLocation(frame);
				if (faceLocation == null) {
					return faceLocation;
				}

				trackedFace = Helpers.RectToRect2d(faceLocation);
				faceTracker.init(frame, trackedFace);
			}
		}
		return faceLocation;
	}

	private void processEyes(Mat frame, Point2fVectorVector faceMarks) {
		Point2fVector firstFace = faceMarks.get(0);
		Point[] points = new Point[(int) firstFace.size()];
		for (int i = 0; i < firstFace.size(); i++) {
			points[i] = new Point((int) firstFace.get(i).x(), (int) firstFace.get(i).y());
		}

		EyeModel leftEye = new EyeModel();
		EyeModel rightEye = new EyeModel();
		
		initializeEyeModels(points, leftEye, rightEye);
		registerEyeCenters(frame, leftEye, rightEye);
		processBlinks(leftEye.getIsBlinking(), rightEye.getIsBlinking());

		this.recalculateCoordinates();
	}

	private void registerEyeCenters(Mat frame, EyeModel leftEye, EyeModel rightEye) {
		Rect leftEyeBall = new Rect((int) leftEye.leftCorner.x(), (int) leftEye.topCorner.y(),
				(int) leftEye.getCornersDistance(), (int) leftEye.getLidsDistance());

		Rect rightEyeBall = new Rect((int) rightEye.leftCorner.x(), (int) rightEye.topCorner.y(),
				(int) rightEye.getCornersDistance(), (int) rightEye.getLidsDistance());

		Point leftEyeResult = EyeService.getEyeCenter(frame.apply(leftEyeBall));
		Point rightEyeResult = EyeService.getEyeCenter(frame.apply(rightEyeBall));

		Point leftEyeCenter = null;
		if (!leftEye.getIsBlinking()) {
			leftEyeCenter = leftEyeResult;
		}
		this.historyLEP.insertNewPoint(leftEyeCenter);

		Point rightEyeCenter = null;
		if (!rightEye.getIsBlinking()) {
			rightEyeCenter = rightEyeResult;
		}
		this.historyREP.insertNewPoint(rightEyeCenter);
	}

	private void initializeEyeModels(Point[] points, EyeModel leftEye, EyeModel rightEye) {
		// left eye interest points
		leftEye.leftCorner = points[36];
		leftEye.topCorner = Helpers.centerOfPoints(points[37], points[38]);
		leftEye.rightCorner = points[39];
		leftEye.bottomCorner = Helpers.centerOfPoints(points[40], points[41]);

		// right eye interest points
		rightEye.leftCorner = points[42];
		rightEye.topCorner = Helpers.centerOfPoints(points[43], points[44]);
		rightEye.rightCorner = points[45];
		rightEye.bottomCorner = Helpers.centerOfPoints(points[46], points[47]);
	}

	private void processBlinks(boolean isLeftEyeBlinking, boolean isRightEyeBlinking) {
		if (controlMouse && isLeftEyeBlinking && !isRightEyeBlinking) {
			LOGGER.log(Level.INFO, "Left click");
			if (controlMouse) {
				int defaultAutoEvent = robot.getAutoDelay();
				robot.setAutoDelay(200);
				robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);
				robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
				robot.setAutoDelay(defaultAutoEvent);
			}
		}

		if (isRightEyeBlinking && !isLeftEyeBlinking) {
			LOGGER.log(Level.INFO, "Right click");
			if (controlMouse) {
				int defaultAutoEvent = robot.getAutoDelay();
				robot.setAutoDelay(200);
				robot.mousePress(InputEvent.BUTTON3_DOWN_MASK);
				robot.mouseRelease(InputEvent.BUTTON3_DOWN_MASK);
				robot.setAutoDelay(defaultAutoEvent);
			}
		}

		if (isRightEyeBlinking && isLeftEyeBlinking) {
			LOGGER.log(Level.INFO, "Normal blink");
		}
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
				Helpers.centerOfPoints(this.historyLEP.getAveragePoint(), this.historyREP.getAveragePoint()));
	}

	@Override
	public void run() {
		int cameraDevice = 0;
		try (VideoCapture capture = new VideoCapture(cameraDevice)) {

			if (!capture.isOpened()) {
				LOGGER.log(Level.SEVERE, "--(!)Error opening video capture");
				System.exit(0);
			}

			

			boolean running = true;
			do {
				if (process) {
					running = eyeControlProcess(capture);
				}

				try {
					Thread.sleep(10);
				} catch (InterruptedException e) {
					LOGGER.log(Level.WARNING, e.getMessage());
				}
			} while (running && !closing);
		}
	}

	private boolean eyeControlProcess(VideoCapture capture) {
		boolean running;
		Mat frame = new Mat();
		running = capture.read(frame);
		
		if (!running || frame.empty()) {
			LOGGER.log(Level.SEVERE, "--(!) No captured frame -- Break!");
			throw new RuntimeException("No more frame");
		}

		int frameWidth = frame.cols();
		int frameHeight = frame.rows();

		if (frameWidth > 0 || frameHeight > 0) {

			frame = commonFrameRefactor(frame, frameWidth, frameHeight);

			try {
				this.triggerImageProcessed(Helpers.toBufferedImage(frame));
				this.processFrame(frame);
			} catch (Exception e) {
				LOGGER.log(Level.SEVERE, e.getMessage());
			}

		}
		
		frame.release();
		System.gc();

		if (controlMouse) {
			this.moveMouse();
		}
		return running;
	}

	private Mat commonFrameRefactor(Mat frame, int frameWidth, int frameHeight) {
		Helpers.bright(frame, brightnessValue);

		Size size = new Size(640, 480);
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
		return frame;
	}

	public void moveMouse() {
		Point anchored = calculateAnchoredPoint();

		if (anchored == null) {
			return;
		}

		robot.mouseMove((int) anchored.x(), (int) anchored.y());
	}

	private Point calculateAnchoredPoint() {
		Dimension screenDimensions = toolkit.getScreenSize();

		double screenWidth = screenDimensions.getWidth();
		double screenHeight = screenDimensions.getHeight();

		double kX = (rightBound - leftBound) / screenWidth;
		double kY = (bottomBound - topBound) / screenHeight;
		Point coord = this.eyeGazeCoordinate.getAveragePoint();
		if (coord == null) {
			return null;
		}

		return validateBounds(screenWidth, screenHeight, kX, kY, coord);
	}

	private Point validateBounds(double screenWidth, double screenHeight, double kX, double kY, Point coord) {
		Point anchored = new Point();

		if (coord.x() > leftBound && coord.x() < rightBound && coord.y() > topBound && coord.y() < bottomBound) {
			anchored.x((int) ((coord.x() - leftBound) / kX));
			anchored.y((int) ((coord.y() - topBound) / kY));
		}

		if (coord.x() > rightBound) {
			anchored.x((int) ((rightBound - leftBound) / kX) - 1);
		}

		if (coord.y() > bottomBound) {
			anchored.y((int) ((bottomBound - topBound) / kY) - 1);
		}
		
		anchored = anchorOnRect(screenWidth, screenHeight, anchored.x(), anchored.y(), 3, 3);
		return anchored;
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
			LOGGER.log(Level.WARNING, "Thread already running");
			return;
		}

		currentThread = new Thread(this);
		currentThread.start();
	}
}
