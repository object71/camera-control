/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol.models;

import com.github.object71.cameracontrol.common.Helpers;
import com.github.object71.cameracontrol.common.ImageProcessedListener;
import com.github.object71.cameracontrol.common.PointHistoryCollection;

import org.opencv.core.Mat;

import java.awt.AWTException;
import java.awt.Dimension;
import java.awt.Image;
import java.awt.Robot;
import java.awt.Toolkit;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.flandmark;
import org.bytedeco.javacpp.flandmark.FLANDMARK_Model;
import org.jnativehook.GlobalScreen;
import org.jnativehook.NativeHookException;
import org.jnativehook.mouse.NativeMouseEvent;
import org.jnativehook.mouse.NativeMouseListener;
import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
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

//		Size size = inputFrame.size();
//		Size commonSize = new Size(1280, 960);
//		if (size.width != commonSize.width || size.height != commonSize.height) {
//			Mat resizedFrame = new Mat(commonSize, inputFrame.type());
//			Imgproc.resize(inputFrame, resizedFrame, commonSize);
//			Mat old = inputFrame;
//			inputFrame = resizedFrame;
//			old.release();
//		}

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
		
		Map<String, Point> marks;

		try (Pointer framePtr = Helpers.getPointer(frame.clone().getNativeObjAddr());
				org.bytedeco.javacpp.opencv_core.Mat img_grayscale_mat = new org.bytedeco.javacpp.opencv_core.Mat(
						framePtr);
				org.bytedeco.javacpp.opencv_core.IplImage img_grayscale = new org.bytedeco.javacpp.opencv_core.IplImage(
						img_grayscale_mat);) {

			int[] bbox = getBoundindBox(faceLocation);
			double[] faceMarks = new double[2 * model.data().options().M()];
			if (flandmark.flandmark_detect(img_grayscale, bbox, model, faceMarks) != 0) {
				
				inputFrame.release();
				return;
			}
			
			

			marks = this.marksToPoints(faceMarks);

		} catch (Exception e) {
			inputFrame.release();
			return;
		}

		Point leftEyeLeftCorner = marks.get("LeftEyeLeftCorner");
		Point leftEyeRightCorner = marks.get("LeftEyeRightCorner");
		Point rightEyeLeftCorner = marks.get("RightEyeLeftCorner");
		Point rightEyeRightCorner = marks.get("RightEyeRightCorner");

		double distanceLeftCorners = Helpers.distanceBetweenPoints(leftEyeLeftCorner, leftEyeRightCorner);
		double distanceRightCorners = Helpers.distanceBetweenPoints(rightEyeLeftCorner, rightEyeRightCorner);

		int commonDistance = (int) (distanceLeftCorners + distanceRightCorners) / 2;

		Rect leftEyeBall = new Rect((int) leftEyeLeftCorner.x, (int) leftEyeLeftCorner.y - commonDistance / 2,
				commonDistance, commonDistance);

		Rect rightEyeBall = new Rect((int) rightEyeLeftCorner.x, (int) rightEyeLeftCorner.y - commonDistance / 2,
				commonDistance, commonDistance);
		
		MinMaxLocResult leftEyeResult = EyeHandler.getEyeCenter(frame.submat(leftEyeBall));
		Point leftEyeCenter = leftEyeResult.maxLoc;
		this.leftEye.insertNewPoint(leftEyeCenter);

		MinMaxLocResult rightEyeResult = EyeHandler.getEyeCenter(frame.submat(rightEyeBall));
		Point rightEyeCenter = rightEyeResult.maxLoc;
		this.rightEye.insertNewPoint(rightEyeCenter);

		this.recalculateCoordinates();

		inputFrame.release();
		frame.release();
	}

	private Map<String, Point> marksToPoints(double[] faceMarks) {
		Map<String, Point> marks = new HashMap<String, Point>();
		
		// left eye interest points
		marks.put("LeftEyeLeftCorner", new Point(faceMarks[10], faceMarks[11]));
//		marks.put("LeftEyeTopCorner", Helpers.centerOfPoints(faceMarks[22], faceMarks[23], faceMarks[24], faceMarks[25]));
		marks.put("LeftEyeRightCorner", new Point(faceMarks[2], faceMarks[3]));
//		marks.put("LeftEyeBottomCorner", Helpers.centerOfPoints(faceMarks[28], faceMarks[29], faceMarks[30], faceMarks[31]));
		
		// right eye interest points
		marks.put("RightEyeLeftCorner", new Point(faceMarks[4], faceMarks[5]));
//		marks.put("RightEyeTopCorner", Helpers.centerOfPoints(faceMarks[34], faceMarks[35], faceMarks[36], faceMarks[37]));
		marks.put("RightEyeRightCorner", new Point(faceMarks[12], faceMarks[13]));
//		marks.put("RightEyeBottomCorner", Helpers.centerOfPoints(faceMarks[40], faceMarks[41], faceMarks[42], faceMarks[43]));
		
		return marks;
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
				
				int frameWidth = frame.cols();
				int frameHeight = frame.rows();

				if (frameWidth > 0 || frameHeight > 0) {
					
					Size size = new Size(480, 320);
					if(frameWidth > size.width) {
						int x = (int) (frameWidth - size.width) / 2;
						int y = (int) (frameHeight - size.height) / 2;
						Mat forDelete = frame;
						frame = frame.submat(new Rect(x, y, (int)size.width, (int)size.height));
						forDelete.release();
					} else if(frameWidth < size.width) {
						Mat resizedFrame = new Mat(size, frame.type());
						Imgproc.resize(frame, resizedFrame, size);
						frame.release();
						frame = resizedFrame;
					}
					
					
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
		
		Point anchored = anchorOnRect(screenWidth, screenHeight, x, y, 3, 3);
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
