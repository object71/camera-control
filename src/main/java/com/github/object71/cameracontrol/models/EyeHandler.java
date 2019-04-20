package com.github.object71.cameracontrol.models;

import java.util.LinkedList;
import java.util.Queue;

import com.github.object71.cameracontrol.common.Constants;
import com.github.object71.cameracontrol.common.GradientsModel;
import com.github.object71.cameracontrol.common.Helpers;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.Indexer;

import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_core.*;

public class EyeHandler {

	public static Point getEyeCenter(Mat eyeRegionSubframe) {
		if (eyeRegionSubframe.rows() < 3 || eyeRegionSubframe.cols() < 3) {
			return null;
		}
		
		int rows = 80;
		int cols = 80;

		Mat sizedImage = resizeImage(eyeRegionSubframe, rows, cols);

		double[] frameAsDoubles = Helpers.matrixToArray(sizedImage);

		GradientsModel gradients = Helpers.computeGradient(frameAsDoubles, rows, cols);

		Mat weight = new Mat(rows, cols, CV_64F);

		double[] sum = new double[rows * cols];
		double[] weightArray = Helpers.matrixToArray(weight);

		for (int y = 0; y < weight.rows(); y++) {
			for (int x = 0; x < weight.cols(); x++) {
				calculatePointGradientValue(rows, cols, gradients, sum, weightArray, y, x);
			}
		}

		Mat out = Helpers.arrayToMatrix(sum, rows, cols, CV_32F);
		Point maxLocation = new Point();
		DoublePointer maxValue = new DoublePointer();
		minMaxLoc(out, (DoublePointer) null, maxValue, (Point) null, maxLocation, (Mat) null);

		if (Constants.enablePostProcessing) {
			Mat floodClone = new Mat(rows, cols, CV_32F);
			int coordinate = (maxLocation.y() * cols) + maxLocation.x();
			double floodThresh = sum[coordinate] * Constants.postProcessingTreshold;
			threshold(out, floodClone, floodThresh, 0.0, THRESH_TOZERO);
			Mat mask = floodKillEdges(floodClone);

			minMaxLoc(out, (DoublePointer) null, (DoublePointer) null, (Point) null, maxLocation, mask);
			return maxLocation;
		}

		eyeRegionSubframe.release();
		sizedImage.release();

		return maxLocation;
	}

	private static Mat resizeImage(Mat eyeRegionSubframe, int rows, int cols) {
		Size size = new Size(rows, cols);
		Mat sizedImage = new Mat(size, eyeRegionSubframe.type());
		resize(eyeRegionSubframe, sizedImage, size);
		return sizedImage;
	}

	private static void calculatePointGradientValue(int rows, int cols, GradientsModel gradients, double[] sum,
			double[] weightArray, int y, int x) {
		int coordinate = (y * cols) + x;

		double valueX = gradients.gradientX[coordinate];
		double valueY = gradients.gradientY[coordinate];
		if (valueX == 0.0 && valueY == 0.0) {
			return;
		}
		for (int cy = 0; cy < rows; cy++) {
			for (int cx = 0; cx < cols; cx++) {
				doSumPoints(rows, cols, sum, weightArray, y, x, coordinate, valueX, valueY, cx, cy);
			}
		}
	}

	private static void doSumPoints(int rows, int cols, double[] sum, double[] weightArray, int y, int x,
			int coordinate, double valueX, double valueY, int cx, int cy) {

		int coordinateC = (cy * cols) + cx;
		// check all other than the current value
		if (x == cx && y == cy) {
			return;
		}

		double dx = x - cx; // the distance to a certain point
		double dy = y - cy; // the distance to a certain point
		double magnitude = Math.sqrt((dx * dx) + (dy * dy)); // actual vector distance

		dx = dx / magnitude;
		dy = dy / magnitude;

		// 0 or positive
		double dotProduct = Math.max(0, dx * valueX + dy * valueY);

		double currentValue = sum[coordinateC];

		if (Constants.enableWeight) {
			double weightValue = 255 - weightArray[coordinate];
			sum[coordinateC] = currentValue + (Math.pow(dotProduct, 2) * (weightValue / Constants.weightDivisor));
		} else {
			sum[coordinateC] = currentValue + Math.pow(dotProduct, 2);
		}
	}

	private static Mat floodKillEdges(Mat matrix) {
    	 int rows = matrix.rows();
         int cols = matrix.rows();
    	
        Mat mask = new Mat(rows, cols, CV_8U, new Scalar(255, 255, 255, 255));
        Queue<Point> todo = new LinkedList<>();
        todo.add(new Point(0, 0));
       
        int currentX = 0;
        int currentY = 0;
        
        try (Indexer matrixIndexer = matrix.createIndexer(); Indexer maskIndexer = mask.createIndexer()) {
	        while (todo.size() > 0) {
	        	
	            Point currentPoint = todo.poll();
	            
	            currentX = currentPoint.x();
	            currentY = currentPoint.y();
	            
	            if (matrixIndexer.getDouble(currentY, currentX, 0) == 0.0) {
                    continue;
                }
	            
	            queueSurroundingPoints(rows, cols, todo, currentX, currentY);
	            
	            matrixIndexer.putDouble(new long[] {currentY, currentX, 0}, 0.0);
	            maskIndexer.putDouble(new long[] {currentY, currentX, 0}, 0.0);
	        }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return mask;
    }

	private static void queueSurroundingPoints(int rows, int cols, Queue<Point> todo, int currentX, int currentY) {
		if (Helpers.isPointInMatrix(currentX + 1, currentY, rows, cols)) {
		    todo.add(new Point(currentX + 1, currentY));
		}

		if (Helpers.isPointInMatrix(currentX - 1, currentY, rows, cols)) {
		    todo.add(new Point(currentX + 1, currentY));
		}

		if (Helpers.isPointInMatrix(currentX, currentY + 1, rows, cols)) {
		    todo.add(new Point(currentX, currentY + 1));
		}

		if (Helpers.isPointInMatrix(currentX, currentY - 1, rows, cols)) {
		    todo.add(new Point(currentX, currentY - 1));
		}
	}
}
