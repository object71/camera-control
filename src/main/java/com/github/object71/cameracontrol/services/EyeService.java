package com.github.object71.cameracontrol.services;

import java.util.LinkedList;
import java.util.Queue;

import com.github.object71.cameracontrol.common.Constants;
import com.github.object71.cameracontrol.models.GradientsModel;
import com.github.object71.cameracontrol.common.Helpers;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.indexer.Indexer;

import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_core.*;

public class EyeService {

	public static Point getEyeCenter(Mat eyeRegionSubframe) {
		
		if (eyeRegionSubframe.rows() < 3 || eyeRegionSubframe.cols() < 3) {
			return null;
		}
		
		int rows = 80;
		int cols = 80;

		Mat sizedImage = resizeImage(eyeRegionSubframe, rows, cols);
		double[] frameAsDoubles = Helpers.matrixToArray(sizedImage);

		// compute X and Y gradients
		GradientsModel gradients = Helpers.computeGradient(frameAsDoubles, rows, cols);

		// sum up both gradients and increase value
		// if for being at even distances from other points
		double[] sum = new double[rows * cols];
		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < cols; x++) {
				calculatePointGradientValue(rows, cols, gradients, sum, y, x);
			}
		}

		// get the maximum value from the gradients sum
		Mat out = Helpers.arrayToMatrix(sum, rows, cols, CV_32F);
		Point maxLocation = new Point();
		DoublePointer maxValue = new DoublePointer();
		minMaxLoc(out, (DoublePointer) null, maxValue, (Point) null, maxLocation, (Mat) null);

		if (Constants.enablePostProcessing) {
			Mat floodClone = new Mat(rows, cols, CV_32F);
			int coordinate = (maxLocation.y() * cols) + maxLocation.x();
			
			// set a percent of the current maximum as a minimum treshold
			// with tresh to zero any values below the treshold will be set to zero
			// other values will stay as they are
			double floodThresh = sum[coordinate] * Constants.postProcessingTreshold;
			threshold(out, floodClone, floodThresh, 0.0, THRESH_TOZERO);
			
			// set any values at the edges of the image to 0 
			// we already know they are not a possible center
			Mat mask = floodKillEdges(floodClone);

			// use the mask with the zeroed out edges and treshold values
			// and test again for the maximum value
			minMaxLoc(out, (DoublePointer) null, (DoublePointer) null, (Point) null, maxLocation, mask);
			return maxLocation;
		}

		// cleanup
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
			int y, int x) {
		int coordinate = (y * cols) + x;

		double valueX = gradients.gradientX[coordinate];
		double valueY = gradients.gradientY[coordinate];
		
		// cut gradients that cannot be a possible center of a circle gradient
		if (valueX == 0.0 || valueY == 0.0) {
			return;
		}
		
		// boost sum values by the distance of other points and its value
		for (int cy = 0; cy < rows; cy++) {
			for (int cx = 0; cx < cols; cx++) {
				doSumPoints(rows, cols, sum, y, x, coordinate, valueX, valueY, cx, cy);
			}
		}
	}

	private static void doSumPoints(int rows, int cols, double[] sum, int y, int x,
			int coordinate, double valueX, double valueY, int cx, int cy) {

		int coordinateC = (cy * cols) + cx;
		
		// ignore of the two coordinates are the same
		if (x == cx && y == cy) {
			return;
		}

		// calculate the distance to the selected point
		double dx = x - cx;
		double dy = y - cy;
		
		// actual vector size
		double magnitude = Math.sqrt((dx * dx) + (dy * dy)); 
		
		// calculating the dot product
		dx = dx / magnitude;
		dy = dy / magnitude;
		double dotProduct = Math.max(0, dx * valueX + dy * valueY);
		
		// sum up the dot product with the previous dot products for the current points
		sum[coordinateC] += Math.pow(dotProduct, 2);
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
