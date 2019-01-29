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

		int rows = 80;
		int cols = 80;

		Mat sizedImage = new Mat();
		Size size = new Size(rows, cols);
		resize(eyeRegionSubframe, sizedImage, size);

		if (rows < 3 || cols < 3) {
			return null;
		}

		double[] frameAsDoubles = Helpers.matrixToArray(sizedImage);

		GradientsModel gradients = Helpers.computeGradient(frameAsDoubles, rows, cols);

		Mat weight = new Mat(rows, cols, CV_64F);
//		GaussianBlur(sizedImage, weight, new Size(Constants.weightBlurSize, Constants.weightBlurSize), 0, 0);

		double[] sum = new double[rows * cols];
		double[] weightArray = Helpers.matrixToArray(weight);

		for (int y = 0; y < weight.rows(); y++) {
			for (int x = 0; x < weight.cols(); x++) {
				int coordinate = (y * cols) + x;

				double valueX = gradients.gradientX[coordinate];
				double valueY = gradients.gradientY[coordinate];
				if (valueX == 0.0 && valueY == 0.0) {
					continue;
				}

				for (int cy = 0; cy < rows; cy++) {
					for (int cx = 0; cx < cols; cx++) {
						int coordinateC = (cy * cols) + cx;
						// check all other than the current value
						if (x == cx && y == cy) {
							continue;
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
							sum[coordinateC] = currentValue
									+ (Math.pow(dotProduct, 2) * (weightValue / Constants.weightDivisor));
						} else {
							sum[coordinateC] = currentValue + Math.pow(dotProduct, 2);
						}
					}
				}
			}
		}

		Mat out = Helpers.arrayToMatrix(sum, rows, cols, CV_32F);
		Point maxLocation = new Point();
		DoublePointer maxValue = new DoublePointer();
		minMaxLoc(out, (DoublePointer) null, maxValue, (Point) null, maxLocation, (Mat) null);

		if (Constants.enablePostProcessing) {
			Mat floodClone = new Mat(rows, cols, CV_32F);
			double floodThresh = maxValue.get() * Constants.postProcessingTreshold;
			threshold(out, floodClone, floodThresh, 0.0, THRESH_TOZERO);
			Mat mask = floodKillEdges(floodClone);

			minMaxLoc(out, (DoublePointer) null, (DoublePointer) null, (Point) null, maxLocation, mask);
			return maxLocation;
		}

		eyeRegionSubframe.release();
		sizedImage.release();

		return maxLocation;
	}

	private static Mat floodKillEdges(Mat matrix) {
		Mat mask = new Mat(matrix.rows(), matrix.cols(), CV_8U, new Scalar(255, 255, 255, 255));
		Queue<Point> todo = new LinkedList<>();
		todo.add(new Point(0, 0));

		while (todo.size() > 0) {
			Point currentPoint = todo.peek();
			todo.poll();
			try (DoubleIndexer matrixIndexer = matrix.createIndexer()) {
				if (matrixIndexer.get((int) currentPoint.y(), (int) currentPoint.x(), 0) == 0.0) {
					continue;
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
			
			// add in every direction
			Point nextPoint = new Point(currentPoint.x() + 1, currentPoint.y()); // right
			if (Helpers.isPointInMatrix(nextPoint, matrix.rows(), matrix.cols())) {
				todo.add(nextPoint);
			}

			nextPoint.x(currentPoint.x() - 1);
			nextPoint.y(currentPoint.y()); // left
			if (Helpers.isPointInMatrix(nextPoint, matrix.rows(), matrix.cols())) {
				todo.add(nextPoint);
			}

			nextPoint.x(currentPoint.x());
			nextPoint.y(currentPoint.y() + 1); // down
			if (Helpers.isPointInMatrix(nextPoint, matrix.rows(), matrix.cols())) {
				todo.add(nextPoint);
			}

			nextPoint.x(currentPoint.x());
			nextPoint.y(currentPoint.y() - 1); // up
			if (Helpers.isPointInMatrix(nextPoint, matrix.rows(), matrix.cols())) {
				todo.add(nextPoint);
			}
			// kill it
			Indexer matrixIndexer = matrix.createIndexer();
			matrixIndexer.putDouble(new long[] { currentPoint.y(), currentPoint.x() }, 0.0);
			matrixIndexer.release();

			Indexer maskIndexer = mask.createIndexer();
			maskIndexer.putDouble(new long[] { currentPoint.y(), currentPoint.x() }, 0.0);
			maskIndexer.release();
		}
		return mask;
	}
}
