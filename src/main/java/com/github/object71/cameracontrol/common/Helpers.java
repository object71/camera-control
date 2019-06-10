/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol.common;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Rect2d;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.Indexer;

import com.github.object71.cameracontrol.models.GradientsModel;

import static org.bytedeco.javacpp.opencv_core.*;

import java.awt.image.BufferedImage;

/**
 *
 * @author hristo
 */
public class Helpers {

	public static boolean isRectangleInMatrix(Rect rectangle, Mat matrix) {
		return rectangle.x() > 0 && rectangle.y() > 0 && rectangle.x() + rectangle.width() < matrix.cols()
				&& rectangle.y() + rectangle.height() < matrix.rows();
	}

	public static Pointer getPointer(long nativeAddress) {
		Pointer pointer = new Pointer() {
			{
				address = nativeAddress;
			}
		};
		return pointer;
	}

	public static Rect2d RectToRect2d(Rect rectangle) {
		return new Rect2d(rectangle.x(), rectangle.y(), rectangle.width(), rectangle.height());
	}

	public static Rect Rect2dToRect(Rect2d rectangle) {
		return new Rect((int) rectangle.x(), (int) rectangle.y(), (int) rectangle.width(), (int) rectangle.height());
	}

	public static boolean isPointInMatrix(Point point, Mat matrix) {
		return point.x() >= 0 && point.x() < matrix.cols() && point.y() >= 0 && point.y() < matrix.rows();
	}

	public static boolean isPointInMatrix(Point point, int rows, int cols) {
		return point.x() >= 0 && point.x() < cols && point.y() >= 0 && point.y() < rows;
	}
	
	public static boolean isPointInMatrix(int x, int y, int rows, int cols) {
		return x >= 0 && x < cols && y >= 0 && y < rows;
	}

	public static double computeDynamicTreshold(double[] matrix, double standardDeviationFactor, int rows, int cols) {
		Mat inputMatrix = new Mat(rows, cols, CV_64FC1);
		try (DoubleIndexer inputMatrixIndexer = inputMatrix.createIndexer()) {
			inputMatrixIndexer.put(0, 0, matrix);

			// calculating the standard deviation and mean deviation
			// standard deviation is the deviation peresented with negative values also
			// mean deviation (absolute deviation) is the deviation presented by only absolute values
			Mat standartMagnitudeGradient = new Mat();
			Mat meanMagnitudeGradient = new Mat();
			meanStdDev(inputMatrix, meanMagnitudeGradient, standartMagnitudeGradient);

			try (DoubleIndexer standartMagnitudeGradientIndexer = standartMagnitudeGradient.createIndexer();
					DoubleIndexer meanMagnitudeGradientIndexer = meanMagnitudeGradient.createIndexer()){

				double standartDeviation = standartMagnitudeGradientIndexer.get(0, 0, 0) / Math.sqrt(rows * cols);
				double result = standartDeviation * standardDeviationFactor + meanMagnitudeGradientIndexer.get(0, 0, 0);

				return result;
			} catch (Exception e) {
				e.printStackTrace();
			}
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		
		return 0;
	}

	public static Point centerOfRect(Rect rectangle) {
		return new Point((int) (rectangle.x() + (rectangle.width() / 2.0)),
				(int) (rectangle.y() + (rectangle.height() / 2.0)));
	}

	public static Point centerOfRect(Rect2d rectangle) {
		return new Point((int) (rectangle.x() + (rectangle.width() / 2.0)),
				(int) (rectangle.y() + (rectangle.height() / 2.0)));
	}

	public static double centerOfRectXAxis(Rect rectangle) {
		return rectangle.x() + (rectangle.width() / 2.0);
	}

	public static double distanceBetweenPoints(Point a, Point b) {
		return Math.sqrt(Math.pow(a.x() - b.x(), 2) + Math.pow(a.y() - b.y(), 2));
	}

	public static Point centerOfPoints(Point a, Point b) {
		if (a == null) {
			return b;
		} else if (b == null) {
			return b;
		} else {
			return new Point((a.x() + b.x()) / 2, (a.y() + b.y()) / 2);
		}
	}

	public static Point centerOfPoints(double faceMarks, double faceMarks2, double faceMarks3, double faceMarks4) {
		return new Point((int) ((faceMarks + faceMarks3) / 2), (int) ((faceMarks2 + faceMarks4) / 2));
	}

	public static double distanceBetweenValues(double a, double b) {
		return a >= b ? a - b : b - a;
	}

	public static GradientsModel computeGradient(double[] input, int rows, int cols) {
		if (cols < 3) {
			throw new IllegalArgumentException("Invalid matrix size");
		}
		double[] gradientX = new double[rows * cols];
		double[] gradientY = new double[rows * cols];
		double[] magnitude = new double[rows * cols];
		double xA, xB, yA, yB, valueX, valueY;

		// compute the gradients for each pixel and also the gradient magnitude
		for (int y = 0; y < rows; y++) {
			for (int x = 1; x < cols; x++) {
				int coordinate = (y * cols) + x;

				xA = x == cols - 1 ? input[coordinate] : input[coordinate + 1];
				xB = x == 0 ? input[coordinate] : input[coordinate - 1];
				gradientX[coordinate] = valueX = xA - xB;

				yA = y == rows - 1 ? input[coordinate] : input[coordinate + cols];
				yB = y == 0 ? input[(y) + (x)] : input[coordinate - cols];
				gradientY[coordinate] = valueY = yA - yB;

				magnitude[coordinate] = Math.sqrt((valueX * valueX) + (valueY * valueY));
			}
		}

		// could be a static treshold but here its used 
		// the standard deviation and only values that are above
		// will be processed further
		double gradientTreshold = Helpers.computeDynamicTreshold(magnitude, Constants.gradientTreshold, rows, cols);

		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < cols; x++) {
				int coordinate = (y * cols) + x;
				valueX = gradientX[coordinate];
				valueY = gradientY[coordinate];
				double magnitudeValue = magnitude[coordinate];
				if (magnitudeValue > gradientTreshold) {
					// recalculate gradients based on the magnitude matrix
					gradientX[coordinate] = valueX / magnitudeValue;
					gradientY[coordinate] = valueY / magnitudeValue;
				} else {
					// zero out values below the treshold
					gradientX[coordinate] = 0;
					gradientY[coordinate] = 0;
				}
			}
		}
		GradientsModel gradients = new GradientsModel();
		gradients.gradientX = gradientX;
		gradients.gradientY = gradientY;
		gradients.magnitude = magnitude;
		gradients.gradientTreshold = gradientTreshold;
		gradients.cols = cols;
		gradients.rows = rows;

		return gradients;
	}

	public static double[] getMatrixMagnitude(double[] matrixX, double[] matrixY, int rows, int cols) {
		double[] magnitudeMatrix = new double[rows * cols];
		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < cols; x++) {
				int coordinate = (y * cols) + x;
				double valueX = matrixX[coordinate];
				double valueY = matrixY[coordinate];
				double magnitudeValue = Math.sqrt((valueX * valueX) + (valueY * valueY));
				magnitudeMatrix[coordinate] = magnitudeValue;
			}
		}
		return magnitudeMatrix;
	}

	public static double[] matrixToArray(Mat matrix) {
		int rows = matrix.rows();
		int cols = matrix.cols();
		double[] result = new double[rows * cols];
		try (Indexer matrixIndexer = matrix.createIndexer()) {
			for (int y = 0; y < rows; y++) {
				for (int x = 0; x < cols; x++) {
					result[y * cols + x] = matrixIndexer.getDouble(y, x, 0);

				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		return result;
	}

	public static Mat arrayToMatrix(double[] matrix, int rows, int cols) {
		return arrayToMatrix(matrix, rows, cols, CV_64F);
	}

	public static Mat arrayToMatrix(double[] matrix, int rows, int cols, int type) {
		Mat result = new Mat(rows, cols, type);
		Indexer resultIndexer = result.createIndexer();

		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < cols; x++) {
				resultIndexer.putDouble(new long[] { y, x }, matrix[y * cols + x]);
			}
		}

		return result;
	}
	
	public static BufferedImage toBufferedImage(Mat matrix) {
		int width = matrix.cols();
		int height = matrix.rows();
		
		BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
		
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				Indexer matrixIndexer = matrix.createIndexer();
				
				int a = 255;
				int r = (int)matrixIndexer.getDouble((long) y, (long) x, (long) 2);
				int g = (int)matrixIndexer.getDouble((long) y, (long) x, (long) 1);
				int b = (int)matrixIndexer.getDouble((long) y, (long) x, (long) 0);
				
				int p = (a<<24) | (r<<16) | (g<<8) | b;
				result.setRGB(x, y, p);
			}
		}
		
		return result;
	}

	public static void bright(Mat matrix, double brightenBy) {
//		for (int y = 0; y < matrix.rows(); y++) {
//			for (int x = 0; x < matrix.cols(); x++) {
//				Indexer matrixIndexer = matrix.createIndexer();
//
//				for (int v = 0; v < matrixIndexer.channels(); v++) {
//					double value = matrixIndexer.getDouble((long) y, (long) x, (long) v) + brightenBy;
//					matrixIndexer.putDouble(new long[] { (long) y, (long) x, (long) v }, value > 255 ? 255 : value);
//				}
//			}
//		}
	}

	public static void possibleCenterFormula(int x, int y, Mat weight, double gx, double gy, Mat output) {

		for (int cy = 0; cy < output.rows(); cy++) {
			for (int cx = 0; cx < output.cols(); cx++) {
				if (x == cx && y == cy) {
					continue;
				}

				double dx = x - cx;
				double dy = y - cy;
				double magnitude = Math.sqrt(Math.pow(dx, 2) + Math.pow(dy, 2));
				dx = dx / magnitude;
				dy = dy / magnitude;

				double dotProduct = Math.max(0, dx * gx + dy * gy);

				try (Indexer outputIndexer = output.createIndexer()) {

					double currentValue = outputIndexer.getDouble(cy, cx, 0);

					if (Constants.enableWeight) {
						try (Indexer weightIndexer = output.createIndexer()) {
							double weightValue = weightIndexer.getDouble(cy, cx, 0);
							outputIndexer.putDouble(new long[] { cy, cx },
									currentValue + (Math.pow(dotProduct, 2) * (weightValue / Constants.weightDivisor)));
						} catch (Exception e) {
							e.printStackTrace();
						}
					} else {
						outputIndexer.putDouble(new long[] { cy, cx }, currentValue + Math.pow(dotProduct, 2));
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
	}
}
