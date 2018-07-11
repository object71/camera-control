/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol.models;

import com.github.object71.cameracontrol.common.Constants;
import com.github.object71.cameracontrol.common.Helpers;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author hristo
 */
public class Eye {
    private Face face;

    public Eye(Face face) {
        this.face = face;
    }

    protected Point getEyeCenter(Mat faceSubframe, Rect eyeRegion) {
        Mat eyeRegionSubframeUnscaled = faceSubframe.submat(eyeRegion);
        Size scaleTo = new Size(Constants.fastEyeWidth, Constants.fastEyeWidth);
                //(Constants.fastEyeWidth / eyeRegionSubframeUnscaled.cols()) * eyeRegionSubframeUnscaled.rows());
        Mat eyeRegionSubframe = new Mat((int)Math.round(scaleTo.width), (int)Math.round(scaleTo.height), CvType.CV_64F);

        Imgproc.resize(eyeRegionSubframeUnscaled, eyeRegionSubframe, scaleTo);

        // y is calculated with the same func - matrix is rotated
        Mat gradientXMatrix = Helpers.computeMatXGradient(eyeRegionSubframe);
        Mat gradientYMatrix = Helpers.computeMatXGradient(eyeRegionSubframe.t()).t();
        Mat magnitudeMatrix = Helpers.getMatrixMagnitude(gradientXMatrix, gradientYMatrix);

        double gradientTreshold = Helpers.computeDynamicTreshold(magnitudeMatrix, Constants.gradientTreshold);

        for(int y = 0; y < eyeRegionSubframe.rows(); y++) {
            for(int x = 0; x < eyeRegionSubframe.cols(); x++) {
                double valueX = gradientXMatrix.get(y, x)[0];
                double valueY = gradientYMatrix.get(y, x)[0];
                double magnitude = magnitudeMatrix.get(y, x)[0];
                if(magnitude > gradientTreshold) {
                    gradientXMatrix.put(y, x, valueX / magnitude);
                    gradientYMatrix.put(y, x, valueY / magnitude);
                } else {
                    gradientXMatrix.put(y, x, 0);
                    gradientYMatrix.put(y, x, 0);
                }
            }
        }

        Mat weight = new Mat(eyeRegionSubframe.rows(), eyeRegionSubframe.cols(), CvType.CV_64F);
        Imgproc.GaussianBlur(eyeRegionSubframe, weight, new Size(Constants.weightBlurSize, Constants.weightBlurSize), 0, 0);
        for(int y = 0; y < weight.rows(); y++) {
            for(int x = 0; x < weight.cols(); x++) {
                weight.put(y, x, 255 - weight.get(y, x)[0]);
            }
        }

        Mat sum = new Mat(eyeRegionSubframe.rows(), eyeRegionSubframe.cols(), CvType.CV_64F);
        for(int y = 0; y < weight.rows(); y++) {
            for(int x = 0; x < weight.cols(); x++) {
                double valueX = gradientXMatrix.get(y, x)[0];
                double valueY = gradientYMatrix.get(y, x)[0];
                if (valueX == 0.0 && valueY == 0.0) {
                    continue;
                }

                Helpers.possibleCenterFormula(x, y, weight, valueX, valueY, sum);
            }
        }

        double numGradients = (weight.rows() * weight.cols());
        Mat out = new Mat(sum.rows(), sum.cols(), CvType.CV_32F);
        sum.convertTo(out, CvType.CV_32F, 1.0 / numGradients);

        MinMaxLocResult result = Core.minMaxLoc(out, null);
        
        // if(kEnablePostProcess) {
        //     Mat floodClone;
        //     double floodThresh = maxVal * Constants.postProcessingTreshold;
        //     threshold(out, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO);
        //     if(kPlotVectorField) {
        //       //plotVecField(gradientX, gradientY, floodClone);
        //       imwrite("eyeFrame.png",eyeROIUnscaled);
        //     }
        //     cv::Mat mask = floodKillEdges(floodClone);
        //     //imshow(debugWindow + " Mask",mask);
        //     //imshow(debugWindow,out);
        //     // redo max
        //     cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);
        // }

        return unscalePoint(result.maxLoc, eyeRegion);
    }

    private static Point unscalePoint(Point point, Rect originalSize) {
        double ratio = Constants.fastEyeWidth / originalSize.width;
        int x = (int)Math.round(point.x / ratio);
        int y = (int)Math.round(point.y / ratio);
        return new Point(x,y);
    }
}
