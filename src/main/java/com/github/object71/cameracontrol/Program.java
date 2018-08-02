/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol;

import com.github.object71.cameracontrol.models.FaceEyeHandler;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_java;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

/**
 *
 * @author hristo
 */
public class Program {

    public static void main(String... args) {
        Loader.load(opencv_java.class);
        int cameraDevice = 0;
        FaceEyeHandler face = new FaceEyeHandler();
        VideoCapture capture = new VideoCapture(cameraDevice);
        //capture.set(Videoio.CAP_PROP_FRAME_WIDTH, 320);
        //capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 240);
        
        if (!capture.isOpened()) {
            System.err.println("--(!)Error opening video capture");
            System.exit(0);
        }

        Mat frame = new Mat();
        while (capture.read(frame)) {
            if (frame.empty()) {
                System.err.println("--(!) No captured frame -- Break!");
                break;
            }
            face.initializeFrameInformation(frame);

            if(face.leftEye.getAveragePoint() != null) {
                Imgproc.circle(frame, face.leftEye.getAveragePoint(), 1, new Scalar(255, 255, 255, 255));
            }
            if(face.rightEye.getAveragePoint() != null) {
                Imgproc.circle(frame, face.rightEye.getAveragePoint(), 1, new Scalar(255, 255, 255, 255));
            }

            HighGui.imshow("Debug", frame);
            
            if (HighGui.waitKey(10) == 27) {
                break;// escape
            }
        }
        System.exit(0);
    }
}
