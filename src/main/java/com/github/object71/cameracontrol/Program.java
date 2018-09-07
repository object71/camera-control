/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol;

import com.github.object71.cameracontrol.common.MarkPoint;
import com.github.object71.cameracontrol.models.FaceHandler;

import org.apache.commons.lang.time.StopWatch;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_java;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

/**
 *
 * @author hristo
 */
public class Program {
	
	private static StopWatch stopwatch = new StopWatch();

    public static void main(String... args) {
    	stopwatch.start();
    	
        Loader.load(opencv_java.class);
        int cameraDevice = 0;
        FaceHandler face = new FaceHandler();
        VideoCapture capture = new VideoCapture(cameraDevice);
        
        if (!capture.isOpened()) {
            System.err.println("--(!)Error opening video capture");
            System.exit(0);
        }

        Mat frame = new Mat();
        
        stopwatch.stop();
        System.out.printf("Initialized in %dmS\n", stopwatch.getTime());
        
        while (capture.read(frame)) {
        	stopwatch.reset();
        	stopwatch.start();
        	
            if (frame.empty()) {
                System.err.println("--(!) No captured frame -- Break!");
                break;
            }
            face.initializeFrameInformation(frame);

            if(face.leftEye.getAveragePoint() != null) {
                Imgproc.circle(frame, face.getFaceMark(MarkPoint.LeftEyeBioCenter), 1, new Scalar(255, 0, 0, 255));
                Imgproc.circle(frame, face.leftEye.getAveragePoint(), 1, new Scalar(255, 0, 255, 0));
            }
            if(face.rightEye.getAveragePoint() != null) {
                Imgproc.circle(frame, face.getFaceMark(MarkPoint.RightEyeBioCenter), 1, new Scalar(255, 0, 0, 255));
                Imgproc.circle(frame, face.rightEye.getAveragePoint(), 1, new Scalar(255, 0, 255, 0));
            }

            HighGui.imshow("Debug", frame);
            
            if (HighGui.waitKey(10) == 27) {
                break;// escape
            }
            
            stopwatch.stop();
            System.out.printf("Frame process time is %dmS\n", stopwatch.getTime());
        }
        
        capture.release();
        System.exit(0);
    }
}
