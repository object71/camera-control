/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol;

import com.github.object71.cameracontrol.models.Face;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_java;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.videoio.VideoCapture;

/**
 *
 * @author hristo
 */
public class Program {

    public static void main(String... args) {
        Loader.load(opencv_java.class);
        int cameraDevice = 0;
        Face face = new Face();
        VideoCapture capture = new VideoCapture(cameraDevice);
        // capture.set(Videoio.CAP_PROP_FRAME_WIDTH, 320);
        // capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 240);
        
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
            HighGui.imshow("Debug", face.getDebugFrame());
            
            if (HighGui.waitKey(10) == 27) {
                break;// escape
            }
        }
        System.exit(0);
    }
}
