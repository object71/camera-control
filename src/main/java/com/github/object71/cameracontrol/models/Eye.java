/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol.models;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;

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
        throw new UnsupportedOperationException();
    }
}
