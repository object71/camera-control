/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol.common;

import org.opencv.core.Rect;

/**
 *
 * @author hristo
 */
public class Constants {
    public static final boolean plotVectorField = false;
    public static final double eyeMultiplierTop = 0.25;
    public static final double eyeMultiplierLeft = 0.13;
    public static final double eyeMultiplierWidth = 0.30;
    public static final double eyeMultiplierHeight = 0.35;
    
    public static final boolean smoothFaceImage = false;
    public static final double smoothFaceFactor = 0.005;
    
    public static final int fastEyeWidth = 50;
    public static final int weightBlurSize = 5;
    public static final boolean enableWeight = true;
    public static final double weightDivisor = 1.0;
    public static final double gradientTreshold = 50.0;
    
    public static final boolean enablePostProcessing = true;
    public static final double postProcessingTreshold = 0.97;
}
