package com.github.object71.cameracontrol.models;

import org.bytedeco.javacpp.opencv_core.Point;

import com.github.object71.cameracontrol.common.Helpers;

public class EyeModel {
	public Point leftCorner;
	public Point topCorner;
	public Point rightCorner;
	public Point bottomCorner;
	
	public double getCornersDistance() {
		return Helpers.distanceBetweenPoints(this.leftCorner, this.rightCorner);
	}
	
	public double getLidsDistance() {
		return Helpers.distanceBetweenPoints(topCorner, bottomCorner);
	}
	
	public boolean getIsBlinking() {
		return this.getLidsDistance() / (2 * this.getCornersDistance()) < 0.15;
	}
}
