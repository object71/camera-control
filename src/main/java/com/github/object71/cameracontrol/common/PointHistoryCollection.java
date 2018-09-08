/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol.common;

import org.opencv.core.Point;

/**
 *
 * @author hristo
 */
public class PointHistoryCollection {
    private Point[] points;
    private int current = 0;
    private int size;
    
    public PointHistoryCollection(int size) {
        points = new Point[size];
        this.size = size;
    }
    
    public Point getAveragePoint() {
        Point result = null;
        for(Point point : points) {
            if(point == null) {
                continue;
            }
            
            if(result == null) {
                result = new Point(point.x, point.y);
            } else {
                result.x = (result.x + point.x) / 2;
                result.y = (result.y + point.y) / 2;
            }
        }
        
        return result;
    }
    
    public void insertNewPoint(Point point) {
        points[current] = point;
        
        current++;
        if(current == size) {
            current = 0;
        }
    }
}
