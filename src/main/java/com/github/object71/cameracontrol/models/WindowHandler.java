//package com.github.object71.cameracontrol.models;
//
//import java.awt.Color;
//import java.awt.Dimension;
//import java.awt.Graphics;
//import java.awt.Graphics2D;
//import java.awt.Rectangle;
//import java.awt.Shape;
//import java.awt.Toolkit;
//import java.awt.Window;
//import java.awt.event.KeyEvent;
//import java.awt.event.KeyListener;
//import java.awt.geom.Ellipse2D;
//import java.util.LinkedList;
//import java.util.Queue;
//
//import org.apache.commons.lang.time.StopWatch;
//import org.opencv.core.Rect;
//import org.opencv.core.Point;
//
//public class WindowHandler extends Window implements KeyListener, Runnable {
//
//    private enum State {
//        Stopped, Running, Callibrating
//    }
//
//    private class Drawable {
//
//        public Shape shape;
//        public Color color;
//
//        public Drawable(Color color, Shape shape) {
//            this.shape = shape;
//            this.color = color;
//        }
//    }
//
//    private static final long serialVersionUID = -193897823101696480L;
//    private FaceHandler face;
//    private State currentState;
//    private StopWatch calibrationTimer = new StopWatch();
//    private Queue<Drawable> drawables = new LinkedList<Drawable>();
//    public Thread executingThread;
//
//    private int[] screenBoundingBox = new int[4];
//
//    @Override
//    public void paint(Graphics g) {
//        Graphics2D g2 = (Graphics2D) g;
//        Color transparent = new Color(0, 0, 0, 0);
//
//        g2.clearRect(0, 0, this.getWidth(), this.getHeight());
//        g2.setBackground(transparent);
//
//        while (!drawables.isEmpty()) {
//            Drawable drawable = drawables.poll();
//
//            g2.setColor(drawable.color);
//            g2.fill(drawable.shape);
//        }
//
//    }
//
//    @Override
//    public void update(Graphics g) {
//
//        if (currentState == State.Running) {
//            // drawables.add(new Drawable(Color.white, new Rectangle(this.getSize())));
//
//            Point coord = this.face.eyeGazeCoordinate.getAveragePoint();
//            if (coord == null) {
//                return;
//            }
//
//            double kX = face.coordinateSystemSide / this.getWidth();
//            double kY = face.coordinateSystemSide / this.getHeight();
//            int x = (int) ((coord.x) / kX);
//            int y = (int) ((coord.y) / kY);
//
//            Shape circle = new Ellipse2D.Double(x - 32, y - 32, 64, 64);
//            drawables.add(new Drawable(Color.red, circle));
//
//        } else if (currentState == State.Callibrating) {
//
//            // drawables.add(new Drawable(Color.white, new Rectangle(this.getSize())));
//            long time = calibrationTimer.getTime();
//
//            if (time > 0 && time < 5000) {
//                drawables.add(new Drawable(Color.blue, new Ellipse2D.Double(-32, -32, 64, 64)));
//                if (time > 4000) {
//                    Point coord = this.face.eyeGazeCoordinate.getAveragePoint();
//                    screenBoundingBox[0] = (int) coord.y;
//                    screenBoundingBox[1] = (int) coord.x;
//                }
//            } else if (time > 5000 && time < 10000) {
//                drawables.add(new Drawable(Color.blue,
//                        new Ellipse2D.Double(this.getWidth() - 32, this.getHeight() - 32, 64, 64)));
//                if (time > 9000) {
//                    Point coord = this.face.eyeGazeCoordinate.getAveragePoint();
//                    screenBoundingBox[2] = (int) coord.y;
//                    screenBoundingBox[3] = (int) coord.x;
//                }
//            } else {
//                this.currentState = State.Running;
//            }
//        }
//
//        paint(g);
//    }
//
//    public WindowHandler(FaceHandler face) {
//        super(null);
//
//        this.face = face;
//        Color transparent = new Color(0, 0, 0, 0);
//        Dimension d = Toolkit.getDefaultToolkit().getScreenSize();
//        this.setAlwaysOnTop(true);
//        this.setSize(d);
//        this.setBounds(new Rectangle(d));
//        this.setVisible(true);
//        this.setBackground(transparent);
//        this.addKeyListener(this);
//
//        this.executingThread = new Thread(this);
//        this.executingThread.start();
//
//    }
//
//    @Override
//    public void keyPressed(KeyEvent arg0) {
//        // TODO Auto-generated method stub
//
//    }
//
//    @Override
//    public void keyReleased(KeyEvent arg0) {
//        if (arg0.getKeyCode() == KeyEvent.VK_ESCAPE) {
//            this.setVisible(false);
//        }
//
//    }
//
//    @Override
//    public void keyTyped(KeyEvent arg0) {
//        // TODO Auto-generated method stub
//
//    }
//
//    @Override
//    public void run() {
//        StopWatch watch = new StopWatch();
//        watch.start();
////		calibrationTimer.start();
//        this.currentState = State.Running;
//        while (this.currentState != State.Stopped) {
//            if (watch.getTime() > 100) {
//                this.update(this.getGraphics());
//
//                watch.stop();
//                watch.reset();
//                watch.start();
//            } else {
//                try {
//                    Thread.sleep(15);
//                } catch (InterruptedException e) {
//                    e.printStackTrace();
//                }
//            }
//        }
//    }
//
//}
