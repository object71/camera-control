package com.github.object71.cameracontrol.models;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Shape;
import java.awt.Window;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.geom.Ellipse2D;

import org.apache.commons.lang.time.StopWatch;
import org.opencv.core.Point;

public class WindowHandler extends Window implements KeyListener, Runnable {
	private enum State {
		Stopped, Running, Callibrating
	}

	private static final long serialVersionUID = -193897823101696480L;
	private FaceHandler face;
	private State currentState;
	public Thread executingThread;

	@Override
	public void paint(Graphics g) {
		Graphics2D g2 = (Graphics2D) g;
		Point coord = this.face.eyeGazeCoordinate.getAveragePoint();
		if (coord == null) {
			return;
		}

		Dimension dimension = this.getSize();
		double kX = this.face.coordinateSystemSide / dimension.getWidth() ;
		double kY = this.face.coordinateSystemSide / dimension.getHeight() ;
		int x = (int) (coord.x / kX);
		int y = (int) (coord.y / kY);

		Shape circle = new Ellipse2D.Double(x - 32, y - 32, 64, 64);
		g2.setColor(Color.red);
		g2.draw(circle);
	}

	@Override
	public void update(Graphics g) {

		if (currentState == State.Running) {
			paint(g);
		}
	}

	public WindowHandler(FaceHandler face) {
		super(null);

		this.face = face;
		Color transparent = new Color(0, 0, 0, 0);

		this.setAlwaysOnTop(false);
		this.setBounds(this.getGraphicsConfiguration().getBounds());
		this.setVisible(true);
		this.setBackground(transparent);
		this.addKeyListener(this);

		this.currentState = State.Running;
		
		this.executingThread = new Thread(this);
		this.executingThread.start();
	}

	@Override
	public void keyPressed(KeyEvent arg0) {
		// TODO Auto-generated method stub

	}

	@Override
	public void keyReleased(KeyEvent arg0) {
		if (arg0.getKeyCode() == KeyEvent.VK_ESCAPE) {
			this.setVisible(false);
		}

	}

	@Override
	public void keyTyped(KeyEvent arg0) {
		// TODO Auto-generated method stub

	}

	@Override
	public void run() {
		StopWatch watch = new StopWatch();
		watch.start();
		while (this.currentState != State.Stopped) {
			if (watch.getTime() > 100) {
				this.update(this.getGraphics());
				
				watch.stop();
				watch.reset();
				watch.start();
			} else {
				try {
					Thread.sleep(10);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
	}

}
