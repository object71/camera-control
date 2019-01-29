package com.github.object71.cameracontrol;

import java.awt.image.BufferedImage;
import java.net.URL;
import java.util.ResourceBundle;

import org.bytedeco.javacpp.opencv_core.Point;

import com.github.object71.cameracontrol.common.ImageProcessedListener;
import com.github.object71.cameracontrol.models.FaceHandler;

import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Slider;
import javafx.scene.control.TextArea;
import javafx.scene.control.ToggleButton;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;

public class CameraControlController implements Initializable, ImageProcessedListener {
	
	private FaceHandler face;

	@Override
	public void initialize(URL location, ResourceBundle resources) {
		
		brightnessSlider.valueProperty().addListener((observable, oldValue, newValue) -> {
			face.brightnessValue = (double) newValue;
		});
		face.startThread();
		
	}
	
	public CameraControlController() {
		face = new FaceHandler();
		face.registerImageProcessedListener(this);
	}
	
	public void closeWindow() {
		face.closing = true;
		try {
			face.currentThread.join();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	@FXML
    private ImageView centerImage;

    @FXML
    private ToggleButton mouseMoveButton;

    @FXML
    private ToggleButton mouseCalibration;

    @FXML
    private ToggleButton processButton;
    
    @FXML
    private Slider brightnessSlider;
    
    @FXML
    private TextArea logArea;
    
    @FXML
    void calibrateOnMouseClick(ActionEvent event) {
    	face.calibrateOnMouseClick = mouseCalibration.isSelected();
    }

    @FXML
    void controlMouse(ActionEvent event) {
    	face.controlMouse = mouseMoveButton.isSelected();
    }

    @FXML
    void startProcess(ActionEvent event) {
    	face.process = processButton.isSelected();
    }

	@Override
	public void onImageProcessed(java.awt.Image image) {
		String log = "";
		Point gaze = face.eyeGazeCoordinate.getAveragePoint();
		if(gaze != null) {
			log += "Cooridnate is (" +  gaze.x() + "," + gaze.y() + ") \n";
		}
		log += "\nleft: " + face.leftBound + "\nright: " + face.rightBound + "\ntop: " + face.topBound + "\nbottom: " + face.bottomBound + "\n";
		logArea.setText(log);
		
		this.centerImage.setImage(SwingFXUtils.toFXImage((BufferedImage) image, null));
		image.getGraphics().dispose();
	}
}
