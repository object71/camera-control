package com.github.object71.cameracontrol;

import java.awt.image.BufferedImage;
import java.net.URL;
import java.util.ResourceBundle;

import com.github.object71.cameracontrol.common.ImageProcessedListener;
import com.github.object71.cameracontrol.models.FaceHandler;

import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.ToggleButton;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;

public class CameraControlController implements Initializable, ImageProcessedListener {
	
	private FaceHandler face;

	@Override
	public void initialize(URL location, ResourceBundle resources) {
		face.startThread();
	}
	
	public CameraControlController() {
		face = new FaceHandler();
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
		this.centerImage.setImage(SwingFXUtils.toFXImage((BufferedImage) image, null));
		
	}
}
