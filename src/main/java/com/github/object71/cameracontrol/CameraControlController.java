package com.github.object71.cameracontrol;

import java.awt.image.BufferedImage;
import java.net.URL;
import java.util.ResourceBundle;

import org.bytedeco.javacpp.opencv_core.Point;

import com.github.object71.cameracontrol.common.ImageProcessedListener;
import com.github.object71.cameracontrol.services.FaceService;

import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Slider;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.ToggleButton;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;

public class CameraControlController implements Initializable, ImageProcessedListener {

    private FaceService face;
    private boolean calibratedOnce = false;

    @Override
    public void initialize(URL location, ResourceBundle resources) {

        brightnessSlider.valueProperty().addListener((observable, oldValue, newValue) -> {
            face.brightnessValue = (double) newValue;
        });
        face.startThread();
        mouseMoveButton.setDisable(true);
        mouseCalibration.setDisable(true);
    }

    public CameraControlController() {
        face = new FaceService();
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
    private CheckBox enableRegionsCheckbox;

    @FXML
    private TextField regionsWidth;

    @FXML
    private TextField regionsHeight;

    @FXML
    private TextField cameraWidth;

    @FXML
    private TextField cameraHeight;

    @FXML
    private TextField eyeRegionWidth;

    @FXML
    private TextField eyeRegionHeight;

    @FXML
    void onAnyFieldChange(ActionEvent event) {
        face.isAnchorEnabled = enableRegionsCheckbox.isSelected();

        try {
            face.coordinateSystemWidth = Integer.parseInt(eyeRegionWidth.getText());
        } catch (NumberFormatException e) {
            eyeRegionWidth.setText("" + face.coordinateSystemWidth);
        }
        
        try {
            face.coordinateSystemHeight = Integer.parseInt(eyeRegionHeight.getText());
        } catch (NumberFormatException e) {
            eyeRegionHeight.setText("" + face.coordinateSystemHeight);
        }
        
        try {
            face.cameraWidth = Integer.parseInt(cameraWidth.getText());
        } catch (NumberFormatException e) {
            cameraWidth.setText("" + face.cameraWidth);
        }
        
        try {
            face.cameraHeight = Integer.parseInt(cameraHeight.getText());
        } catch (NumberFormatException e) {
            cameraHeight.setText("" + face.cameraHeight);
        }
        
        try {
            face.anchorWidth = Integer.parseInt(regionsWidth.getText());
        } catch (NumberFormatException e) {
            regionsWidth.setText("" + face.anchorWidth);
        }
        
        try {
            face.anchorHeight = Integer.parseInt(regionsHeight.getText());
        } catch (NumberFormatException e) {
            regionsHeight.setText("" + face.anchorHeight);
        }
    }

    @FXML
    void calibrateOnMouseClick(ActionEvent event) {
        if (mouseCalibration.isSelected()) {
            if(!calibratedOnce) {
                calibratedOnce = true;
                mouseMoveButton.setDisable(false);
            }
            face.calibrateOnMouseClick = true;
            mouseMoveButton.setSelected(false);
            face.controlMouse = false;
        } else {
            face.calibrateOnMouseClick = false;
        }
    }

    @FXML
    void controlMouse(ActionEvent event) {
        if (mouseMoveButton.isSelected()) {
            face.controlMouse = true;
            mouseCalibration.setSelected(false);
            face.calibrateOnMouseClick = false;
        } else {
            face.controlMouse = false;
        }
    }

    @FXML
    void startProcess(ActionEvent event) {
        if (processButton.isSelected()) {
            face.process = true;
            mouseMoveButton.setDisable(!calibratedOnce);
            mouseCalibration.setDisable(false);
        } else {
            face.process = false;
            mouseMoveButton.setDisable(true);
            mouseCalibration.setDisable(true);
        }
    }

    @Override
    public void onImageProcessed(java.awt.Image image) {

        this.centerImage.setImage(SwingFXUtils.toFXImage((BufferedImage) image, null));
        image.getGraphics().dispose();
    }
}
