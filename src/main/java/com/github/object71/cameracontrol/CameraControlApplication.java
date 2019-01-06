/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.object71.cameracontrol;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_java;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.event.EventHandler;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;

/**
 *
 * @author hristo
 */
public class CameraControlApplication extends Application {

	public static void main(String... args) {
		Loader.load(opencv_java.class);
		launch(args);
	}

	@Override
	public void start(Stage primaryStage) throws Exception {
//		Parent root = FXMLLoader.load(getClass().getClassLoader().getResource("fxml/CameraControl.fxml"));
		FXMLLoader loader = new FXMLLoader(getClass().getClassLoader().getResource("fxml/CameraControl.fxml"));
		Parent root = (Parent)loader.load();
		Scene scene = new Scene(root, 640, 480);
		
		primaryStage.setTitle("Camera control");
        primaryStage.setScene(scene);
        primaryStage.setMinWidth(640);
        primaryStage.setMinHeight(480);
        primaryStage.show();
        primaryStage.setOnCloseRequest(new EventHandler<WindowEvent>() {
            public void handle(WindowEvent we) {
                System.out.println("Stage is closing");
                CameraControlController controller = (CameraControlController)loader.getController();
                controller.closeWindow();
                Platform.exit();
                System.exit(0);
            }
        });
		
	}
}
