#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_MainWindow.h"
#include "ConvertMatQImage.h"
#include "FaceRec.h"
#include <opencv2/opencv.hpp>
#include <qtimer.h>
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = Q_NULLPTR);
	QTimer* _timer;
	FaceRec* _faceRec;
private slots:
	void on_pushButton_clicked();
private:
    Ui::MainWindowClass ui;

};
