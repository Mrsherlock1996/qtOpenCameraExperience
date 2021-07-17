#include "MainWindow.h"

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	_timer = new QTimer();
	_faceRec = new FaceRec();

	connect(_timer, &QTimer::timeout, _faceRec, &FaceRec::begainToCommonFaceRec);
}
void MainWindow::on_pushButton_clicked()
{
	_faceRec->_camera = cv::VideoCapture(0);
	_faceRec->_cameraState = true; //摄像头开启标志
	_faceRec->_recState = true; //检测开始标志
	_faceRec->_uiShowLabel = ui.label; //传递QLabel地址
	//double fps = _faceRec->_camera.get(cv::CAP_PROP_FPS);
	//double delay = 1000 / fps;
	_timer->start(30);//30ms读取一帧, 这里潜藏bug,最好需要先摄像头fps
}
