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
	_faceRec->_cameraState = true; //����ͷ������־
	_faceRec->_recState = true; //��⿪ʼ��־
	_faceRec->_uiShowLabel = ui.label; //����QLabel��ַ
	//double fps = _faceRec->_camera.get(cv::CAP_PROP_FPS);
	//double delay = 1000 / fps;
	_timer->start(30);//30ms��ȡһ֡, ����Ǳ��bug,�����Ҫ������ͷfps
}
