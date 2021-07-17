//
//FaceRec��ʵ�ֵĹ���:
//��⹦��:
//	ʶ����������,��ע���
//	ʶ���������,��ע�̿�
//	ʶ������ʾ�ڵ��õ�ui->QLabel��
//
//ѵ������:
//	���ڵ���FaceRecognizer��������ѵ��ģ����Ҫ��ʱ�ܾ�,���Խ�ѵ���ͼ���Ϊ�����ӿ�,
//	ʹ��haarcascadeclassifierѵ��ͬ����ʱ�Ͼ�,���:
//		ѵ��������xml�ļ�,������ʽ"ָ����·��+ģ����������.XML"
//		���ʱֱ�Ӽ��ظ�����xml�ļ�,���м��
//
//
//
//����ʶ������ַ���
//����1:
//		ͨ��CascadeClassifier��⵽����,�ټ���ѵ���õİ���"���"��Ϣ��xml, ��CascadeClassifier��ѵ����Ϣ, �����ڶ���Cascade����, �Ը�����ʶ��ȷ����Ӧ���
//����2:
//		��ʹ��cv::face::FaceReconizer��������,��"���"��������ѵ��, ������cascade classifier�������, ʹ��FaceRecognizer�������Ԥ��
//
//cv::faceģ��
//cv::face::EigenFaceRecognizer and cv::face::FisherFaceRecognizer���̳���cv::face::BasicFaceRecognizer,
//	���߶�����һ��premise assumption :
//		    �����ͼƬ����ѵ��ʱ��ͬ�ߴ��equal size(�ݶ�800,800),�����ǻҶ�ͼ,
//	���߶������Ը��¶���
//cv::Algorithm::FaceRecognizer::BasicFaceRecognizer
//����BasicFaceRecognizer����������ͼƬ��û��������Ҫ��,ֻ��Ҫ������һ��image��label��Ӧ����
//	BasicFaceRecognizer���Ը��¶���
//
//txt�ļ��ĸ�ʽ:
//	ͼƬ1·��
//	��ǩ
//	ͼƬ2·��
//	��ǩ
//
//
#pragma once

#include <QObject>
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
#include <qlabel.h>
#include "ConvertMatQImage.h"
#include <iostream>
#include <vector>
class FaceRec : public QObject
{
	Q_OBJECT

public:
	FaceRec(QObject *parent);
	FaceRec();
	~FaceRec();
	std::string _haarFaceDataPath; //opencv������ѵ���õ�haarcascade��xml�ļ�·��,���ڴ�������ʶ��
	std::string _trainSetTxtFilePath; //ѵ������Ϣ�ļ�·��"C:/a.txt"

	
	std::vector<cv::Mat> _images;//�洢train set
	std::vector<int> _labels;//�洢train set labels
	//ֻ��Ҫ��ǩ��ͼƬ��Ϣ�Ƕ�Ӧ�ļ�����ɶ������ʶ��

	cv::Size _size; //���ڼ�¼EigenFaceRecʱ����Ҫ�ĳߴ�,��ʼ��Ĭ��Ϊ800,800,��֤train��predictʱsizeһ��

	cv::VideoCapture _camera;
	bool _cameraState = false; //����ͷ״̬
	bool _trainState = false;//ѵ����ʼ��־����
	bool _recState=false;//�Ƿ�ʼʶ��ı�־, trueʱʶ��, false������ʶ��
	bool _trainResult = false;//ѵ�����
	ConvertMatQImage _cvt; //����Mat��QImageת��

	cv::Mat _frame; //֡
	std::vector<cv::Rect> _faces; //�������������
	cv::Mat _recFace; //ʶ�������

	QLabel* _uiShowLabel; //�洢��ʾ��ui�����ָ��

	void begainToRec(QString modelXmlAbsPath,QLabel* uiLabel );//ֱ�ӽ������ʾ��ui��

	void startTrain(QString trainSetTxtFilePath, QString resultXMLFilePath);//ѵ��ģ��,����xml

	void begainToCommonFaceRec();
};
