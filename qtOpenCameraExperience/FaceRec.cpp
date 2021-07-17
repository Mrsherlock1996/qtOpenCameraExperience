#include "FaceRec.h"

FaceRec::FaceRec(QObject *parent)
	: QObject(parent)
{
	_size = cv::Size(800, 800); //��֤train��predictʱsizeһ��
}
FaceRec::FaceRec()
{
	_size = cv::Size(800, 800);
}
FaceRec::~FaceRec()
{
}

//��ʼ�����������,�����ʶ��
//ͨ�����ݵ�xml�ļ�·�������ʶ��, ��Ƶ��ʾ�ڴ��ݵ�QLabelָ����
//Ӧ��ʶ��ǰ��_cameraState��_recState
void FaceRec::begainToRec(QString modelXmlAbsPath, QLabel * uiLabel)
{
	using namespace std;
	using namespace cv;
	if (_recState == true) //Ӧ�ڵ��ú���ǰ����־����ʹ��
	{
		//�������ʹ�õ�opencvѵ���õ�cascade��������xml�ļ�:
		//			haarcascade_frontalface_alt_tree.xml �ѷŵ���Ŀ�ļ���
		string _haarFaceDataPath = "haarcascade_frontalface_alt_tree.xml";
		string xmlFilePath = modelXmlAbsPath.toStdString();


		//����cv::face::BasicFaceRecognizer����
		Ptr<cv::face::BasicFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
		model->read(xmlFilePath); //����ѵ���õ����ʶ��ģ��



		//��������ͷ, ʶ������
		//ע����Ҫ��֡��ת����QLabel�н�����ʾ,��ҪMat��QImage��ת��
		//ʹ��opencv�ṩ�õ�CascadeClassifierXML�ļ�ʶ���������

		//Open Camera 
		//��Ҫ�ⲿָ�_cameraState���ó�tru���ܽ���֡ʶ��
		_camera = VideoCapture (0);//����Ĭ������ͷ
		if (_camera.isOpened())
		{
			Mat frame; //֡
			vector<Rect> faces; //�������������
			Mat recFace; //ʶ�������
			CascadeClassifier detector(_haarFaceDataPath); //���ؼ���������
			ConvertMatQImage cvt; //����Mat��QImageת��

			//��ÿһ֡����ʶ����
			while (_camera.read(frame)&&_cameraState)  //��������ͨ���ⲿָ�_cameraStateдΪfalse����ֹͣ֡ʶ��
			{
				flip(frame, frame, 1);//ͼƬ���ҷ�ת
				//�Ե�ǰ֡frame���ж�߶�haar�������
				Mat resizeAndGray;
				cvtColor(frame, resizeAndGray, COLOR_BGR2GRAY);
				resize(resizeAndGray, resizeAndGray, _size);//ʹ�ó�ʼ��ʱ�ĳߴ�,��֤train��predictʱsizeһ��
				detector.detectMultiScale(resizeAndGray, faces, 1.1, 3, 0, Size(50, 50), Size(1000, 1000)); //�Ա�֡��resize�������м��

				
				int label = _labels[0];//������ݱ�ǩ��ÿ��labels����һ��,�������ȡ��һ��
				int recLabel;//Ԥ���ǩֵ

				//���������������֡
				for (int i = 0; i < faces.size(); i++)
				{
					Mat face = frame(faces[i]);//�ӵ�ǰ֡�и���Rect����ü�����λ��������

					//ʹ��BasicFaceRecognizer�������predict
					stringstream temp;
					temp << model->predict(face);
					temp >> recLabel;

					//�Լ�⵽�������������ʶ��,����Ԥ��ֵ�ͱ�ǩֵƥ��
					//��Ӧ����Ŀ���
					if (recLabel == label)
					{
						//����green��,����
						rectangle(frame, faces[i], Scalar(0, 255, 0), 1, 8, 0);
						putText(frame, string("master"), faces[i].tl(), FONT_HERSHEY_PLAIN, 1, Scalar(255, 0, 0), 2, 8);
					}
					else
					{
						rectangle(frame, faces[i], Scalar(0, 0, 255), 1, 8, 0);
						putText(frame, string("passerby"), faces[i].tl(), FONT_HERSHEY_PLAIN, 1, Scalar(255, 0, 0), 2, 8);
					}
				}
				//��ʾ��ָ��ui->label��,����forѭ������Ϊ���ܴ��ڶ�����������
				QImage img;
				img = cvt.matToQImage(&frame);
				uiLabel->setPixmap(QPixmap::fromImage(img));
			}
		}
		else
		{
			qDebug() << "camera open failed\n";
			_cameraState = false; //û�д�����ͷ,�Զ���Ϊfalse
		}
	}
}


//ѵ��ģ�ͺ���
//����ѵ������txt��Ϣ�ļ��Ķ�ȡ·�� �� ģ�͵�xml�ļ��ı��������ļ���·��
//���øú���Ӧ�ȴ򿪱�־����Ϊtrue
//�β�2, �����ļ�·��"a/b/aa.xml"�����ļ���·��"a/b"
void FaceRec::startTrain(QString trainSetTxtFilePath, QString resultXMLFilePath)
{
	using namespace std;
	using namespace cv;
	if (_trainState == true) //ѵ����־����
	{
		//�������ʹ�õ�opencvѵ���õ�cascade��������xml�ļ�:
		string _haarFaceDataPath = "haarcascade_frontalface_alt_tree.xml"; //haarcascade_frontalface_alt_tree.xml �ѷŵ���Ŀ�ļ���
		string _trainSetTxtFilePath = trainSetTxtFilePath.toStdString(); //fileName��txt�ļ�
		ifstream file(_trainSetTxtFilePath, ifstream::in);//file�Ǵ洢���������txt�ļ�,

		//���ļ�,��ȡͼƬ�ͱ�ǩ
		if (file)
		{
			//��ȡ�������train set

			string samplePath, sampleLabel;//���ڴ洢����������·���Ͷ�Ӧ��ǩ
			Mat resizeScaleAndGray;
			while (getline(file, samplePath)) //��ȡ��һ��,������·��
			{
				getline(file, sampleLabel);//��ȡ�ڶ���,��������ǩ
				if (!sampleLabel.empty() && !samplePath.empty())
				{
					stringstream temp;//����ǩת����int
					int labelInt = 0;
					temp << sampleLabel;
					temp >> labelInt;

					//����trains set 
					_labels.push_back(labelInt);//������ݼ�annotation
					//�̶��ߴ�EigenFaceRecognizer��Ҫѵ���ͼ��ʱ�̶��ߴ���Ϊ�Ҷ�ͼ
					resizeScaleAndGray = imread(samplePath, IMREAD_GRAYSCALE);//�ҶȻ���֡
					resize(resizeScaleAndGray, resizeScaleAndGray, _size);//ʹ�ó�ʼ��ʱ�ĳߴ�
					_images.push_back(resizeScaleAndGray);//��䵽���ݼ�images��

					//debugʹ��
					//int i = 0;
					//cout << "samplePath:" << samplePath << endl;
					//cout << "sampleLabel:" << sampleLabel << endl;
					//cout << "labels[" << i << "]: " << labels[i] << endl;
					//cout << "images[" << i << "]: " << images[i] << endl;
					//i++;
				}
			}

			////�����򵥽�����֤����debugʹ��
			////��һ�������Ŀ��
			//int height = images[0].rows;
			//int width = images[0].cols;
			//cout << "imageHeight= " << height << endl;
			//cout << "imageWidth= " << width << endl;
			////�򵥵Ľ�����֤, ȡǰS-1������, �������һ��������Ϊtest set�����modelԤ������
			//Mat testSample = images[int(images.size()) - 1];
			//int testLabel = labels[labels.size() - 1];
			//cout << "testLabel=" << testLabel << endl;
			//namedWindow("testImage", WINDOW_AUTOSIZE);
			//imshow("testImage", testSample);
			//cv::waitKey(60);
			////����test ����(�����һ��sample,��������ѵ��)
			//images.pop_back();
			//labels.pop_back();

			//����cv::face::BasicFaceRecognizer����
			Ptr<cv::face::BasicFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
			model->train(_images, _labels);
			//ѵ��������xml�ļ�, ������ʽ"ָ����·��+ģ����������.XML"
			model->save(resultXMLFilePath.toStdString() + "/EigenFaceRec.xml");
			_trainResult = true;//ѵ��ok�������
		}
		else
		{
			_trainResult = false;//û�н���ѵ��, ����ѵ�����
		}
	}
}

void FaceRec::begainToCommonFaceRec()
{
	using namespace std;
	using namespace cv;
	if (_recState == true) //Ӧ�ڵ��ú���ǰ����־����ʹ��
	{
		//�������ʹ�õ�opencvѵ���õ�cascade��������xml�ļ�:
		//			haarcascade_frontalface_alt_tree.xml �ѷŵ���Ŀ�ļ���
		string _haarFaceDataPath = "haarcascade_frontalface_alt_tree.xml";

		//��������ͷ, ʶ������
		//ע����Ҫ��֡��ת����QLabel�н�����ʾ,��ҪMat��QImage��ת��
		//ʹ��opencv�ṩ�õ�CascadeClassifierXML�ļ�ʶ���������

		//Open Camera 
		//��Ҫ�ⲿָ�_cameraState���ó�tru���ܽ���֡ʶ��

		//_camera = VideoCapture(0);//������ͷ�����ͱ��������뿪,�����Ƶ��������ͷ
		if (_camera.isOpened())
		{
			//Mat frame; //֡
			//vector<Rect> faces; //�������������
			//Mat recFace; //ʶ�������
			//���óɳ�Ա, ��������ÿ�ε��øú��������������ڴ�,�´ε���ֻ���¼���
			CascadeClassifier detector(_haarFaceDataPath); //���ؼ���������



			//��ÿһ֡����ʶ����
			//��ʹ��while�������������, ʵ����Ƶ��QTimer��Ƶ���ñ�����
			_camera.read(_frame);
			if (!_frame.empty() && _cameraState)  //��������ͨ���ⲿָ�_cameraStateдΪfalse����ֹͣ֡ʶ��
			{
				flip(_frame, _frame, 1);//ͼƬ���ҷ�ת
				//�Ե�ǰ֡frame���ж�߶�haar�������
				//Mat resizeAndGray;
				//cvtColor(frame, resizeAndGray, COLOR_BGR2GRAY);
				//resize(resizeAndGray, resizeAndGray, _size);//ʹ�ó�ʼ��ʱ�ĳߴ�,��֤train��predictʱsizeһ��
				detector.detectMultiScale(_frame, _faces, 1.1, 3, 0, Size(50, 50), Size(5000,5000)); //�Ա�֡��resize�������м��

				//���������������֡
				for (int i = 0; i < _faces.size(); i++)
				{
					Mat face = _frame(_faces[i]);//�ӵ�ǰ֡�и���Rect����ü�����λ��������
					rectangle(_frame, _faces[i], Scalar(255, 0, 0), 1, 8, 0);
					putText(_frame, string("FACE"), _faces[i].tl(), FONT_HERSHEY_PLAIN, 1, Scalar(255, 0, 0), 2, 8);					
				}
				//��ʾ��ָ��ui->label��,����forѭ������Ϊ���ܴ��ڶ�����������
				QImage img;
				img = _cvt.matToQImage(&_frame);
				_uiShowLabel->setPixmap(QPixmap::fromImage(img));

			}
		}
		else
		{
			qDebug() << "camera open failed\n";
			_cameraState = false; //û�д�����ͷ,�Զ���Ϊfalse
		}
	}
}


