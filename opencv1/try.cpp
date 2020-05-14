#include "opencv2\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>

#include <opencv2/opencv.hpp>
#include <vector>
#include <assert.h>



using namespace cv;
using namespace std;



struct Blob
{
	cv::Size matContainSize;
	cv::Rect boundingRect;
	std::vector<cv::Point2i> points;

	Blob(Blob* b) {};
	Blob() {};
};

std::vector<Blob> FindBlobs(Mat& matBinary, cv::Size minSize, cv::Size maxSize)
{
	assert(matBinary.channels() == 1, "Image is not binary");


	std::vector<Blob> blobs;

	// Fill the label_image with the blobs
	// 0  - background
	// 1  - unlabelled foreground
	// 2+ - labelled foreground

	uchar label_count = 2; // starts at 2 because 0,1 are used already

	for (int y = 0; y < matBinary.rows; y++)
	{
		for (int x = 0; x < matBinary.cols; x++)
		{
			uchar val = matBinary.at<uchar>(y, x);
			if (val != 255)
				continue;

			cv::Rect rect;
			cv::floodFill(matBinary, cv::Point(x, y), ++label_count, &rect);
			if (label_count == 255)
				label_count = 2;

			if (minSize.area() > 0 && (rect.width < minSize.width || rect.height < minSize.height))
				continue;

			if (maxSize.area() > 0 && (rect.width > maxSize.width || rect.height > maxSize.height))
				continue;

			Blob blob;

			for (int i = rect.y; i < (rect.y + rect.height); i++)
			{
				for (int j = rect.x; j < (rect.x + rect.width); j++)
				{
					if (matBinary.at<uchar>(i, j) == label_count)
						blob.points.push_back(cv::Point2i(j, i));
				}
			}



			if (blob.points.size() == 0)
				continue;

			blob.boundingRect = rect;
			blob.matContainSize = matBinary.size();
			blobs.push_back(blob);

		}
	}
	return blobs;
}

int main()
{
	CascadeClassifier cascade = cv::CascadeClassifier("bienso.xml");

	CascadeClassifier kytu;// = cv::CascadeClassifier("kytu.xml");
	kytu.load("kytu.xml");


	//load ảnh và chuyển thành ảnh xám
	//cv::Mat matGray = imread("./Bike_back/0569.jpg", IMREAD_GRAYSCALE);

	cv::Mat matGray = imread("a21.jpg", IMREAD_GRAYSCALE);

	/*59-N1
	62926*/
	//detect
	std::vector<cv::Rect> rects;

	// tìm kiếm
	cascade.detectMultiScale(matGray, rects, 1.1, 3, CV_HAAR_SCALE_IMAGE);

	//in ra số lượng đối tượng phát hiện được
	std::cout << "Detected " << rects.size() << " objects";

	Mat img_color = matGray;
	Mat face_roi;

	//Cắt biển số 
	//Lỗi khi có 2 biển số

	for (int n = 0; n < rects.size(); n++) {
		rectangle(img_color, rects[n], cv::Scalar(255, 0, 0), 2);
		matGray(rects[n]).copyTo(face_roi);
	}

	Mat xccw;

	int hs = img_color.cols / 600;

	if (hs == 0)
		hs = 1;

	resize(img_color, xccw, Size(img_color.cols / hs , img_color.rows / hs));
	imshow("VJ Face Detector", xccw);

	if (rects.size() == 0)
	{
		waitKey(0);
		return 0;
	}
		
	//In biển số ra màng hình

	std::cout << "\nKich thuoc la cols " << face_roi.cols << "\n của rows là " << face_roi.rows << "";

	Mat bx_cat;

	cv::resize(face_roi, bx_cat, Size(408, 300));

	cv::blur(bx_cat, bx_cat, Size(3, 3), Point(0, 0), 0);

	adaptiveThreshold(bx_cat, bx_cat, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 35, 6);
	imshow("Nhi phan nguong dong", bx_cat);

	std::vector<Rect> r_kt;

	kytu.detectMultiScale(bx_cat, r_kt, 1.1, 2,0,Size(35, 40));

	//kytu.detectMultiScale(bx_cat, r_kt, 1.1, 3, 0, cv::Size(30, 30));
	
	for (int i = 0; i < r_kt.size() - 1; i++)
	{
		for (int j = i + 1; j < r_kt.size(); j++)
		{
			if (r_kt[i].y < 120 && r_kt[j].y > 120)
			{
				Rect tamp;
				tamp = r_kt[i];
				r_kt[i] = r_kt[j];
				r_kt[j] = tamp;
			}

			if (r_kt[i].y > 120 && r_kt[j].y < 120)
			{
				Rect tamp;
				tamp = r_kt[i];
				r_kt[i] = r_kt[j];
				r_kt[j] = tamp;
			}

			if (r_kt[i].y < 120 && r_kt[j].y < 120 && r_kt[i].x > r_kt[j].x)
			{
				Rect tamp;
				tamp = r_kt[i];
				r_kt[i] = r_kt[j];
				r_kt[j] = tamp;

			}
			if (r_kt[i].y > 120 && r_kt[j].y > 120 && r_kt[i].x > r_kt[j].x)
			{
				Rect tamp;
				tamp = r_kt[i];
				r_kt[i] = r_kt[j];
				r_kt[j] = tamp;

			}
		}
	}
	for (int i = 0; i < r_kt.size(); i++)
	{
		int h, w, x, y;
		x = r_kt[i].x;
		y = r_kt[i].y;
		h = r_kt[i].height;
		w = r_kt[i].width;
		std::cout << "\nx : " << x << " - y : " << y << " - h : " << h << " - w " << w;

		if (y >= 60 && y <= 130)
			r_kt.erase(r_kt.begin() + i);
	}

	std::cout << "\nNhan dien duoc " << r_kt.size() << " ki tu";
	for (int n = 0; n < r_kt.size(); n++) {		
		rectangle(bx_cat, r_kt[n], cv::Scalar(0,255, 0), 2); 
		Mat tachChuoi;
		bx_cat(r_kt[n]).copyTo(tachChuoi);
		char name[20];
		sprintf_s(name, "face_%d.png", n);
		imshow(name, tachChuoi);

	}
	imshow("bien so", bx_cat);
	waitKey(0);
	return 0;
}

