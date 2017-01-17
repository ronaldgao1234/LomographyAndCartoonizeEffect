#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
//#include "opencv2/highgui.hpp"
#include <vector>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat img = imread("lena.png");

void showHistoCallback() {
	//separate image in bgr
	//use a vector to store each matrix
	vector <Mat> bgr;
	split(img, bgr);

	//create the histogram for 256 bins
	//the number of possible values [0...2555]
	int numbins = 256;
	
	//set the ranges for bgr, last is not included
	float range[] = { 0, 256 };
	const float* histRange = { range };

	Mat b_hist, g_hist, r_hist;
	//input image, 
	calcHist(&bgr[0], 1, 0, Mat(), b_hist, 1, &numbins, &histRange, true, false);
	calcHist(&bgr[1], 1, 0, Mat(), g_hist, 1, &numbins, &histRange);
	calcHist(&bgr[2], 1, 0, Mat(), r_hist, 1, &numbins, &histRange);

	//draw the historgram
	int width = 512;
	int height = 300;

	//create image with gray base
	//scalar value 20i + 20j + 20k
	Mat histImage(height, width, CV_8UC3, Scalar(20, 20, 20));

	//normalize the histograms to the height of the image
	normalize(b_hist, b_hist, 0, height, NORM_MINMAX);
	normalize(g_hist, g_hist, 0, height, NORM_MINMAX);
	normalize(r_hist, r_hist, 0, height, NORM_MINMAX);

	int binStep = cvRound((float)width / (float)numbins);
	for (int i = 1; i < numbins; i++) 
	{
		line(histImage, Point(binStep*(i - 1), height - cvRound(b_hist.at<float>(i - 1))),
			Point(binStep*(i), height - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0));
		line(histImage, Point(binStep*(i - 1), height - cvRound(g_hist.at<float>(i - 1))),
			Point(binStep*(i), height - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0));
		line(histImage, Point(binStep*(i - 1), height - cvRound(r_hist.at<float>(i - 1))),
			Point(binStep*(i), height - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255));
	}
	namedWindow("Histogram");
	imshow("Histogram", histImage);
}

void equalize() {
	Mat result;
	Mat ycrcb;
	cvtColor(img, ycrcb, COLOR_BGR2YCrCb);
	vector <Mat> channels;
	split(ycrcb, channels);
	
	//equalize the y channels only
	equalizeHist(channels[1], channels[1]);
	//merge the result channels
	merge(channels, ycrcb);
	cvtColor(ycrcb, result, COLOR_YCrCb2BGR);
	namedWindow("Equalized");
	imshow("Equalized", result);
}

//filters
void lomo(){
	//lomo doesn't work
	Mat result;
	const double exponential_e = std::exp(1.0);
	
	//create lookup table for color curve effect
	Mat lut(1, 256, CV_8UC1);
	for (int i = 0; i < 256; i++) {
		float x = (float)i / 256.0;
		/*lut.at<uchar>(i) = cvRound(256 * (1 / (1 + pow(exponential_e, -((x - 0.5) / 0.1)))));
	*/
		lut.at<uchar>(i) = cvRound(256 * (1 / (1 + pow(exponential_e, -((x - 0.5) / 0.1)))));
	}
	//split the image channels and apply curve transform only to red channel
	vector <Mat> bgr;
	split(img, bgr);
	LUT(bgr[2], lut, bgr[2]);
	//merge result
	merge(bgr, result);

	//create image for halo dark
	Mat halo(img.rows, img.cols, CV_32FC3, Scalar(0.3, 0.3, 0.3));

	//create circle
	circle(halo, Point(img.cols / 2, img.rows / 2), img.cols / 3, Scalar(1, 1, 1), -1);
	blur(halo, halo, Size(img.cols / 3, img.rows / 3));

	Mat resultf;
	result.convertTo(resultf, CV_32FC3);

	//multiply our result with halo to basically apply it. first need to convert values tho
	multiply(resultf, halo, resultf);

	resultf.convertTo(result, CV_8UC3);

	namedWindow("lomo");
	imshow("lomo", result);
}

void cartoonize() {
	namedWindow("Result2");
	namedWindow("Result");
	/**EDGES**/
	//apply median filter to remove possible noise
	Mat imgMedian;
	medianBlur(img, imgMedian, 7);

	//detect edges with canny
	Mat imgCanny;
	Canny(imgMedian, imgCanny, 50, 50*3);
	
	//dilate the edges
	Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
	dilate(imgCanny, imgCanny, kernel);
	
	//Scale edge values to 1 and invert values
	imgCanny = imgCanny / 255;
	//changes from black backgraound white foreground to white background black foreground
	imgCanny = 1 - imgCanny;
	
	Mat imgCannyf;
	imgCanny.convertTo(imgCannyf, CV_32F);
	
	//blur the edges to do smooth effect. undo this line later
	//gives smooth result line?
	blur(imgCannyf, imgCannyf, Size(5, 5));
	
	/*Color*/
	//Apply bilateral filter to homogenize color
	Mat imgBF;
	bilateralFilter(img, imgBF, 9, 150.0, 150.0);
	
	//truncate the colors
	Mat result = imgBF / 25;
	result = result * 25;
	
	/*Merge color and edges*/
	//create a 3 channels for edges
	//destination matrix
	Mat imgCanny3c;
	//the three channels to merge
	Mat cannyChannels[] = { imgCannyf, imgCannyf, imgCannyf };
	merge(cannyChannels, 3, imgCanny3c);

	//convert to float
	Mat resultf;
	result.convertTo(resultf, CV_32FC3);
	
	//multiply color and edge matrices
	multiply(resultf, imgCanny3c, resultf);

	////convert to 8 bits to show user
	resultf.convertTo(result, CV_8UC3);

	imshow("Result2", result);

}

int main(int argc, const char** argv){
	namedWindow("Input");
	//actually used functions since u need QT function support for createButton
	cartoonize();
	//lomo();
	//equalize();
	//showHistoCallback();
	
	//create UI buttons
	//createButton("Show histogram", showHistoCallback, NULL, QT_PUSH_BUTTON, 0);
	//createButton("Equalize historgram", equalizeCallback, NULL, QT_PUSH_BUTTON, 0);
	//createButton("Lomography effect", lomoCallback, NULL, QT_PUSH_BUTTON, 0);
	//createButton("Cartoonize effect", cartoonizeCallback, NULL, QT_PUSH_BUTTON, 0);
	
	imshow("Input", img);
	waitKey(0);
	return 0;
}

