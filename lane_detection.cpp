#include "ipp.h"
#include "ippcc.h"
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <omp.h>
#include <string>
#include <algorithm>

using namespace std; 
using namespace cv;
double right_m, left_m, img_center;
bool right_flag, left_flag;
Point right_b, left_b;


Mat deNoise(Mat inputImage)
{
	IppiSize size, tsize;
	size.width = inputImage.cols;
	size.height = inputImage.rows;
	Ipp8u* S_img = (Ipp8u*)ippsMalloc_8u(size.width*size.height);
	Ipp8u* D_img = (Ipp8u*)ippsMalloc_8u(size.width*size.height);


	ippiCopy_8u_C1R((const Ipp8u*)inputImage.data, size.width, S_img, size.width, size);
	int srcStep = 0, dstStep = 0;
	
	IppiSize roiSize = { size.width, size.height };
	Ipp8u kernel = 5;
	Ipp8u sigma = 0.3*(kernel*(kernel - 1)) * 0.5 - 1;
	int temp_buffer_size = 0, ispecSize = 0;
	IppiBorderType border = ippBorderConst;
	Ipp32f borderValue = 0.5f;

	int numOfChannels = 1;
	ippiFilterGaussianGetBufferSize(roiSize, kernel, ipp8u, numOfChannels, &ispecSize, &temp_buffer_size);
	Ipp8u *pBuffer = (Ipp8u*)ippsMalloc_8u(temp_buffer_size); //storing in new variable
	IppFilterGaussianSpec* pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u(ispecSize);
	ippiFilterGaussianInit(roiSize, kernel, sigma, border, ipp8u, numOfChannels, pSpec, pBuffer);
	ippiFilterGaussianBorder_8u_C1R(S_img, size.width, D_img, size.width, roiSize, borderValue, pSpec, pBuffer);
	Size s;
	s.width = inputImage.cols;
	s.height = inputImage.rows;
	Mat output(s, CV_8U, D_img);
	cvtColor(output, output, COLOR_GRAY2BGR);
	
	return output;
}
Mat edgeDetect(Mat blurImg)
{
	Mat output, kernel;
	Point anchor;
	cvtColor(blurImg, output, COLOR_BGR2GRAY);
	//adaptiveThreshold(output, output, 140, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7,12);
	threshold(output, output, 150, 255, THRESH_BINARY_INV);
	anchor = Point(-1, -1);
	kernel = Mat(1, 3, CV_32F);

	kernel.at<float>(0, 0) = -1;
	kernel.at<float>(0, 1) = 0;
	kernel.at<float>(0, 2) = 1;

	filter2D(output, output, -1, kernel, anchor, 0, BORDER_DEFAULT);
	return output;
}
Mat masking(Mat edges)
{
	int width = edges.cols;
	int height = edges.rows;
	Mat output, mask;
	mask = Mat::zeros(edges.size(), edges.type());
	Point pts[4] = { Point(0,height),Point(width/3.6, height/1.37),Point(width/1.8, height/1.37),Point(width,height) };
	fillConvexPoly(mask, pts, 4, Scalar(255, 0, 0));
	bitwise_and(edges, mask, output);
	return output;
}
vector<Vec4i> houghLines(Mat maskImg)
{
	vector<Vec4i> lines;
	HoughLinesP(maskImg, lines, 1, CV_PI / 180, 20, 20, 30);
	return lines;
}
vector<vector<Vec4i>> lineSeparation(vector<Vec4i> line, Mat maskImg)
{
	vector<vector<Vec4i>> output(2);
	size_t j = 0;
	Point right, left;
	double slope_threshold = 0.3;
	vector<double> slopes;
	vector<Vec4i> selected_lines;
	vector<Vec4i> right_line, left_line;


	for (auto i : line)
	{
		right = Point(i[0], i[1]);
		left = Point(i[2], i[3]);
		double slope = (static_cast<double>(left.y) - static_cast<double>(right.y)) /
			(static_cast<double>(left.x) - static_cast<double>(right.x) + 0.00001);

		if (abs(slope) > slope_threshold)
		{
			slopes.push_back(slope);
			selected_lines.push_back(i);
		}
	}
	img_center = static_cast<double>(maskImg.cols / 2);
	while (j < selected_lines.size())
	{
		right = Point(selected_lines[j][0], selected_lines[j][1]);
		left = Point(selected_lines[j][2], selected_lines[j][3]);
		if (slopes[j] > 0 && left.x > img_center && right.x > img_center)
		{
			right_line.push_back(selected_lines[j]);
			right_flag = true;
		}
		else if (slopes[j] < 0 && left.x < img_center && right.x < img_center)
		{
			left_line.push_back(selected_lines[j]);
			left_flag = true;
		}
		j++;
	}
	output[0] = right_line;
	output[1] = left_line;
	return output;
}

vector<Point> regression(vector<vector<Vec4i>> left_right_lines, Mat image)
{
	vector<Point> output(4);
	Point right_one, right_two;
	Point left_one, left_two;
	Vec4d right_line, left_line;

	vector<Point> right_pts, left_pts;

	if (right_flag == true)
	{
		for (auto i : left_right_lines[0])
		{
			right_one = Point(i[0], i[1]);
			right_two = Point(i[2], i[3]);

			right_pts.push_back(right_one);
			right_pts.push_back(right_two);
		}
		if (right_pts.size() > 0)
		{
			fitLine(right_pts, right_line, DIST_L2, 0, 0.01, 0.01);
			right_m = right_line[1] / right_line[0];
			right_b = Point(right_line[2], right_line[3]);
		}

	}
	if (left_flag == true)
	{
		for (auto j : left_right_lines[1])
		{
			left_one = Point(j[0], j[1]);
			left_two = Point(j[2], j[3]);

			left_pts.push_back(left_one);
			left_pts.push_back(left_two);
		}
		if (left_pts.size() > 0)
		{
			fitLine(left_pts, left_line, DIST_L2, 0, 0.01, 0.01);
			left_m = left_line[1] / left_line[0];
			left_b = Point(left_line[2], left_line[3]);
		}
	}
	double right_ini_x, right_fin_x, left_ini_x, left_fin_x;

	int ini_y = image.rows;
	int fin_y = image.cols / 2.25;

	right_ini_x = (ini_y - right_b.y) / right_m + right_b.x;
	right_fin_x = (fin_y - right_b.y) / right_m + right_b.x;
	left_ini_x = (ini_y - left_b.y) / left_m + left_b.x;
	left_fin_x = (fin_y - left_b.y) / left_m + left_b.x;

	output[0] = Point(right_ini_x, ini_y);
	output[1] = Point(right_fin_x, fin_y);
	output[2] = Point(left_ini_x, ini_y);
	output[3] = Point(left_fin_x, fin_y);
	return output;
}

string turn()
{
	string output;
	double vanish_x, thresh_vp = 10;
	vanish_x = static_cast<double>(((right_m*right_b.x) - (left_m*left_b.x) - right_b.y + left_b.y) / (right_m - left_m));
	if (vanish_x < (img_center - thresh_vp))
	{
		output = "Left";
	}
	else if (vanish_x > (img_center + thresh_vp))
	{
		output = "Right";
	}
	else if (vanish_x >= (img_center - thresh_vp) && vanish_x <= (img_center + thresh_vp))
	{
		output = "Forward";
	}
	return output;
}
void plotLane(Mat inputImage, vector<Point> lane)
{
	vector<Point> poly_points;
	Mat output;

	inputImage.copyTo(output);

	poly_points.push_back(lane[2]);
	poly_points.push_back(lane[0]);
	poly_points.push_back(lane[1]);
	poly_points.push_back(lane[3]);
	//cout << lane[2] << " "<< lane[0] << " "<< lane[1]<<" "<< lane[3]<<endl; 
	fillConvexPoly(output, poly_points, Scalar(255, 255, 255));
	addWeighted(output, 0.3, inputImage, 1.0 - 3.0, 0, inputImage);
	line(inputImage, lane[0], lane[1], Scalar(0, 255, 255), 2);
	line(inputImage, lane[2], lane[3], Scalar(0, 255, 255), 2);

	imshow("Lane", inputImage);
}

int main()
{
	string dir;
	Mat input;
	Mat bgr, gray, blur; 
	vector<Vec4i> new_vec;
	vector<Point> regress;
	vector<vector<Vec4i>> line_sep;
	double start, end;
	double difftime;
	VideoCapture capture("road.avi");
	double fps; 	
	int length = int(capture.get(CAP_PROP_FRAME_COUNT));
	capture >> input;
	int width = input.cols/4;
	int height = input.rows/4;
	if (!capture.isOpened())
	{
		cout << "error can't open file ";
		return -1;
	}
	while (1)
	{
		start = omp_get_wtime();
		capture >> input;
		fps = capture.get(CAP_PROP_FPS);

		resize(input, input, Size(width, height));
		
		imshow("input", input);
		cvtColor(input, input, COLOR_BGR2GRAY);

		if (input.empty())
		{
			cout << "frame is empty! ";
			break;
		}
		
		blur = deNoise(input);
		bgr = edgeDetect(blur);
		gray = masking(bgr);
		new_vec = houghLines(gray);
		line_sep = lineSeparation(new_vec, gray);
		regress = regression(line_sep, gray);
		plotLane(bgr, regress);
		dir = turn();
		end = omp_get_wtime();
		
		difftime = (end - start);//in seconds
		fps = 1/difftime;
		if (dir == "Forward")
		{
			cout <<"size = " << width << "x" << height 
				<< " direction: " << dir  << " fps:"
				<< fps << endl;
		}
		if (dir == "Left")
		{
			cout << "size = " << width << "x" << height
				<< " direction: " << "smooth left"<<
				" fps: " << fps << endl;
		}
		if (dir == "Right")
		{
			cout << "size = " << width << "x" << height
				<< " direction: " << "smooth right" 
				<< " fps: " << fps << endl;
		}


		//imshow("masking", input);
		if (waitKey(30) == 27) // Wait for 'esc' key press to exit
		{
			break;
		}
		//cout << "Fps: " << fps<<endl;
	}
	capture.release();
	destroyAllWindows();
	return 0;
}