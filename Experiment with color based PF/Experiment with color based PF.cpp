#include <stdio.h>
#include <time.h>
#include <cstdio>
#include <math.h>
#include <ctype.h>
#include <iostream>
#include<functional>  
#include <vector> 
#include <fstream>
#include "stdafx.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "ColorProbTrack.h"
#include <windows.h>
#include <Kinect.h>

//include Cuda head files
#include "cuda_runtime.h"    
#include "device_launch_parameters.h"  

//include depth data process head files
using namespace cv;
#include <queue> 
using namespace std;

//for color information process
#define BIN 40 
#define BOARD 3  
#define LAMBDA 10 
#define TIME_DELAY 1000000 
#define IMGWIDTH 30
#define SAMPLE 100

// for real-time tracking in this lab
#define VAR_U 2 
#define VAR_V 1
#define VAR_UPRIME 0.1
#define VAR_VPRIME 0.1
#define VAR_W 0.2
#define VAR_H 0.2
#define VAR_WPRIME 0.1
#define VAR_HPRIME 0.1
#define FACTOR 0.1

//for depth data process
#define VertexNum 2500
#define THRESHOLD 100000
#define DEPTHW 200
#define N 6  

Mat* DrawSpinImage(int x, int y);
void getSpinImage(int x, int y, float spin[IMGWIDTH][IMGWIDTH]);
float spin_refer[IMGWIDTH][IMGWIDTH];

int FrameNumber = 0;

double stepX, stepY;


int FindMin(float disk[], int s[]);
float global_x_prime, global_y_prime;
float GetRand();

void onMouse(int event, int x, int y, int, void*);
void DrawRectangle(cv::Mat& img, cv::Rect box);
Region AreaToRegion(Rect area);
float smaller(float a, float b);
float bigger(float a, float b);
CvRect RegionToArea(Region r);
float CalcDistance(float hist[BIN][BIN], float hist_ref[BIN][BIN]);
double CalcProb(double d);
CvScalar hsv2rgb(float hue);
void Initdata(CvRect area);
void ReInitData();
double uniform_random(void);
void predict();
double gaussian_random(void);
int FindIndex(double r_num);
void update(float weight, float delta_u, float delta_v, float delta_w, float delta_h);
CvRect ImProcess(IplImage *hue, IplImage *sat, float hist_ref[BIN][BIN], CvRect track_win_last);

IplImage *frame = 0;
IplImage *image, *frame_static;
IplImage *result, *hsv, *hue, *sat;
IplImage *histimg_ref, *histimg;

int i, j, k, h;
int bin_w, bin_h, bin_t = 0, c;

float hist_ref[BIN][BIN], hist[BIN][BIN];
CvRect area, track_win, track_win_last;
CvScalar color, im_hue, im_sat;


const float STEP_HUE = 256.0 / BIN;
const float STEP_SAT = 256.0 / BIN;

int hbins = BIN, hbins_ref = BIN;
float hranges_arr[] = { 0, 179 };
float *hranges = hranges_arr;


int selecting;
int tracking;
int output;
Rect g_rectangle;
Mat showImage;
Point P_org, P_end, P_win;
IterData data;
IterData data1;


UINT16 *pBuffer = NULL;
int depthData[424 * 512];
int depthWidth = 512;
int depthHeight = 424;
int colorWidth = 1920;
int colorHeight = 1080;
cv::Mat depthBufferMat(depthHeight, depthWidth, CV_16UC1);
cv::Mat depthMat(depthHeight, depthWidth, CV_8UC1);
cv::Mat colorBufferMat(colorHeight, colorWidth, CV_8UC4);
cv::Mat colorMat(colorHeight / 2, colorWidth / 2, CV_8UC4);
cv::Mat coordinateMapperMat(424, 512, CV_8UC4);

//depth********depth********depth********depth********depth********depth********depth********depth********depth********depth********depth********depth********depth********
#define VAR_U_1 5
#define VAR_V_1 5
#define VAR_UPRIME_1 0.1
#define VAR_VPRIME_1 0.1

int PointSelect, PointTrack;
struct Particlepoint {
	int x, y, z, x_prime, y_prime, x_1, y_1, z_1;
	float x_1_prime, y_1_prime, SampleWeight, AccumWeight;
};
Particlepoint point[SAMPLE];
Point P_track, P_extre, P_transtart;
int refeFlag;
Point p_win, p_win_last;
float referDistance;

struct ArcNode {
	int adjvex;
	int weight;
	ArcNode *next;
};

struct VertexNode {
	int vertex;
	int ind;
	bool v;//define whether it belongs to the background
	ArcNode *firstarc;
};

vector<VertexNode>adjlist;

void DFS(int v, bool visited[])
{
	visited[v] = true;
	ArcNode *p = adjlist[v].firstarc;

	while (p)
	{
		int j = p->adjvex;
		if (!visited[j]) DFS(j, visited);
		p = p->next;
	}
}

void ConstructMesh(int DepthData[], vector<VertexNode>& adjlist, Point start, int width, int height)
{
	int temp;
	bool visited[90000] = { false };
	adjlist.resize(width*height);

	//initialize vertices
	for (int i = 0; i <height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int index = width*i + j;

			adjlist[index].vertex = int(start.x + j) + int(start.y + i) * 512;
			adjlist[index].ind = index;

			adjlist[index].v = false;
			adjlist[index].firstarc = NULL;

		}
	}

	//initialize edges
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int index = width*i + j;

			if (j + 1 < width)
			{
				temp = abs(DepthData[adjlist[index].vertex] - DepthData[adjlist[index + 1].vertex]);

				if (temp < 10) //threshold for not connect but pixel neighbored points where 100mm=10cm
				{

					ArcNode *s = new ArcNode;
					s->adjvex = index + 1;
					s->weight = temp;
					s->next = adjlist[index].firstarc;
					adjlist[index].firstarc = s;

					ArcNode *s1 = new ArcNode;
					s1->adjvex = index;
					s1->weight = temp;
					s1->next = adjlist[index + 1].firstarc;
					adjlist[index + 1].firstarc = s1;
				}

			}

			if (i + 1 < height)
			{

				temp = abs(DepthData[adjlist[index].vertex] - DepthData[adjlist[index + width].vertex]);

				if (temp < 10)
				{

					ArcNode *s2 = new ArcNode;
					s2->adjvex = index + width;
					s2->weight = temp;
					s2->next = adjlist[index].firstarc;
					adjlist[index].firstarc = s2;

					ArcNode *s3 = new ArcNode;
					s3->adjvex = index;
					s3->weight = temp;
					s3->next = adjlist[index + width].firstarc;
					adjlist[index + width].firstarc = s3;
				}
			}

		}
	}

	//use DFS to find the connected components
	int v = (p_win.x - start.x) + (p_win.y - start.y) * width;
	DFS(v, visited);

	//Plot connected componets
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (visited[width*i + j] == true)
			{
				adjlist[width*i + j].v = true;
				CvPoint pp; pp.x = j + start.x; pp.y = i + start.y;
				cvCircle(image, pp, 1, CV_RGB(255, 0, 0), 1);
			}
		}
	}

	

}

void ShortestPath(Point s, vector<VertexNode>& adjlist, vector<int>& dist) {

	//int Path[VertexNum];//the vertex just before vertex i
	vector<int> Path;
	Path.resize(track_win.width*track_win.height);

	//int S[VertexNum] = { 0 };//record whether s[i] is added
	vector<bool> visited;
	visited.resize(track_win.width*track_win.height);

	for (int i = 0; i < dist.size(); i++)
	{
		dist[i] = INT_MAX;
		Path[i] = -1;
		visited[i] = false;
	}

	//int v = 0;
	int v = (s.x - P_transtart.x) + (s.y - P_transtart.y)*track_win.width;
	dist[v] = 0;

	queue <VertexNode >  q;
	q.push(adjlist[v]);

	while (!q.empty())
	{
		VertexNode cd = q.front();
		q.pop();

		int u = cd.ind;

		if (visited[u] || adjlist[u].v == false) continue;

		visited[u] = true;

		ArcNode *p = adjlist[u].firstarc;

		while (p != NULL)
		{
			int v = p->adjvex;

			if (!visited[v] && dist[v] > dist[u] + p->weight)
			{
				dist[v] = dist[u] + p->weight;
				Path[v] = u;
				q.push(adjlist[v]);
			}
			p = p->next;
		}


	}


}

// generate uniform distributed values range [-1, 1]
float GetRand(void)
{
	return 2.0 * rand() / RAND_MAX - 1.0;
}

void PredictPoint(Point start)
{
	for (i = 0; i < SAMPLE; i++)
	{
		point[i].x = bigger(start.x, int(point[i].x_1 + point[i].x_1_prime + VAR_U_1 *GetRand()));
		point[i].x = smaller((start.x + DEPTHW), point[i].x);

		point[i].y = bigger(start.y, int(point[i].y_1 + point[i].y_1_prime + VAR_V_1 *GetRand()));
		point[i].y = smaller((start.y + DEPTHW), point[i].y);


		point[i].x_prime = int(point[i].x_1_prime + VAR_UPRIME_1*GetRand());
		point[i].y_prime = int(point[i].y_1_prime + VAR_VPRIME_1*GetRand());


		Point p; p.x = point[i].x; p.y = point[i].y;
		cvCircle(result, p, 5, CV_RGB(105, 100, 50),1);
	}
}

void UpdatePoint(int number, float delta_x, float delta_y)
{
	for (i = 0; i < SAMPLE; i++)
	{
		point[i].x_1 = point[number].x;
		point[i].x_1_prime = delta_x;
		point[i].y_1_prime = delta_y;
	}

	P_transtart.x = P_transtart.x + delta_x;
	P_transtart.y = P_transtart.y + delta_y;

}
//depth********depth********depth********depth********depth********depth********depth********depth********depth********depth********depth********depth********depth********depth********
// Safe release for interfaces
template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{

	if (pInterfaceToRelease != NULL)
	{
		pInterfaceToRelease->Release();
		pInterfaceToRelease = NULL;
	}
}

void onMouse(int event, int x, int y, int, void*)
{

	if (selecting)
	{
		P_end = cvPoint(x, y);

	}


	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
	{
		P_org = Point(x, y);
		P_end = Point(x, y);
		selecting = 1;
		tracking = 0;
	}

	break;
	case CV_EVENT_LBUTTONUP:
	{
		tracking = 1;
	}
	break;

	case CV_EVENT_MBUTTONDOWN:
	{
		//P_start = Point(x, y);

		P_extre = Point(x, y);
		printf("P_extre:%d,%d\n", P_extre.x, P_extre.y);

		PointSelect = 1;
	}
	break;

	case CV_EVENT_RBUTTONDOWN:
	{

		P_track = Point(x, y);
		printf("P_track:%d,%d\n", P_track.x, P_track.y);
		PointTrack = 1;
	}
	break;

	}
}

// compare the value of two floats and return the smaller one
float smaller(float a, float b)
{
	if (a < b)
		return a;
	else
		return b;
}

// compare the value of two floats and return the bigger one
float bigger(float a, float b)
{
	if (a > b)
		return a;
	else
		return b;
}

// generate uniform distributed values range [0, 1]
double uniform_random(void)
{
	return (double)rand() / (double)RAND_MAX;
}

// generate Gaussian distributed values with mean=0, variance=1, and amplifier=1
double gaussian_random(void)
{
	static int next_gaussian = 0;
	static double saved_gaussian_value;

	double fac, rsq, v1, v2;

	if (next_gaussian == 0)
	{
		do
		{
			v1 = 2.0*uniform_random() - 1.0;
			v2 = 2.0*uniform_random() - 1.0;
			rsq = v1*v1 + v2*v2;
		} while (rsq >= 1.0 || rsq == 0.0);

		fac = sqrt(-2.0*log(rsq) / rsq);
		saved_gaussian_value = v1*fac;
		next_gaussian = 1;

		return v2*fac;
	}
	else
	{
		next_gaussian = 0;
		return saved_gaussian_value;
	}
}

// find the index of a sample whose value is most close to "rand"
int FindIndex(float rand)
{
	int low, middle, high;

	low = 0;
	high = SAMPLE;

	while (high>(low + 1))
	{
		middle = (high + low) / 2;
		if (rand > data1.accum_weight[middle])
			low = middle;
		else high = middle;
	}

	return low;
}

// convert rectangular area to state vector
Region AreaToRegion(CvRect area)
{
	Region r;

	r.u = float(area.x) + float(area.width / 2);
	r.v = float(area.y) + float(area.height / 2);
	r.w = float(area.width);
	r.h = float(area.height);

	return r;
}

// convert state vector to rectangular area
CvRect RegionToArea(Region r)
{
	CvRect area;

	area.height = bigger(BOARD, r.h);
	area.width = bigger(BOARD, r.w);

	if ((r.v - area.height / 2 < 0) || (r.v + area.height / 2 > image->height))
		area.height = smaller(image->height - r.v, r.v);

	if ((r.u - area.width / 2 < 0) || (r.u + area.width / 2 > image->width))
		area.width = smaller(image->width - r.u, r.u);

	area.x = r.u - area.width / 2;
	area.y = r.v - area.height / 2;

	return area;
}

// calculate Bhattacharyya distance between two color histograms
float CalcDistance(float hist[BIN][BIN], float hist_ref[BIN][BIN])
{
	int i, j;
	float distance, temp = 0.0;

	for (i = 0; i < BIN; i++)
		for (j = 0; j < BIN; j++)
			temp = temp + sqrt(hist[i][j] * hist_ref[i][j]);

	distance = sqrt(1 - temp);

	return distance;
}

// calculate weight of a sample given the Bhattacharyya distance "d"
float CalcProb(float d)
{
	float probility;

	probility = exp(-LAMBDA * d * d);

	return probility;
}

// initialize samples 
void Initdata(CvRect area)
{
	int i;
	Region r;

	r = AreaToRegion(area);

	for (i = 0; i < SAMPLE; i++)
	{
		data1.sample_t_1[i].u = r.u;
		data1.sample_t_1[i].v = r.v;
		data1.sample_t_1[i].u_prime = 0.0;
		data1.sample_t_1[i].v_prime = 0.0;
		data1.sample_t_1[i].w = r.w;
		data1.sample_t_1[i].h = r.h;
		data1.sample_t_1[i].w_prime = 0.0;
		data1.sample_t_1[i].h_prime = 0.0;

	}
}


// Re-initialize samples when a target is missing in current video frame
void ReInitData()
{
	for (int i = 0; i < SAMPLE; i++)
	{
		data1.sample_t_1[i].u = (image->width - BOARD) * fabs(uniform_random());
		data1.sample_t_1[i].v = (image->height - BOARD) * fabs(uniform_random());
		data1.sample_t_1[i].w = image->width * fabs(uniform_random());
		data1.sample_t_1[i].h = image->height * fabs(uniform_random());
		data1.sample_t_1[i].u_prime = 0;
		data1.sample_t_1[i].v_prime = 0;
		data1.sample_t_1[i].w_prime = 0;
		data1.sample_t_1[i].h_prime = 0;
	}
}

// Sample propagation based on the system dynamic model: M_t = A*M_{t-1} + n_{t-1}
void predict()
{
	int i;

	for (i = 0; i < SAMPLE; i++)
	{
		data1.sample_t[i].u = bigger(BOARD, data1.sample_t_1[i].u + data1.sample_t_1[i].u_prime
			+ VAR_U * gaussian_random());
		data1.sample_t[i].u = smaller(image->width - BOARD, data1.sample_t[i].u);
		data1.sample_t[i].v = bigger(BOARD, data1.sample_t_1[i].v + data1.sample_t_1[i].v_prime
			+ VAR_V * gaussian_random());
		data1.sample_t[i].v = smaller(image->height - BOARD, data1.sample_t[i].v);

		data1.sample_t[i].w = bigger(BOARD, data1.sample_t_1[i].w + data1.sample_t_1[i].w_prime
			+ VAR_W * gaussian_random());
		data1.sample_t[i].w = smaller(image->width - BOARD, data1.sample_t[i].w);
		data1.sample_t[i].h = bigger(BOARD, data1.sample_t_1[i].h + data1.sample_t_1[i].h_prime
			+ VAR_H * gaussian_random());
		data1.sample_t[i].h = smaller(image->height - BOARD, data1.sample_t[i].h);


		data1.sample_t[i].u_prime = data1.sample_t_1[i].u_prime + VAR_UPRIME * gaussian_random();
		data1.sample_t[i].v_prime = data1.sample_t_1[i].v_prime + VAR_VPRIME * gaussian_random();
		data1.sample_t[i].w_prime = data1.sample_t_1[i].w_prime + VAR_WPRIME * gaussian_random();
		data1.sample_t[i].h_prime = data1.sample_t_1[i].h_prime + VAR_HPRIME * gaussian_random();
	}
}

// update the posterior PDF based on weights of samples
void update(float weight, float delta_u, float delta_v, float delta_w, float delta_h)
{
	int i, index;
	float rand;

	for (i = 0; i < SAMPLE; i++)
		data1.accum_weight[i] = data1.accum_weight[i] / weight;

	for (i = 0; i < SAMPLE; i++)
	{
		rand = fabs(uniform_random());
		index = FindIndex(rand);

		//more samples will generated around the previous particles with high probability
		data1.sample_t_1[i] = data1.sample_t[index]; 
		//data1.sample_t_1[i] = data1.sample_t[index];

		data1.sample_t_1[i].u_prime = delta_u;
		data1.sample_t_1[i].v_prime = delta_v;
		data1.sample_t_1[i].w_prime = delta_w;
		data1.sample_t_1[i].h_prime = delta_h;
	}
}

// main body of particle filtering
CvRect ImProcess(IplImage *hue, IplImage *sat, float hist_ref[BIN][BIN], CvRect track_win_last)
{
	int i, j, k, h, in, index;
	float v_max;
	float hist[BIN][BIN];
	float dist[SAMPLE], weight_total = 0.0, weight_high = 0.0;
	float delta_u, delta_v, delta_w, delta_h;

	CvScalar im_hue, im_sat;
	CvRect trackwin;

	// sample propagation
	predict();

	for (in = 0; in < SAMPLE; in++)
	{
		v_max = 0.0;
		trackwin = RegionToArea(data1.sample_t[in]);

		// establish 2D hue and saturation histogram of a sample
		cvSetImageROI(hue, trackwin);
		cvSetImageROI(sat, trackwin);
		for (i = 0; i < BIN; i++)
			for (j = 0; j < BIN; j++)
				hist[i][j] = 0.0;
		for (i = 0; i < trackwin.height; i++)
		{
			for (j = 0; j < trackwin.width; j++)
			{
				im_hue = cvGet2D(hue, i, j);
				im_sat = cvGet2D(sat, i, j);
				k = int(im_hue.val[0] / STEP_HUE);
				h = int(im_sat.val[0] / STEP_SAT);
				hist[k][h] = hist[k][h] + 1.0;
			}
		}
		for (i = 0; i < BIN; i++)
		{
			for (j = 0; j < BIN; j++)
			{
				hist[i][j] = hist[i][j] / (trackwin.height*trackwin.width);
				if (hist[i][j] > v_max)
					v_max = hist[i][j];
			}
		}
		cvResetImageROI(hue);
		cvResetImageROI(sat);

		cvRectangle(result, cvPoint(trackwin.x, trackwin.y), cvPoint(trackwin.x +
			trackwin.width, trackwin.y + trackwin.height), CV_RGB(255, 0, 255), 1);

		// calculate Bhattacharyya distance between reference histogram and sample histogram
		dist[in] = CalcDistance(hist, hist_ref);
		//printf("The distance is %f \n", dist[in]);

		// calculate sample weight based on Bhattacharyya distance
		data1.sample_weight[in] = CalcProb(dist[in]);   //sample weight is 

		data1.accum_weight[in] = weight_total;

		// find the index of the sample with the highest weight
		if (weight_high < data1.sample_weight[in])
		{
			weight_high = data1.sample_weight[in];
			index = in;
		}

		// accumulate weights
		weight_total = weight_total + data1.sample_weight[in];
	}

	// a target is lost if all samples have small weights, 
	// then set the size of tracking window equal to zero 
	if (weight_total < SAMPLE * FACTOR)
	{
		trackwin.x = trackwin.y = 0;
		trackwin.height = trackwin.width = 0;
		printf("Frame is lost!!!!");
		return trackwin;
	}
	// else, pick up the sample of highest weight as the result,
	// and update the posterior PDF based on weights of samples
	else
	{
		trackwin = RegionToArea(data1.sample_t[index]);

		delta_u = trackwin.x - track_win_last.x;
		delta_v = trackwin.y - track_win_last.y;
		delta_w = trackwin.width - track_win_last.width;
		delta_h = trackwin.height - track_win_last.height;

		update(weight_total, delta_u, delta_v, delta_w, delta_h);

		printf("\n");

		return trackwin;
	}
}

//*******************************************************************************************************************************************


int main()
{

	// name and position windows
	cvNamedWindow("Color Probabilistic Tracking - Samples", 1);
	cvMoveWindow("Color Probabilistic Tracking - Samples", 0, 0);
	cvNamedWindow("Color Probabilistic Tracking - Result", 1);
	cvMoveWindow("Color Probabilistic Tracking - Result", 1000, 0);

	//control mouse
	setMouseCallback("Color Probabilistic Tracking - Samples", onMouse, 0);

	cv::setUseOptimized(true);

	// Sensor
	IKinectSensor* pSensor;
	HRESULT hResult = S_OK;
	hResult = GetDefaultKinectSensor(&pSensor);
	if (FAILED(hResult)) {
		std::cerr << "Error : GetDefaultKinectSensor" << std::endl;
		return -1;
	}

	hResult = pSensor->Open();
	if (FAILED(hResult)) {
		std::cerr << "Error : IKinectSensor::Open()" << std::endl;
		return -1;
	}

	// Source
	IColorFrameSource* pColorSource;
	hResult = pSensor->get_ColorFrameSource(&pColorSource);
	if (FAILED(hResult)) {
		std::cerr << "Error : IKinectSensor::get_ColorFrameSource()" << std::endl;
		return -1;
	}

	IDepthFrameSource* pDepthSource;
	hResult = pSensor->get_DepthFrameSource(&pDepthSource);
	if (FAILED(hResult)) {
		std::cerr << "Error : IKinectSensor::get_DepthFrameSource()" << std::endl;
		return -1;
	}

	/*IBodyIndexFrameSource* pBodyIndexSource;
	hResult = pSensor->get_BodyIndexFrameSource(&pBodyIndexSource);*/

	// Reader
	IColorFrameReader* pColorReader;
	hResult = pColorSource->OpenReader(&pColorReader);
	if (FAILED(hResult)) {
		std::cerr << "Error : IColorFrameSource::OpenReader()" << std::endl;
		return -1;
	}

	IDepthFrameReader* pDepthReader;
	hResult = pDepthSource->OpenReader(&pDepthReader);
	if (FAILED(hResult)) {
		std::cerr << "Error : IDepthFrameSource::OpenReader()" << std::endl;
		return -1;
	}

	//IBodyIndexFrameReader* pBodyIndexReader;//saferealease
	//hResult = pBodyIndexSource->OpenReader(&pBodyIndexReader);

	// Description
	IFrameDescription* pColorDescription;
	hResult = pColorSource->get_FrameDescription(&pColorDescription);
	if (FAILED(hResult)) {
		std::cerr << "Error : IColorFrameSource::get_FrameDescription()" << std::endl;
		return -1;
	}

	int colorWidth = 0;
	int colorHeight = 0;
	pColorDescription->get_Width(&colorWidth); // 1920
	pColorDescription->get_Height(&colorHeight); // 1080
	unsigned int colorBufferSize = colorWidth * colorHeight * 4 * sizeof(unsigned char);

	cv::Mat colorBufferMat(colorHeight, colorWidth, CV_8UC4);
	cv::Mat colorMat(colorHeight / 2, colorWidth / 2, CV_8UC4);
	cv::namedWindow("Color");

	RGBQUAD* m_pDepthRGBX;
	m_pDepthRGBX = new RGBQUAD[512 * 424];// create heap storage for color pixel data in RGBX format

	IFrameDescription* pDepthDescription;
	hResult = pDepthSource->get_FrameDescription(&pDepthDescription);
	if (FAILED(hResult)) {
		std::cerr << "Error : IDepthFrameSource::get_FrameDescription()" << std::endl;
		return -1;
	}

	int depthWidth = 0;
	int depthHeight = 0;
	pDepthDescription->get_Width(&depthWidth); // 512
	pDepthDescription->get_Height(&depthHeight); // 424
	unsigned int depthBufferSize = depthWidth * depthHeight * sizeof(unsigned short);

	cv::Mat depthBufferMat(depthHeight, depthWidth, CV_16UC1);
	UINT16* pDepthBuffer = nullptr;
	cv::Mat depthMat(depthHeight, depthWidth, CV_8UC1);
	cv::namedWindow("Depth");

	//UINT32 nBodyIndexSize = 0;
	//BYTE* pIndexBuffer = nullptr;//This hasn't been safe realease yet

	// Coordinate Mapper
	ICoordinateMapper* pCoordinateMapper;
	hResult = pSensor->get_CoordinateMapper(&pCoordinateMapper);
	if (FAILED(hResult)) {
		std::cerr << "Error : IKinectSensor::get_CoordinateMapper()" << std::endl;
		return -1;
	}

	cv::Mat coordinateMapperMat(depthHeight, depthWidth, CV_8UC4);
	cv::namedWindow("CoordinateMapper");

	unsigned short minDepth, maxDepth;
	pDepthSource->get_DepthMinReliableDistance(&minDepth);
	pDepthSource->get_DepthMaxReliableDistance(&maxDepth);




	while (1) {

		double t = (double)getTickCount();


		// Color Frame
		IColorFrame* pColorFrame = nullptr;
		hResult = pColorReader->AcquireLatestFrame(&pColorFrame);
		if (SUCCEEDED(hResult)) {
			hResult = pColorFrame->CopyConvertedFrameDataToArray(colorBufferSize, reinterpret_cast<BYTE*>(colorBufferMat.data), ColorImageFormat::ColorImageFormat_Bgra);
			if (SUCCEEDED(hResult)) {
				cv::resize(colorBufferMat, colorMat, cv::Size(), 0.5, 0.5);
			}
		}
		//SafeRelease( pColorFrame );

		// Depth Frame
		IDepthFrame* pDepthFrame = nullptr;
		hResult = pDepthReader->AcquireLatestFrame(&pDepthFrame);

		if (SUCCEEDED(hResult)) {
			hResult = pDepthFrame->AccessUnderlyingBuffer(&depthBufferSize, reinterpret_cast<UINT16**>(&depthBufferMat.data));

		}

		if (SUCCEEDED(hResult)) {
			hResult = pDepthFrame->AccessUnderlyingBuffer(&depthBufferSize, &pDepthBuffer);

			if (SUCCEEDED(hResult))
			{
				RGBQUAD* pRGBX = m_pDepthRGBX;

				// end pixel is start + width*height - 1
				const UINT16* pBufferEnd = pDepthBuffer + (512 * 424);
				int index = 0;

				while (pDepthBuffer < pBufferEnd)
				{
					USHORT depth = *pDepthBuffer;

					BYTE intensity = static_cast<BYTE>((depth >= 50) && (depth <= 5000) ? (depth % 256) : 0);

					pRGBX->rgbRed = intensity;
					pRGBX->rgbGreen = intensity;
					pRGBX->rgbBlue = intensity;

					depthData[index] = depth;

					++index;
					++pRGBX;
					++pDepthBuffer;
				}
			}
		}

		Mat DepthImage(424, 512, CV_8UC4, m_pDepthRGBX);
		//SafeRelease( pDepthFrame );

		// Mapping (Depth to Color)
		if (SUCCEEDED(hResult)) {
			std::vector<ColorSpacePoint> colorSpacePoints(depthWidth * depthHeight);
			hResult = pCoordinateMapper->MapDepthFrameToColorSpace(depthWidth * depthHeight, reinterpret_cast<UINT16*>(depthBufferMat.data), depthWidth * depthHeight, &colorSpacePoints[0]);
			if (SUCCEEDED(hResult)) {
				coordinateMapperMat = cv::Scalar(0, 0, 0, 0);
				for (int y = 0; y < depthHeight; y++) {
					for (int x = 0; x < depthWidth; x++) {
						unsigned int index = y * depthWidth + x;
						ColorSpacePoint point = colorSpacePoints[index];
						int colorX = static_cast<int>(std::floor(point.X + 0.5));
						int colorY = static_cast<int>(std::floor(point.Y + 0.5));
						unsigned short depth = depthBufferMat.at<unsigned short>(y, x);
						if ((colorX >= 0) && (colorX < colorWidth) && (colorY >= 0) && (colorY < colorHeight)/* && ( depth >= minDepth ) && ( depth <= maxDepth )*/) {
							coordinateMapperMat.at<cv::Vec4b>(y, x) = colorBufferMat.at<cv::Vec4b>(colorY, colorX);
						}
					}
				}
			}
		}

		if (SUCCEEDED(hResult))
		{

			//Particle Filter Start from here
			frame = &(IplImage)coordinateMapperMat;//transorm mat into IplImage

			if (image == 0)
			{
				// initialize variables and allocate buffers 
				image = cvCreateImage(cvGetSize(frame), 8, 4);//every pixel has 4 channels
				image->origin = frame->origin;

				result = cvCreateImage(cvGetSize(frame), 8, 4);
				result->origin = frame->origin;

				hsv = cvCreateImage(cvGetSize(frame), 8, 3);

				hue = cvCreateImage(cvGetSize(frame), 8, 1);

				sat = cvCreateImage(cvGetSize(frame), 8, 1);

				histimg_ref = cvCreateImage(cvGetSize(frame), 8, 3);
				histimg_ref->origin = frame->origin;
				cvZero(histimg_ref);

				histimg = cvCreateImage(cvGetSize(frame), 8, 3);
				histimg->origin = frame->origin;
				cvZero(histimg);

				bin_w = histimg_ref->width / BIN;
				bin_h = histimg_ref->height / BIN;


				data1.sample_t = reinterpret_cast<Region *> (malloc(sizeof(Region)* SAMPLE));
				data1.sample_t_1 = reinterpret_cast<Region *> (malloc(sizeof(Region)* SAMPLE));
				data1.sample_weight = reinterpret_cast<double *> (malloc(sizeof(double)* SAMPLE));
				data1.accum_weight = reinterpret_cast<double *> (malloc(sizeof(double)* SAMPLE));

			}

			cvCopy(frame, image);
			cvCopy(frame, result);
			cvCvtColor(image, hsv, CV_BGR2HSV);//image ~ hsv


			if (tracking)
			{
				//v_max = 0.0;

				cvSplit(hsv, hue, 0, 0, 0);//hsv->hue
				cvSplit(hsv, 0, 0, sat, 0);//hsv-saturation

				if (selecting)
				{

					// get the selected target area
					//ref_v_max = 0.0;
					area.width = abs(P_org.x - P_end.x);
					area.height = abs(P_org.y - P_end.y);
					area.x = MIN(P_org.x, P_end.x);
					area.y = MIN(P_org.y, P_end.y);

					cvZero(histimg_ref);

					// build reference histogram
					cvSetImageROI(hue, area);
					cvSetImageROI(sat, area);

					// zero reference histogram
					for (i = 0; i < BIN; i++)
						for (j = 0; j < BIN; j++)
							hist_ref[i][j] = 0.0;

					// calculate reference histogram
					for (i = 0; i < area.height; i++)
					{
						for (j = 0; j < area.width; j++)
						{
							im_hue = cvGet2D(hue, i, j);
							im_sat = cvGet2D(sat, i, j);
							k = int(im_hue.val[0] / STEP_HUE);
							h = int(im_sat.val[0] / STEP_SAT);
							hist_ref[k][h] = hist_ref[k][h] + 1.0;
						}
					}


					// rescale the value of each bin in the reference histogram 
					// and show it as an image
					for (i = 0; i < BIN; i++)
					{
						for (j = 0; j < BIN; j++)
						{
							hist_ref[i][j] = hist_ref[i][j] / (area.height*area.width);
						}
					}

					cvResetImageROI(hue);
					cvResetImageROI(sat);

					// initialize tracking and samples
					track_win = area;
					Initdata(track_win);
					track_win_last = track_win;

					// set up flag of tracking
					selecting = 0;

				}

				// sample propagation and weighting
				track_win = ImProcess(hue, sat, hist_ref, track_win_last);
				FrameNumber++;

				track_win_last = track_win;
				cvZero(histimg);

				// draw the one RED bounding box
				cvRectangle(image, cvPoint(track_win.x, track_win.y), cvPoint(track_win.x + track_win.width, track_win.y + track_win.height), CV_RGB(255, 0, 0), 2);
				printf("width = %d, height = %d\n", track_win.width, track_win.height);

				//save certian images
				if (FrameNumber % 10 == 0)
				{
					if (FrameNumber / 10 == 1) cvSaveImage("./imageout1.jpg", image);

					if (FrameNumber / 10 == 2) cvSaveImage("./imageout2.jpg", image);

					if (FrameNumber / 10 == 3) cvSaveImage("./imageout3.jpg", image);

					if (FrameNumber / 10 == 4) cvSaveImage("./imageout4.jpg", image);

					if (FrameNumber / 10 == 5) cvSaveImage("./imageout5.jpg", image);

					if (FrameNumber / 10 == 6) cvSaveImage("./imageout6.jpg", image);

					if (FrameNumber / 10 == 7) cvSaveImage("./imageout7.jpg", image);

					if (FrameNumber / 10 == 8) cvSaveImage("./imageout8.jpg", image);
				}

				//save certian images
				if (FrameNumber % 10 == 0)
				{
					if (FrameNumber / 10 == 1) cvSaveImage("./resultout1.jpg", result);

					if (FrameNumber / 10 == 2) cvSaveImage("./resultout2.jpg", result);

					if (FrameNumber / 10 == 3) cvSaveImage("./resultout3.jpg", result);

					if (FrameNumber / 10 == 4) cvSaveImage("./resultout4.jpg", result);

					if (FrameNumber / 10 == 5) cvSaveImage("./resultout5.jpg", result);

					if (FrameNumber / 10 == 6) cvSaveImage("./resultout6.jpg", result);

					if (FrameNumber / 10 == 7) cvSaveImage("./resultout7.jpg", result);

					if (FrameNumber / 10 == 8) cvSaveImage("./resultout8.jpg", result);
				}

				//draw a same bounding box in DepthImage
				rectangle(DepthImage, track_win, CV_RGB(255, 0, 0), 2);

				//******************************************************Geodesic Distance***************************************************************************************
					//Point propagation and weight
					if (PointTrack == 1)
					{

						if (PointSelect == 1)//only visit once
						{

							// initialize tracking and samples
							for (int i = 0; i < SAMPLE; i++)
							{
								point[i].x_1 = P_track.x;
								point[i].y_1 = P_track.y;
								point[i].z_1 = depthData[P_track.x + P_track.y * 512];
								point[i].x_1_prime = 0.0;
								point[i].y_1_prime = 0.0;
							}
							
							refeFlag = 1;

							p_win = P_track;

							//p_transtart is the start point of the surface mesh
							P_transtart.x = track_win.x;
							P_transtart.y = track_win.y;

							PointSelect = 0;

						}

						//construct the graph(mesh)
						ConstructMesh(depthData, adjlist, P_transtart,track_win.width,track_win.height);

						//calculate shortest path
						vector<int> vertexDist;
						vertexDist.resize(track_win.width*track_win.height);

						ShortestPath(P_extre, adjlist,  vertexDist);

						cvCircle(image, P_extre, 3, CV_RGB(0, 255, 0),1);

						//generate the refernce distance for comparing
						if (refeFlag > 0)
						{
							cvCircle(image, p_win, 3, CV_RGB(0, 0, 255), 1);

							int track = abs(P_transtart.x - P_track.x) + track_win.width * abs(P_transtart.y - P_track.y);

							referDistance = vertexDist[track];

							refeFlag = 0;
						}
						
						//samples propagation
						PredictPoint(p_win);

						//get geodesic distance for each sample. 
						//find the sample which have most similar distance to the refernce distance
						float Dist, AbsDist, WinDist, minAbsDist = 10000;
						int number,sum=0,count=0;

						for (int i = 0; i < SAMPLE; i++)
						{
							int t = abs(P_transtart.x - point[i].x) + track_win.width * abs(P_transtart.y - point[i].y);
							if (adjlist[t].v == false) { count++; continue; }

							int refer = abs(point[i].x - P_transtart.x) + track_win.width * abs(point[i].y - P_transtart.y);
							Dist = vertexDist[refer];

							AbsDist = fabs(referDistance - Dist);

							//point[i].SampleWeight = AbsDist;
							//point[i].AccumWeight = sum;
							//sum = sum + AbsDist;

							if (AbsDist < minAbsDist) { AbsDist = Dist; number = i; WinDist = Dist; }

						}

						//for (int i = 0; i < SAMPLE; i++)
						//{
						//	point[i].SampleWeight = point[i].SampleWeight / sum;
						//	point[i].AccumWeight = point[i].AccumWeight / sum;
						//}

						printf("referDist = %f, winDist = %f, discardPoints = %d\n", referDistance, WinDist,count);
						
						p_win_last = p_win;
						p_win.x = point[number].x;
						p_win.y = point[number].y;

						//samples re-location
						float deltaX = p_win.x - p_win_last.x;
						float deltaY = p_win.y - p_win_last.y;

						UpdatePoint(number, deltaX, deltaY);

						cvCircle(image, p_win, 5, CV_RGB(0, 0, 0));
					}



				//	//****************************************************************************************************************************************
			}

			// if still selecting a target, show the RED selected area
			else cvRectangle(image, P_org, P_end, CV_RGB(255, 0, 0), 1);

		}

		imshow("Depth", DepthImage);
		cvShowImage("Color Probabilistic Tracking - Samples", image);
		cvShowImage("Color Probabilistic Tracking - Result", result);

		SafeRelease(pColorFrame);
		SafeRelease(pDepthFrame);
		//SafeRelease(pBodyIndexFrame);

		cv::imshow("Color", colorMat);
		cv::imshow("Depth", DepthImage);
		cv::imshow("CoordinateMapper", coordinateMapperMat);

		//END OF THE TIME POINT
		t = ((double)getTickCount() - t) / getTickFrequency();
		t = 1 / t;

		//cout << "FPS:" << t << "FrameNumber\n" << FrameNumebr<< endl;
		printf("FPS:%f Frame:%d \n\n", t, FrameNumber);


		if (cv::waitKey(30) == VK_ESCAPE) {
			break;
		}
	}

	SafeRelease(pColorSource);
	SafeRelease(pDepthSource);
	//SafeRelease(pBodyIndexSource);

	SafeRelease(pColorReader);
	SafeRelease(pDepthReader);
	//SafeRelease(pBodyIndexReader);

	SafeRelease(pColorDescription);
	SafeRelease(pDepthDescription);
	SafeRelease(pCoordinateMapper);
	if (pSensor) {
		pSensor->Close();
	}
	SafeRelease(pSensor);
	cv::destroyAllWindows();


	cvReleaseImage(&image);
	cvReleaseImage(&result);
	cvReleaseImage(&histimg_ref);
	cvReleaseImage(&histimg);
	cvReleaseImage(&hsv);
	cvReleaseImage(&hue);
	cvReleaseImage(&sat);
	cvDestroyWindow("Color Probabilistic Tracking - Samples");
	cvDestroyWindow("Color Probabilistic Tracking - Result");

	return 0;
}

