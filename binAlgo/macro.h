#ifndef MACRO_H
#define MACRO_H 1

#include <opencv2/features2d/features2d.hpp>
#include <vector>
/*
#define CV_GetReal2D(i,k,l) ((float)((i)->imageData[(k)*(i)->widthStep+(l)*(i)->nChannels]))
#define CV_SetReal2D(i,k,l,v) (i)->imageData[(k)*(i)->widthStep+(l)*(i)->nChannels]=(float)v
#define CV_GetChar2D(i,k,l) ((uchar)((i)->imageData[(k)*(i)->widthStep+(l)*(i)->nChannels]))
#define CV_SetChar2D(i,k,l,v) (i)->imageData[(k)*(i)->widthStep+(l)*(i)->nChannels]=(unsigned char)v
#define CV_GetInt2D(img,i,j) (((int *)((img)->imageData))[(i)*(img)->widthStep+(j)*(img)->nChannels])
#define CV_SetInt2D(i,k,l,v) (i)->imageData[(k)*(i)->widthStep+(l)*(i)->nChannels]=(int)v
#define CV_GetReal3DMat(means,k,l,m) ((uchar *)((means)->imageData + k*(means)->widthStep))[l*(means)->nChannels + m] //cvGetReal2D(mat,k,l)
#define CV_SetReal3DMat(means,k,l,m,v) ((uchar *)((means)->imageData + k*(means)->widthStep))[l*(means)->nChannels + m]=v //cvGetReal2D(mat,k,l)
#define CV_GetReal2DMat(mat,k,l) ((float)((mat)->data.fl[(k)*(mat)->width+(l)]))
#define CV_SetReal2DMat(mat,k,l,v) (mat)->data.fl[(k)*(mat)->width+(l)]=(float)v
*/

#define CV_GetReal2D(i,k,l) ((unsigned char)((i)->imageData[(k)*(i)->widthStep+(l)*(i)->nChannels]))
#define CV_SetReal2D(i,k,l,v) (i)->imageData[(k)*(i)->widthStep+(l)*(i)->nChannels]=(unsigned char)v
#define CV_GetInt2D(img,i,j) (((int *)((img)->imageData))[(i)*(img)->widthStep+(j)*(img)->nChannels])
#define CV_SetInt2D(i,k,l,v) (i)->imageData[(k)*(i)->widthStep+(l)*(i)->nChannels]=(int)v
#define CV_GetReal3DMat(means,k,l,m) ((uchar *)((means)->imageData + k*(means)->widthStep))[l*(means)->nChannels + m] //cvGetReal2D(mat,k,l)
#define CV_SetReal3DMat(means,k,l,m,v) ((uchar *)((means)->imageData + k*(means)->widthStep))[l*(means)->nChannels + m]=v //cvGetReal2D(mat,k,l)
#define CV_GetReal2DMatI(mat,k,l) ((unsigned int)((mat)->data.i[(k)*(mat)->width+(l)])) 
#define CV_SetReal2DMatI(mat,k,l,v) (mat)->data.i[(k)*(mat)->width+(l)]=(unsigned int)v 
#define CV_GetReal2DMatF(mat,k,l) ((float)((mat)->data.i[(k)*(mat)->width+(l)])) 
#define CV_SetReal2DMatF(mat,k,l,v) (mat)->data.i[(k)*(mat)->width+(l)]=(float)v 
#define CV_GetReal2DMatD(mat,k,l) ((double)((mat)->data.db[(k)+(l)])) 
#define CV_SetReal2DMatD(mat,k,l,v) (mat)->data.db[(k)+(l)]=(double)v 



#define CV_GetChar2D(i,k,l) CV_GetReal2D(i,k,l)
#define CV_SetChar2D(i,k,l,v) CV_SetReal2D(i,k,l,v)

using namespace std;


typedef struct StatisticCC{
	int posx,posy,pos1x,pos1y;
	double centerX,centerY;
	int nbPixels;
	double ecart;
	char ccClass;
	vector<int*> pixels;
	StatisticCC(){centerX=0,centerY=0;posx=99999;posy=99999;pos1x=0;pos1y=0;nbPixels=0;ccClass=0;ecart=0;};
}StatisticCC;

class PointsSURF{
public:
	CvMemStorage* storage;
	CvSeq* imageKeypoints;
	CvSeq* imageDescriptors;
	int nbPoints;

	PointsSURF(){storage=NULL;}
	~PointsSURF(){cvReleaseMemStorage(&storage);}
	PointsSURF(vector<StatisticCC>);
	PointsSURF(int);
};


class PointsSURF1{
public:
	cv::Mat imageDescriptors;
	vector<cv::KeyPoint> imageKeypoints;
};



#endif