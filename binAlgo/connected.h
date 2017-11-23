#ifndef _CONNECTED_H
#define _CONNECTED_H


#pragma warning(disable:4251)
#include <vector>
#include <algorithm>
#include <Windows.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>

#include "OpImg.h"
#include "macro.h"
//#include "SplitMerge.h"

using namespace std;

class OpImg;

#define SAME(i,j) ((i)==(j))
#define is_root_label(id) (labels[id]==id)


#define UPDATE_STATISTICS(j,i,lab) {			\
	if(out[lab].posx>i) out[lab].posx=i;		\
	if(out[lab].pos1x<=i) out[lab].pos1x=i+1;	\
	if(out[lab].posy>j)	out[lab].posy=j;		\
	if(out[lab].pos1y<=j) out[lab].pos1y=j+1;	\
	out[lab].ccClass++;						\
}

#define DISTANCE(a,b,x,y) {\
	distTmp=(unsigned int)sqrt(pow(a-x,2.0)+pow(b-y,2.0));	\
			if(dist>distTmp){								\
				dist=distTmp;								\
				number=iter;								\
			}												\
}


class ConnectedComponents
{
public:
  int width;
  double param1, param2;
  int nbPixelIncrease, nbIterFilter;


	ConnectedComponents(int soft_maxlabels) : labels(soft_maxlabels) {
    clear();
    param1 = 21;
    param2 = 8;
    width = 15;
    nbPixelIncrease = 29;
    nbIterFilter = 2;
	}
	void centerImg(IplImage *);
	void clear() {
		fill(labels.begin(), labels.end(), 0);
		highest_label = 1;
	}
	IplImage * connected( IplImage *img, CvMat *out=NULL,const IplImage *imgColor=NULL,bool filter=true);
	void connectedBis(IplImage *img, CvMat *out, CvMat *labelimgFinal);
	IplImage *removeNoise( IplImage *img, CvMat *labelimg);

	void filterCC(IplImage *,CvMat *);
	void filterCC(const IplImage *imgColor,const IplImage *imgBin, const IplImage *imgTern, CvMat *labelimg);
	void filterSmallDots( IplImage * finalImg, int sizeOfDots );
	void label_image(const IplImage *img, CvMat *out,unsigned char=128,bool=false);
	void relabel_image_simple(CvMat *);
	vector<StatisticCC> relabel_image_WithBoxes(CvMat *, IplImage *img);
	void relabel_image_WithColor(CvMat *labelimg, IplImage *img,unsigned char color);
	void label_image_diff(const IplImage *img, CvMat *out,unsigned char=128);
	CvMat * connectedComp(const IplImage *img, CvMat *out=NULL);
  void removeIsolatedBlackPixels( IplImage *img, CvMat *labelimg );
  void computeMeansCC(IplImage *testImg, CvMat *labels, unsigned char color, double threshold=0.5);

	vector<unsigned int> labels;
private:

	double* projVerticale,* projHorizontale;

	void finalise(vector<StatisticCC> &,IplImage *);
	double widthMean,heightMean;
	double widthVar,heightVar;
	double widthMeanN,heightMeanN;
	double densite;
	int searchCloseCCText(vector<StatisticCC> ,StatisticCC );
	int hasCloseC(vector<StatisticCC> ,int );

	vector<StatisticCC> computeStatisticsCC(const IplImage *,CvMat *);
	int classFinale(vector<StatisticCC> blobs,int i);
	int findText(vector<StatisticCC> blobs,int i);

	inline int find_root(int id) {
		unsigned int root=id;
		while (labels[root]<root) {
			root = labels[root];
		}
		return root;
	}
	inline void setRoot(unsigned int i, unsigned int root){
		//Input: A node i of the tree.
		//Input: The root node of the tree of node i.
		//Make all nodes in the path of node i point to root.
		unsigned int j;
		while (labels[i]<i){
			j=labels[i];
			labels[i]=root;
			i=j;
		}
		labels[i]=root;
	}

	inline int find(int i){
		//Input: A node i of tree of node i.
		//Output: The root node of tree of node i.
		// Find the root of the tree of node i
		// and compress the path in the process.
		int root=find_root(i);
		setRoot(i,root);
		return root;
	};

	inline int merge(int i, int j) {
		if(i!=j){
			int root=find_root(i);
			int rootj=find_root(j);
			if(root>rootj){
				setRoot(i,rootj);
				return rootj;
			}else{
				setRoot(j,root);
				return root;
			}
		}else
			return find_root(i);
	}

	inline unsigned int new_label() {
		if(highest_label+1 > labels.capacity())
			labels.reserve(highest_label*2);
		labels.resize(highest_label+1);
		labels[highest_label] = highest_label;
		return highest_label++;
	}

	inline unsigned int new_label_Statistic(vector<StatisticCC> stats,int x,int y,bool couleur){
		//first same function :
		if(highest_label+1 > labels.capacity()){
			labels.reserve(highest_label*2);
		}
		labels.resize(highest_label+1);
		labels[highest_label] = highest_label;
		//now save statistics :
		if(highest_label+1 > stats.capacity()){
			stats.reserve(highest_label*2);
		}
		stats.resize(highest_label+1);
		stats[highest_label].posx=x;
		stats[highest_label].posy=y;
		return highest_label++;
	}


	IplImage * relabel_image(const IplImage *, CvMat *out, bool updateTernaryImg=false);
	void relabel_imageProba(IplImage *img, CvMat *labelimg);
	void relabel_image_Noise(const IplImage *, CvMat *out,IplImage *img);

	void projection(const IplImage *img,const vector<StatisticCC> stats);
	float computeColorMean(const IplImage* imgSrc,CvMat* labels,int width,int x,int y,int idLabel);
	void seedsExtands(const IplImage *imgTern,const IplImage *imgBin,const IplImage *imgColor);
	unsigned int highest_label;
	unsigned int real_highest_label;
};



#endif // _CONNECTED_H
