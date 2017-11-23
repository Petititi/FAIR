#include "connected.h"


int computeColorVector(const IplImage* imgSrc,const IplImage* labelimg,int width,int x,int y,float* values)
{
	int nb=0;
	int nbBack=0;
	int nbForground=0;
	unsigned char* ligneLabel;
	unsigned char* ligneBin;
	int xDeb=x-width/2;
	int yDeb=y-width/2;
	if(xDeb<0) xDeb=0;
	if(yDeb<0) yDeb=0;
	if(xDeb+width>imgSrc->width) xDeb=imgSrc->width-width-1;
	if(yDeb+width>imgSrc->height) yDeb=imgSrc->height-width-1;
	int xFin=xDeb+width;
	int yFin=yDeb+width;

	//count the number of black (resp white) pixel on each connected component border
	for(int i=yDeb;i<yFin;i++){
		ligneLabel = (unsigned char*)(labelimg->imageData + labelimg->widthStep*i);
		ligneBin = (unsigned char*)(imgSrc->imageData+imgSrc->widthStep*i);
		for(int j=xDeb;j<xFin;j++){
			if(ligneLabel[j]==32){
				values[nb]=ligneBin[j];
				nb++;
				nbForground++;
			}
			if(ligneLabel[j]==80){
				values[nb]=ligneBin[j];
				nb++;
				nbBack++;
			}
		}
	}
	return nb;
}

void recomputeEdges( const IplImage * greyLevels, const IplImage * imgTern, int width /* 16 */, double param1 /* 16 */, double param2 /* 4 */ ){
	unsigned char* ligneBin;
	unsigned char* ligneImg;
	float* values=new float[width*width];

	for(int i=greyLevels->height-1;i>=0;i--){
		ligneImg = (unsigned char*)(greyLevels->imageData+greyLevels->widthStep*i);
		ligneBin = (unsigned char*)(imgTern->imageData+imgTern->widthStep*i);
		for(int j=0;j<greyLevels->width-1;j++){
			if(ligneBin[j]==32){
				//we should recompute the two means around this pixel using Kmean.
				int nbVal=computeColorVector(greyLevels,imgTern,width,j,i,values);
				/*if(nbVal<20){
					ligneBin[j]=0;//OK, verified with extensive experiments
				}
        else*/
        {
					float mean[2];
					mean[0]=0;
          mean[1]=0;
          OpImg::segmenteKMean(values,nbVal,mean);


					//then set the correct value (black or white) for this pixel

					//mean[CLASS_BACKGROUND] is for the background, so it should be higher than the mean of text...
					if(mean[CLASS_BACKGROUND]<mean[CLASS_TEXT]){
						double echange=mean[CLASS_BACKGROUND];
						mean[CLASS_BACKGROUND]=mean[CLASS_TEXT];
						mean[CLASS_TEXT]=echange;
					}
					//if the means have correct values :
					if((mean[CLASS_BACKGROUND]>mean[CLASS_TEXT]+param1)&&mean[CLASS_BACKGROUND]<253.0&&mean[CLASS_TEXT]>1.0){

						if(ligneImg[j]-mean[CLASS_TEXT]<mean[CLASS_BACKGROUND]-ligneImg[j])
							ligneBin[j]=0;//so it's text...
						else
							ligneBin[j]=254;

						//ligneBin[j]=128+ligneImg[j]-mean[CLASS_TEXT];
					}else{
            ligneBin[j]=128;//so it's text... Indeed, we only refine the previous segmentation, so we trust the previous step...
					}
				}
			}
		}
	}

	delete [] values;
}

void ConnectedComponents::seedsExtands(const IplImage *imgTern,const IplImage *imgBin,const IplImage *imgColor){

	CvMat* labelimg = cvCreateMat(imgTern->height,imgTern->width,CV_32SC1);

	//first recompute the CC (white and black):
	label_image(imgTern,labelimg,0,true);
	unsigned int* ligneLabel;
	unsigned char* ligneBin;
	unsigned char* ligneBinPrev;
	unsigned char* ligneBinNext;
	unsigned char* ligneImg;
	unsigned char* ligneImgPrev;
	unsigned char* ligneImgNext;
	vector<StatisticCC> labs=computeStatisticsCC(imgTern,labelimg);

	unsigned int root;

	//relabel dangerous pixels:
	for(int i=labelimg->height-2;i>=1;i--){
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		//ligneBin = (unsigned char*)(imgBin->imageData+imgBin->widthStep*i);
		//ligneBinPrev = (unsigned char*)(imgBin->imageData+imgBin->widthStep*(i-1));
		//ligneBinNext = (unsigned char*)(imgBin->imageData+imgBin->widthStep*(i+1));
		ligneImg = (unsigned char*)(imgTern->imageData+imgTern->widthStep*i);
		ligneImgPrev = (unsigned char*)(imgTern->imageData+imgTern->widthStep*(i-1));
		ligneImgNext = (unsigned char*)(imgTern->imageData+imgTern->widthStep*(i+1));
		for(int j=1;j<labelimg->width-2;j++){
			if(ligneImg[j]==0){//&&ligneBin[j]==0
				root=ligneLabel[j];
				if(root>0){

					if(ligneImgPrev[j]==128)//&&ligneBinPrev[j]>128)
          {
						ligneImgPrev[j]=64;
						//ligneBin[j]=128;
					}
					if(ligneImg[j-1]==128)//&&ligneBin[j-1]>128)
          {
						ligneImg[j-1]=64;
						//ligneBin[j]=128;
					}
					if(ligneImgNext[j]==128)//&&ligneBinNext[j]>128)
          {
						ligneImgNext[j]=64;
						//ligneBin[j]=128;
					}
					if(ligneImg[j+1]==128)//&&ligneBin[j+1]>128)
          {
						ligneImg[j+1]=64;
						//ligneBin[j]=128;
					}

				}
			}
		}
	}
	int varAltern=16;
	//seed expension:
	//relabel dangerous pixels:
	for(int cptAlt=0;cptAlt<nbPixelIncrease;cptAlt++){
		for(int i=labelimg->height-2;i>=1;i--){
			ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
			ligneImg = (unsigned char*)(imgTern->imageData+imgTern->widthStep*i);
			ligneImgPrev = (unsigned char*)(imgTern->imageData+imgTern->widthStep*(i-1));
			ligneImgNext = (unsigned char*)(imgTern->imageData+imgTern->widthStep*(i+1));
			for(int j=1;j<labelimg->width-2;j++){
				if(ligneImg[j]==128){//extend...
					if(ligneImgPrev[j]==80-varAltern){
						ligneImg[j]=80+varAltern;
					}
					if(ligneImg[j-1]==80-varAltern){
						ligneImg[j]=80+varAltern;
					}
					if(ligneImgNext[j]==80-varAltern){
						ligneImg[j]=80+varAltern;
					}
					if(ligneImg[j+1]==80-varAltern){
						ligneImg[j]=80+varAltern;
					}
				}
			}
		}
		varAltern=varAltern*-1;
	}
	//set the same color:
	for(int i=labelimg->height-1;i>=0;i--){
		ligneImg = (unsigned char*)(imgTern->imageData+imgTern->widthStep*i);
		for(int j=0;j<labelimg->width-1;j++){
			if(ligneImg[j]==80-varAltern||ligneImg[j]==80+varAltern){
				ligneImg[j]=80;
			}
		}
	}

	varAltern=16;
	//now grow into black regions:
	for(int cptAlt=0;cptAlt<5;cptAlt++){
		for(int i=labelimg->height-2;i>=1;i--){
			ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
			ligneImg = (unsigned char*)(imgTern->imageData+imgTern->widthStep*i);
			ligneImgPrev = (unsigned char*)(imgTern->imageData+imgTern->widthStep*(i-1));
			ligneImgNext = (unsigned char*)(imgTern->imageData+imgTern->widthStep*(i+1));
			for(int j=1;j<labelimg->width-2;j++){
				if(ligneImg[j]==0){//extend...
					if(ligneImgPrev[j]==80||ligneImgPrev[j]==128-varAltern){
						ligneImg[j]=128+varAltern;
					}
					if(ligneImg[j-1]==80||ligneImg[j-1]==128-varAltern){
						ligneImg[j]=128+varAltern;
					}
					if(ligneImgNext[j]==80||ligneImgNext[j]==128-varAltern){
						ligneImg[j]=128+varAltern;
					}
					if(ligneImg[j+1]==80||ligneImg[j+1]==128-varAltern){
						ligneImg[j]=128+varAltern;
					}
				}
			}
		}
		varAltern=varAltern*-1;
	}
	//set the same color:
	for(int i=labelimg->height-1;i>=0;i--){
		ligneImg = (unsigned char*)(imgTern->imageData+imgTern->widthStep*i);
		for(int j=0;j<labelimg->width-1;j++){
			if(ligneImg[j]==128-varAltern||ligneImg[j]==128+varAltern){
				ligneImg[j]=32;
			}
		}
	}

	recomputeEdges(imgColor,imgTern, width, param1, param2);

	//set the same color:
	for(int i=labelimg->height-1;i>=0;i--){
		ligneImg = (unsigned char*)(imgTern->imageData+imgTern->widthStep*i);
		for(int j=0;j<labelimg->width-1;j++){
			if(ligneImg[j]>0&&ligneImg[j]<200){
				ligneImg[j]=128;
			}
		}
  }

	cvReleaseMat(&labelimg);

}

void ConnectedComponents::filterSmallDots( IplImage * finalImg, int sizeOfDots )
{
	CvMat *labelimg = cvCreateMat(finalImg->height,finalImg->width,CV_32SC1);

	label_image(finalImg,labelimg,0,true);
	vector<StatisticCC> statistics=computeStatisticsCC(finalImg,labelimg);
	double *nbVal=new double[real_highest_label];
	double *nbBackground=new double[real_highest_label];
	double *nbWhite=new double[real_highest_label];
	for(unsigned int i=0;i<real_highest_label;i++){
		nbVal[i]=0;
		nbBackground[i]=0;
		nbWhite[i]=0;
	}
	unsigned int root;
	int nbBlackDots=sizeOfDots*sizeOfDots/2;
	for(int i=1;i<labelimg->height-2;i++){
		unsigned int* ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		unsigned char*ligneImg = (unsigned char*)(finalImg->imageData+finalImg->widthStep*i);
		for(int j=1;j<labelimg->width-2;j++){
			if(ligneImg[j]==0){
				root=ligneLabel[j];
				if(statistics[root].nbPixels<nbBlackDots)//||(statistics[root].nbPixels<2*nbBlackDots&&(abs(statistics[root].pos1x-statistics[root].posx)<sizeOfDots||abs(statistics[root].pos1y-statistics[root].posy)<sizeOfDots))
					ligneImg[j]=255;
			}
		}
	}

	cvReleaseMat(&labelimg);
}

IplImage * ConnectedComponents::removeNoise( IplImage *img, CvMat *labelimg)
{
	bool mustDelete=false;
	if(labelimg==NULL){
		labelimg = cvCreateMat(img->height,img->width,CV_32SC1);
		mustDelete=true;
	}
	IplImage * finalImg = this->connected(img,labelimg);
	
	label_image_diff(img,labelimg);
	relabel_image_Noise(img,labelimg,finalImg);
	
	if(mustDelete)
		cvReleaseMat(&labelimg);
	return finalImg;
}

void ConnectedComponents::connectedBis(IplImage *img, CvMat *labelimg, CvMat *labelimgFinal)
{
	bool mustDelete=false;
	if(labelimg==NULL){
		labelimg = cvCreateMat(img->height,img->width,CV_32SC1);
		mustDelete=true;
	}
	label_image(img,labelimg);
	relabel_imageProba(img,labelimg);
	if(mustDelete)
		cvReleaseMat(&labelimg);
	//filterCC(out,labelimg);
	
    unsigned int* ligneLabel = (unsigned int*)(labelimg->data.i);
	double* ligneImg = (double*)(labelimgFinal->data.db);
	for(int i=0;i<labelimg->height;i++){
		unsigned char* ligneLabel = (unsigned char*)(img->imageData + img->widthStep*i);
		double* ligneLabelOut = (double*)(labelimgFinal->data.db + labelimgFinal->width*i);
		for(int j=0;j<labelimg->width;j++){
			ligneLabelOut[j]=ligneLabel[j]-128;
		}
	}
}

void ConnectedComponents::removeIsolatedBlackPixels( IplImage *img, CvMat *labelimg )
{
  label_image(img,labelimg, 0);

  unsigned int* ligneLabel;
  unsigned int* ligneLabelEnd;
  unsigned char* ligneImg;
  unsigned char* ligneImgPrev;
  unsigned char* ligneImgNext;
  unsigned int root;

  double *nbVal;
  double *nbBlack;
  double *nbWhite;

  real_highest_label = 1;
  for(unsigned int id=1; id<labels.size(); id++){
    if(labels[id]==id){
      labels[id] = real_highest_label;
      real_highest_label++;
    }else{
      labels[id] = labels[labels[id]];
    }
  }
  nbVal=new double[real_highest_label];
  nbBlack=new double[real_highest_label];
  nbWhite=new double[real_highest_label];
  for(unsigned int i=0;i<real_highest_label;i++){
    nbVal[i]=0;
    nbBlack[i]=0;
    nbWhite[i]=0;
  }
  ligneLabel = (unsigned int*)(labelimg->data.i);
  ligneLabelEnd = (unsigned int*)(labelimg->data.i + labelimg->width*(labelimg->height-1));
  for(int i=0;i<labelimg->width;i++){
    ligneLabel[i]=labels[ligneLabel[i]];
    ligneLabelEnd[i]=labels[ligneLabelEnd[i]];
  }
  //count the number of black (resp white) pixel on each connected component border
  for(int i=1;i<labelimg->height-1;i++){
    ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
    ligneImg = (unsigned char*)(img->imageData+img->widthStep*i);
    ligneImgPrev = (unsigned char*)(img->imageData+img->widthStep*(i-1));
    ligneImgNext = (unsigned char*)(img->imageData+img->widthStep*(i+1));
    for(int j=1;j<labelimg->width-1;j++){
      if(ligneImg[j]==0){
        root=labels[ligneLabel[j]];
        //update the value of the label :
        nbVal[root]++;
        ligneLabel[j]=root;
        if(ligneImgPrev[j]<=50){
          nbBlack[root]++;
        }else{
          if(ligneImgPrev[j]>=200){
            nbWhite[root]++;
          }
        }
        if(ligneImg[j-1]<=50){
          nbBlack[root]++;
        }else{
          if(ligneImg[j-1]>=200){
            nbWhite[root]++;
          }
        }
        if(ligneImgNext[j]<=50){
          nbBlack[root]++;
        }else{
          if(ligneImgNext[j]>=200){
            nbWhite[root]++;
          }
        }
        if(ligneImg[j+1]<=50){
          nbBlack[root]++;
        }else{
          if(ligneImg[j+1]>=200){
            nbWhite[root]++;
          }
        }
      }
    }
  }

  for(int i=1;i<labelimg->height-1;i++){
    ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
    ligneImg = (unsigned char*)(img->imageData+img->widthStep*i);
    for(int j=1;j<labelimg->width-1;j++){
      if(ligneImg[j]==0)
      {
        root=ligneLabel[j];
        if(nbWhite[root]<=1)
        {
          ligneImg[j]=128;
        }
      }
    }
  }
  delete nbWhite;
  delete nbBlack;
  delete nbVal;
}

IplImage * ConnectedComponents::connected( IplImage *img, CvMat *labelimg/*=NULL*/,const IplImage *imgColor/*=NULL*/,bool filter/*=false*/ )
{
	bool mustDelete=false;
	if(labelimg==NULL)
  {
		labelimg = cvCreateMat(img->height,img->width,CV_32SC1);
		mustDelete=true;
	}
  
  removeIsolatedBlackPixels( img, labelimg );

	label_image(img,labelimg);
	
	IplImage *out=relabel_image(img, labelimg, true);
	//filterCC(out,labelimg);
	
  if(filter)
  {

    filterCC(imgColor,out,img,labelimg);
		for(int nbIter=0;nbIter<=nbIterFilter;nbIter++){
      removeIsolatedBlackPixels( img, labelimg );
      /*
#ifdef _DEBUG
  cvSaveImage("imgBinFinal.bmp",out);
  cvSaveImage("imgTernFinal.bmp",img);
  cvShowImage("imgTernFinal",img);
  cvWaitKey(0);
#endif*/

			cvReleaseImage(&out);
			label_image(img,labelimg);
      out=relabel_image(img,labelimg);

			filterCC(imgColor,out,img,labelimg);
		}
	}
	if(mustDelete)
		cvReleaseMat(&labelimg);
	return out;
}

void ConnectedComponents::computeMeansCC(IplImage *testImg, CvMat *labelimg, unsigned char color, double threshold)
{

  relabel_image_simple( labelimg );
  float *meansValue = new float[real_highest_label];
  float *nbVals = new float[real_highest_label];
  for(int i=0;i<real_highest_label;i++){
    nbVals[i] = 0;
    meansValue[i] = 0;
  }
  for(int i=0;i<labelimg->height;i++){
    unsigned int* ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
    unsigned char* ligneImg = (unsigned char*)(testImg->imageData + testImg->widthStep*i);
    for(int j=0;j<labelimg->width;j++){
      if( ligneLabel[j]!=0 )
      {
        unsigned int tmp=ligneLabel[j];
        nbVals[tmp]++;
        meansValue[tmp]+=ligneImg[j];
      }
    }
  }
  for(int i=0;i<real_highest_label;i++){
    if(nbVals[i]>5)
    {
      meansValue[i] /= nbVals[i];
    }
    else
      meansValue[i] = 0;
  }

  float centers[2];
  centers[0] = centers[1] = 0;
  float centersSize[2];
  centersSize[0] = centersSize[1] = 0;
  OpImg::segmenteKMean( meansValue, real_highest_label, centers );
  //OpImg::segmenteKMean( nbVals, real_highest_label, centersSize );
  //cout<<centers[0]<<" "<<centers[1]<<endl;
  double highThreshold = (centers[0]+centers[1])/this->param1;
  double lowThreshold = highThreshold/this->param2;

  for(int i=0;i<labelimg->height;i++)
  {
    unsigned int* ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
    unsigned char* ligneImg = (unsigned char*)(testImg->imageData + testImg->widthStep*i);
    for(int j=0;j<labelimg->width;j++){
      if( ligneLabel[j]!=0 )
      {
        unsigned int tmp=ligneLabel[j];
        if( meansValue[tmp] > highThreshold )
          ligneImg[j] = 255;
        else
        {
          if( meansValue[tmp] > lowThreshold )
          {
            ligneImg[j] = 128;
          }
          else
          {
            ligneImg[j] = 0;
          }
        }
      }
      else
        ligneImg[j] = 0;
    }
  }
  delete [] meansValue;
  delete [] nbVals;
}

/*
void ConnectedComponents::centerImg(IplImage *img){
	SplitMerge sm;
	Rect *region=new Rect;
	region->x=0;
	region->y=0;
	region->x1=img->width;
	region->y1=img->height;
	vector<Rect*> regionsTMPV=sm.splitV(img,region);
	vector<Rect*> regionsTMPH;
	vector<Rect*> regions;
	
	while(regionsTMPV.size()>0||regionsTMPH.size()>0){
		while(regionsTMPV.size()>0){
			Rect *regionAnalyse=regionsTMPV.back();
			regionsTMPV.pop_back();
			vector<Rect*> regionsTMP1;
			regionsTMP1=sm.splitH(img,regionAnalyse);

			if(regionsTMP1.size()==0){//region atomic :
				regions.push_back(regionAnalyse);
			}else{
				delete regionAnalyse;
				while(regionsTMP1.size()>0){
					regionsTMPH.push_back(regionsTMP1.back());
					regionsTMP1.pop_back();
				}
			}
		}


		while(regionsTMPH.size()>0){
			Rect *regionAnalyse=regionsTMPH.back();
			regionsTMPH.pop_back();
			vector<Rect*> regionsTMP1;
			regionsTMP1=sm.splitV(img,regionAnalyse);

			if(regionsTMP1.size()==0){//region atomic :
				regions.push_back(regionAnalyse);
			}else{
				delete regionAnalyse;
				while(regionsTMP1.size()>0){
					regionsTMPV.push_back(regionsTMP1.back());
					regionsTMP1.pop_back();
				}
			}
		}
	}
	
	int *nbText=new int[img->height];
	int *nbTextH=new int[img->width];
	for(int i=0;i<img->height;i++){
		nbText[i]=0;
	}
	for(int i=0;i<img->width;i++){
		nbTextH[i]=0;
	}
	unsigned char* ligneImg=NULL;
	int cpt=0;
	while(regions.size()>0){
		cpt++;
		Rect *rec=regions.back();
		regions.pop_back();

		if(sm.analyseRegion(img,rec)){//it's text...
			for(int tt=rec->x;tt<rec->x1;tt++)
				nbTextH[tt]++;
			for(int tt=rec->y;tt<rec->y1;tt++)
				nbText[tt]++;
		}
		delete rec;
	}

	//find the top and bottom :
	unsigned char position=0;
	for(int r=region->y; r<region->y1&&position==0; r++){
		ligneImg = (unsigned char*)(img->imageData+img->widthStep*r);
		if(nbText[r]<5&&position==0)
			for(int i=region->x;i<region->x1;i++){
				ligneImg[i]=254;
			}
		else{
			position++;
		}
	}
	for(int r=region->y1-1; r>region->y&&position==1; r--){
		ligneImg = (unsigned char*)(img->imageData+img->widthStep*r);
		if(nbText[r]<5)
			for(int i=region->x;i<region->x1;i++){
				ligneImg[i]=254;
			}
		else{
			position++;
			break;
		}
	}
	int maxNb=0,minNb=200;
	for(int i=0;i<img->width;i++){
		if(maxNb<nbTextH[i])
			maxNb=nbTextH[i];
		if(minNb>nbTextH[i])
			minNb=nbTextH[i];
	}
	
	//find the left and right :
	if(maxNb<40){//pas assez de valeurs, on suppose la simple colonne :
		for(int i=region->x;i<region->x1;i++){
			if(nbTextH[i]>maxNb/10){
				break;
			}else{
				for(int r=region->y; r<region->y1; r++){
					ligneImg = (unsigned char*)(img->imageData+img->widthStep*r);
					ligneImg[i]=254;
				}
			}
		}
		for(int i=region->x1-1;i>region->x;i--){
			if(nbTextH[i]>maxNb/10){
				break;
			}else{
				for(int r=region->y; r<region->y1; r++){
					ligneImg = (unsigned char*)(img->imageData+img->widthStep*r);
					ligneImg[i]=254;
				}
			}
		}
	}else{
		bool estDedans=false;
		for(int r=region->y; r<region->y1; r++){
			ligneImg = (unsigned char*)(img->imageData+img->widthStep*r);
			for(int i=region->x;i<region->x1;i++){
				if(!estDedans){//on est dans le vide...
					if(nbTextH[i]>maxNb/5)
						estDedans=true;
					else
						ligneImg[i]=254;
				}else{
					if(nbTextH[i]<maxNb/15)
						estDedans=false;
				}
			}
		}
	}
	delete region;
	delete [] nbText;
	delete [] nbTextH;
	
};
*/
CvMat * ConnectedComponents::connectedComp(const IplImage *img, CvMat *labelimg)
{
	if(labelimg==NULL){
		labelimg = cvCreateMat(img->height,img->width,CV_32SC1);
	}
	label_image(img,labelimg,0);
	relabel_image_simple(labelimg);

	return labelimg;
}

void ConnectedComponents::label_image_diff(const IplImage *img, CvMat *labelimg,unsigned char valBlob)
{
	clear();
	cvSet(labelimg,cvScalar(0));

    unsigned int* ligneLabel = (unsigned int*)(labelimg->data.i);
    unsigned int* ligneLabelPrev;
	unsigned char* ligneImg = (unsigned char*)(img->imageData);
	unsigned char* ligneImgPrev;

	if(CV_GetReal2D(img,0,0)!=valBlob)
		CV_SetReal2DMatI(labelimg,0,0,new_label());

	// label the first row :
	for(int c=1, r=0; c<img->width; c++) {
		int test=ligneImg[c];
		if(ligneImg[c]!=valBlob){
			if(ligneImg[c]==ligneImg[c-1])
				ligneLabel[c]=ligneLabel[c-1];
			else
				ligneLabel[c]=new_label();
		}
	}

	// label subsequent row.
	for(int r=1; r<img->height; r++){
		// label the first pixel on this row.
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*r);
		ligneLabelPrev = (unsigned int*)(labelimg->data.i + labelimg->width*(r-1));
		ligneImg = (unsigned char*)(img->imageData+img->widthStep*r);
		ligneImgPrev = (unsigned char*)(img->imageData+img->widthStep*(r-1));

    if(ligneImg[0]!=valBlob){
      if(ligneImg[0]==ligneImgPrev[0])
        ligneLabel[0]=ligneLabelPrev[0];
      else{
        if(ligneImg[0]==ligneImgPrev[1])
          ligneLabel[0]=ligneLabelPrev[1];
        else{
          ligneLabel[0]=new_label();
        }
      }
		}

		// label subsequent pixels on this row.
		for(int c=1; c<img->width; c++)	{

      if(ligneImg[c]!=valBlob){
        if(ligneImg[c-1]!=valBlob){
          if(ligneImgPrev[c-1]!=valBlob){
            ligneLabel[c]=merge(ligneLabel[c-1],ligneLabelPrev[c-1]);
            if(ligneImgPrev[c]!=valBlob){
              ligneLabel[c]=merge(ligneLabel[c-1],ligneLabelPrev[c]);
              if(ligneImgPrev[c+1]!=valBlob){
                ligneLabel[c]=merge(ligneLabel[c-1],ligneLabelPrev[c+1]);
              }
            }else{
              if(ligneImgPrev[c+1]!=valBlob){
                ligneLabel[c]=merge(ligneLabel[c-1],ligneLabelPrev[c+1]);
              }
            }
          }else{
            if(ligneImgPrev[c]!=valBlob){
              ligneLabel[c]=merge(ligneLabel[c-1],ligneLabelPrev[c]);
              if(ligneImgPrev[c+1]!=valBlob){
                ligneLabel[c]=merge(ligneLabel[c-1],ligneLabelPrev[c+1]);
              }
            }else{
              if(ligneImgPrev[c+1]!=valBlob){
                ligneLabel[c]=merge(ligneLabel[c-1],ligneLabelPrev[c+1]);
              }else{
                ligneLabel[c]=ligneLabel[c-1];
              }
            }
          }
        }else{
          if(ligneImgPrev[c-1]!=valBlob){
            if(ligneImgPrev[c]!=valBlob){
              ligneLabel[c]=merge(ligneLabelPrev[c-1],ligneLabelPrev[c]);
              if(ligneImgPrev[c+1]!=valBlob){
                ligneLabel[c]=merge(ligneLabelPrev[c-1],ligneLabelPrev[c+1]);
              }
            }else{
              if(ligneImgPrev[c+1]!=valBlob){
                ligneLabel[c]=merge(ligneLabelPrev[c-1],ligneLabelPrev[c+1]);
              }else{
                ligneLabel[c]=ligneLabelPrev[c-1];
              }
            }
          }else{
            if(ligneImgPrev[c]!=valBlob){
              if(ligneImgPrev[c+1]!=valBlob){
                ligneLabel[c]=merge(ligneLabelPrev[c],ligneLabelPrev[c+1]);
              }else{
                ligneLabel[c]=ligneLabelPrev[c];
              }
            }else{
              if(ligneImgPrev[c+1]!=valBlob){
                ligneLabel[c]=ligneLabelPrev[c+1];
              }else{
                ligneLabel[c]=new_label();
              }
            }
          }
        }
			}
		}
	}
}

void ConnectedComponents::label_image(const IplImage *img, CvMat *labelimg,unsigned char valBlob,bool connectivity8)
{
  clear();
  cvSet(labelimg,cvScalar(0));

  unsigned int* ligneLabel = (unsigned int*)(labelimg->data.i);
  unsigned int* ligneLabelPrev;
  unsigned char* ligneImg = (unsigned char*)(img->imageData);
  unsigned char* ligneImgPrev;

  if(CV_GetReal2D(img,0,0)==valBlob)
    CV_SetReal2DMatI(labelimg,0,0,new_label());

  // label the first row :
  for(int c=1, r=0; c<img->width; c++) {
    int test=ligneImg[c];
    if(ligneImg[c]==valBlob){
      if(ligneImg[c]==ligneImg[c-1])
        ligneLabel[c]=ligneLabel[c-1];
      else
        ligneLabel[c]=new_label();
    }
  }

	// label subsequent row.
	for(int r=1; r<img->height; r++){
		// label the first pixel on this row.
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*r);
		ligneLabelPrev = (unsigned int*)(labelimg->data.i + labelimg->width*(r-1));
		ligneImg = (unsigned char*)(img->imageData+img->widthStep*r);
		ligneImgPrev = (unsigned char*)(img->imageData+img->widthStep*(r-1));

		if(ligneImg[0]==valBlob){
			if(connectivity8){
				if(ligneImg[0]==ligneImgPrev[0])
					ligneLabel[0]=ligneLabelPrev[0];
				else{
					if(ligneImg[0]==ligneImgPrev[1])
						ligneLabel[0]=ligneLabelPrev[1];
					else{
						ligneLabel[0]=new_label();
					}
				}
			}else{
				if(ligneImg[0]==ligneImgPrev[0])
					ligneLabel[0]=ligneLabelPrev[0];
				else{
					ligneLabel[0]=new_label();
				}
			}
		}

		// label subsequent pixels on this row.
		for(int c=1; c<img->width; c++)	{

			if(ligneImg[c]==valBlob){
				if(connectivity8){
					if(ligneImg[c-1]==valBlob){
						if(ligneImgPrev[c-1]==valBlob){
							ligneLabel[c]=merge(ligneLabel[c-1],ligneLabelPrev[c-1]);
							if(ligneImgPrev[c]==valBlob){
								ligneLabel[c]=merge(ligneLabel[c-1],ligneLabelPrev[c]);
								if(ligneImgPrev[c+1]==valBlob){
									ligneLabel[c]=merge(ligneLabel[c-1],ligneLabelPrev[c+1]);
								}
							}else{
								if(ligneImgPrev[c+1]==valBlob){
									ligneLabel[c]=merge(ligneLabel[c-1],ligneLabelPrev[c+1]);
								}
							}
						}else{
							if(ligneImgPrev[c]==valBlob){
								ligneLabel[c]=merge(ligneLabel[c-1],ligneLabelPrev[c]);
								if(ligneImgPrev[c+1]==valBlob){
									ligneLabel[c]=merge(ligneLabel[c-1],ligneLabelPrev[c+1]);
								}
							}else{
								if(ligneImgPrev[c+1]==valBlob){
									ligneLabel[c]=merge(ligneLabel[c-1],ligneLabelPrev[c+1]);
								}else{
									ligneLabel[c]=ligneLabel[c-1];
								}
							}
						}
					}else{
						if(ligneImgPrev[c-1]==valBlob){
							if(ligneImgPrev[c]==valBlob){
								ligneLabel[c]=merge(ligneLabelPrev[c-1],ligneLabelPrev[c]);
								if(ligneImgPrev[c+1]==valBlob){
									ligneLabel[c]=merge(ligneLabelPrev[c-1],ligneLabelPrev[c+1]);
								}
							}else{
								if(ligneImgPrev[c+1]==valBlob){
									ligneLabel[c]=merge(ligneLabelPrev[c-1],ligneLabelPrev[c+1]);
								}else{
									ligneLabel[c]=ligneLabelPrev[c-1];
								}
							}
						}else{
							if(ligneImgPrev[c]==valBlob){
								if(ligneImgPrev[c+1]==valBlob){
									ligneLabel[c]=merge(ligneLabelPrev[c],ligneLabelPrev[c+1]);
								}else{
									ligneLabel[c]=ligneLabelPrev[c];
								}
							}else{
								if(ligneImgPrev[c+1]==valBlob){
									ligneLabel[c]=ligneLabelPrev[c+1];
								}else{
									ligneLabel[c]=new_label();
								}
							}
						}
					}
				}else{
					if(ligneImg[c-1]==valBlob){
						if(ligneImgPrev[c]==valBlob){
							ligneLabel[c]=merge(ligneLabel[c-1],ligneLabelPrev[c]);
						}else{
							ligneLabel[c]=ligneLabel[c-1];
						}
					}else{
						if(ligneImgPrev[c]==valBlob){
							ligneLabel[c]=ligneLabelPrev[c];
						}else{
							ligneLabel[c]=new_label();
						}
					}
				}
			}
		}
	}
}

vector<StatisticCC> ConnectedComponents::relabel_image_WithBoxes(CvMat *labelimg, IplImage *img){
	real_highest_label = 1;

	for(unsigned int id=1; id<labels.size(); id++){
		if(labels[id]==id){
			labels[id] = real_highest_label;
			real_highest_label++;
		}else{
			labels[id] = labels[labels[id]];
		}
	}

	vector<StatisticCC> out(real_highest_label);
	int *pos;
	
	unsigned short lab;
	StatisticCC* scc;
	for(int i=0;i<labelimg->height;i++){
		unsigned int* ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		for(int j=0;j<labelimg->width;j++){
			lab=labels[ligneLabel[j]];
			if(lab>0){
				ligneLabel[j]=254;
				scc=&out[lab];
				if(scc->posx>j) scc->posx=j;
				if(scc->pos1x<=j) scc->pos1x=j+1;
				if(scc->posy>i)	scc->posy=i;
				if(scc->pos1y<=i) scc->pos1y=i+1;
				scc->centerX+=j;
				scc->centerY+=i;
				pos=new int [2];
				pos[0]=j;
				pos[1]=i;
				scc->pixels.push_back(pos);
				scc->nbPixels++;
			}else
				ligneLabel[j]=0;
		}
	}
	double angle=0, nbPix=0;
	static CvFont* font=NULL;
	if(font==NULL){
		font=new CvFont;
		cvInitFont(font,CV_FONT_HERSHEY_SIMPLEX,0.5,0.5,0,2,8);
	}
	//cvSet(img,cvScalar(0));
	for(int i=1;i<real_highest_label;i++){
		scc=&out[i];
		scc->centerX/=scc->nbPixels;
		scc->centerY/=scc->nbPixels;
		angle=0;
		nbPix=scc->pixels.size();
		double weight=0;
		for(int j=0;j<nbPix;j++){
			double dist=abs(scc->centerX-scc->pixels[j][0])+abs(scc->centerY-scc->pixels[j][1]);
			double angleTmp=atan((scc->centerX-scc->pixels[j][0])/(scc->centerY-scc->pixels[j][1]));
			angle+=atan((scc->centerX-scc->pixels[j][0])/(scc->centerY-scc->pixels[j][1]))*dist;
			if(scc->nbPixels<20){
				CV_SetInt2D(img,scc->pixels[j][1],scc->pixels[j][0],0);
				((unsigned int*)(labelimg->data.i + labelimg->width*scc->pixels[j][1]))[scc->pixels[j][0]]=0;
			}
			delete scc->pixels[j];
			weight+=dist;
		}
		angle=angle/weight;
		if(scc->nbPixels>20){
			float sin_dir = sin(angle);
			float cos_dir = cos(angle);
			stringstream s;
			s<<180.0 * (angle) / CV_PI ;
			//cvLine(img,cvPoint(scc->centerX,scc->centerY),cvPoint(scc->centerX+10*cos_dir,scc->centerY-10*sin_dir),cvScalar(255));
			cvCircle( img, cvPoint(scc->centerX,scc->centerY),2,cvScalar(128),2);
			//cvRectangle(img,cvPoint(out[i].posx,out[i].posy),cvPoint(out[i].pos1x,out[i].pos1y),cvScalar(170));
			//cvPutText(img,s.str().c_str(),cvPoint(scc->centerX,scc->centerY),font,cvScalar(170));
		}
	}/*
	for(int i=1;i<labelimg->height-1;i++){
		unsigned int* ligneLabel0 = (unsigned int*)(labelimg->data.i + labelimg->width*(i-1));
		unsigned int* ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		unsigned int* ligneLabel1 = (unsigned int*)(labelimg->data.i + labelimg->width*(i+1));
		for(int j=1;j<labelimg->width-1;j++){
			CV_SetInt2D(img,i,j,0);
			if(ligneLabel[j]==254){
				if((ligneLabel0[j]==0&&ligneLabel1[j]==0)||(ligneLabel[j-1]==0&&ligneLabel[j+1]==0)||(ligneLabel0[j-1]==0&&ligneLabel1[j+1]==0)||(ligneLabel1[j-1]==0&&ligneLabel0[j+1]==0)
					||(ligneLabel1[j-1]==0&&ligneLabel0[j]==0)||(ligneLabel1[j]==0&&ligneLabel0[j+1]==0)||(ligneLabel0[j-1]==0&&ligneLabel1[j]==0)||(ligneLabel0[j]==0&&ligneLabel1[j+1]==0)){
					ligneLabel[j]=0;
				}
			}
		}
	}
	for(int i=1;i<real_highest_label;i++){
		scc=&out[i];
		if(scc->nbPixels>20){
			int color=scc->nbPixels+50;
			if(color>254)
				color=254;
			cvCircle( img, cvPoint(scc->centerX,scc->centerY),2,cvScalar(color));
			cvRectangle(img,cvPoint(out[i].posx,out[i].posy),cvPoint(out[i].pos1x,out[i].pos1y),cvScalar(170));
		}
	}*/
	return out;
}
void ConnectedComponents::relabel_image_WithColor(CvMat *labelimg, IplImage *img,unsigned char color){
	real_highest_label = 1;

	for(unsigned int id=1; id<labels.size(); id++){
		if(labels[id]==id){
			labels[id] = real_highest_label;
			real_highest_label++;
		}else{
			labels[id] = labels[labels[id]];
		}
	}

	vector<StatisticCC> out(real_highest_label);
	
	unsigned short lab;
	for(int i=0;i<labelimg->height;i++){
		unsigned int* ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		for(int j=0;j<labelimg->width;j++){
			lab=labels[ligneLabel[j]];
			if(lab>0){
				ligneLabel[j]=lab;
				if(out[lab].posx>j) out[lab].posx=j;
				if(out[lab].pos1x<=j) out[lab].pos1x=j+1;
				if(out[lab].posy>i)	out[lab].posy=i;
				if(out[lab].pos1y<=i) out[lab].pos1y=i+1;
				out[lab].centerX+=j;
				out[lab].centerY+=i;
				out[lab].nbPixels++;
			}
		}
	}
	
	for(int i=0;i<labelimg->height;i++){
		unsigned int* ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		unsigned char* ligneImg = (unsigned char*)(img->imageData + img->widthStep*i);
		for(int j=0;j<labelimg->width;j++){
			if((ligneLabel[j]>0)&&(out[ligneLabel[j]].nbPixels<1000)){
				ligneImg[j]=color;
			}
		}
	}
}
void ConnectedComponents::relabel_image_simple(CvMat *labelimg)
{
	real_highest_label=1;
	for(unsigned int id=1; id<labels.size(); id++){
		if(labels[id]==id){
			labels[id] = real_highest_label;
			real_highest_label++;
		}else{
			labels[id] = labels[labels[id]];
		}
	}
	for(int i=0;i<labelimg->height;i++){
		unsigned int* ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		for(int j=0;j<labelimg->width;j++){
			unsigned int tmp=ligneLabel[j];
			unsigned int tmpo=labels[ligneLabel[j]];
			ligneLabel[j]=labels[ligneLabel[j]];
		}
	}
}

int closeToCC(const IplImage *img, CvMat *labelimg,int i,int j){
	unsigned int* ligneLabel;
	unsigned char* ligneImg;
	ligneImg = (unsigned char*)(img->imageData+img->widthStep*i);
	ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*(i));
	if(ligneImg[j]==128)
		return ligneLabel[j];
	if(ligneImg[j+1]==128)
		return ligneLabel[j+1];
	if(ligneImg[j-1]==128)
		return ligneLabel[j-1];
	
	ligneImg = (unsigned char*)(img->imageData+img->widthStep*(i+1));
	ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*(i+1));
	if(ligneImg[j]==128)
		return ligneLabel[j];
	
	ligneImg = (unsigned char*)(img->imageData+img->widthStep*(i-1));
	ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*(i-1));
	if(ligneImg[j]==128)
		return ligneLabel[j];
	return -1;
}

int nbBlackClose(const IplImage *img,int i,int j,int taille=1){
	int out=0;
	unsigned char* ligneImg;
	for(int i1=i-taille;i1<=i+taille;i1++){
		if(i1>0&&i1<img->height-1){
			ligneImg = (unsigned char*)(img->imageData+img->widthStep*i1);
			for(int j1=j-taille;j1<=j+taille;j1++)
			{
				if(j1>0&&j1<img->width-1){
					if(ligneImg[j1]==0)
						out++;
				}
			}
		}
	}
	return out;
}

void ConnectedComponents::relabel_image_Noise(const IplImage *img, CvMat *labelimg,IplImage *out)
{

	real_highest_label = 1;
	unsigned int* ligneLabel;
	unsigned int* ligneLabelEnd;
	unsigned char* ligneImg;
	unsigned char* ligneImgPrev;
	unsigned char* ligneImgNext;
	unsigned char* ligneOut;
	unsigned char* ligneOutPrev;
	unsigned char* ligneOutNext;

	for(unsigned int id=1; id<labels.size(); id++){
		if(labels[id]==id){
			labels[id] = real_highest_label;
			real_highest_label++;
		}else{
			labels[id] = labels[labels[id]];
		}
	}
	double *nbVal=new double[real_highest_label];
	double *nbBlack=new double[real_highest_label];
	double *nbWhite=new double[real_highest_label];
	for(unsigned int i=0;i<real_highest_label;i++){
		nbVal[i]=0;
		nbBlack[i]=0;
		nbWhite[i]=0;
	}
	unsigned int root;
	for(int i=0;i<labelimg->height;i++){
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		ligneLabel[0]=labels[ligneLabel[0]];
		ligneLabel[labelimg->width-1]=labels[ligneLabel[labelimg->width-1]];
	}

	ligneLabel = (unsigned int*)(labelimg->data.i);
	ligneLabelEnd = (unsigned int*)(labelimg->data.i + labelimg->width*(labelimg->height-1));
	for(int i=0;i<labelimg->width;i++){
		ligneLabel[i]=labels[ligneLabel[i]];
		ligneLabelEnd[i]=labels[ligneLabelEnd[i]];
	}
	//count the number of black (resp white) pixel on each connected component border
	for(int i=1;i<labelimg->height-1;i++){
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		ligneImg = (unsigned char*)(img->imageData+img->widthStep*i);
		ligneImgPrev = (unsigned char*)(img->imageData+img->widthStep*(i-1));
		ligneImgNext = (unsigned char*)(img->imageData+img->widthStep*(i+1));
		ligneOut = (unsigned char*)(out->imageData+out->widthStep*i);
		ligneOutPrev = (unsigned char*)(out->imageData+out->widthStep*(i-1));
		ligneOutNext = (unsigned char*)(out->imageData+out->widthStep*(i+1));
		for(int j=1;j<labelimg->width-1;j++){
			if(ligneImg[j]!=128){
				root=labels[ligneLabel[j]];
				nbVal[root]++;
				//update the value of the label :
				ligneLabel[j]=root;
				if((ligneImgPrev[j]==128||ligneImg[j-1]==128||ligneImg[j+1]==128||ligneImgNext[j]==128)&&
				(ligneOutPrev[j]!=128||ligneOut[j-1]!=128||ligneOut[j+1]!=128||ligneOutNext[j]!=128)&&ligneImg[j]!=128){
					if(ligneImg[j]<60)
						nbBlack[root]++;
					else
						nbWhite[root]++;
				}
			}
		}
	}

	//Supprime les cliques entourée de blanc
	for(int i=0;i<labelimg->height;i++){
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		ligneImg = (unsigned char*)(img->imageData+img->widthStep*i);
		ligneOut = (unsigned char*)(out->imageData+img->widthStep*i);
		for(int j=0;j<labelimg->width;j++){
			if(ligneImg[j]!=128){
				root=ligneLabel[j];
				if(nbWhite[root]<=4*nbBlack[root])
					ligneOut[j]=254;
			}
		}
	}

	
	label_image(img,labelimg,0,true);
	vector<StatisticCC> statistics=computeStatisticsCC(out,labelimg);
	for(int i=0;i<labelimg->height;i++){
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		ligneImg = (unsigned char*)(img->imageData+img->widthStep*i);
		ligneOut = (unsigned char*)(out->imageData+img->widthStep*i);
		for(int j=0;j<labelimg->width;j++){
			if(ligneImg[j]!=128){
				root=ligneLabel[j];/*
				int sizeCC=(statistics[root].pos1x-statistics[root].posx)*(statistics[root].pos1y-statistics[root].posy);
				if((statistics[root].pos1x-statistics[root].posx)<=2||(statistics[root].pos1y-statistics[root].posy)<=2||sizeCC<9){
					ligneOut[j]=254;
				}
				if((double)statistics[root].nbPixels/(double)sizeCC>0.6){
					ligneOut[j]=254;
				}
				*/
				if(statistics[root].nbPixels<6){
					ligneOut[j]=254;
				}
			}
		}
	}
	statistics.clear();//*/
	
	
	delete nbWhite;
	delete nbBlack;
	delete nbVal;

}
void ConnectedComponents::relabel_imageProba(IplImage *img, CvMat *labelimg){
	real_highest_label = 1;
	unsigned int* ligneLabel;
	unsigned int* ligneLabelEnd;
	unsigned char* ligneImg;
	unsigned char* ligneImgPrev;
	unsigned char* ligneImgNext;

	for(unsigned int id=1; id<labels.size(); id++){
		if(labels[id]==id){
			labels[id] = real_highest_label;
			real_highest_label++;
		}else{
			labels[id] = labels[labels[id]];
		}
	}
	double *nbVal=new double[real_highest_label];
	double *nbBlack=new double[real_highest_label];
	double *nbWhite=new double[real_highest_label];
	double *probas=new double[real_highest_label];
	for(unsigned int i=0;i<real_highest_label;i++){
		nbVal[i]=0;
		nbBlack[i]=0;
		nbWhite[i]=0;
		probas[i]=0;
	}
	unsigned int root;
	for(int i=0;i<labelimg->height;i++){
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		ligneLabel[0]=labels[ligneLabel[0]];
		ligneLabel[labelimg->width-1]=labels[ligneLabel[labelimg->width-1]];
	}

	ligneLabel = (unsigned int*)(labelimg->data.i);
	ligneLabelEnd = (unsigned int*)(labelimg->data.i + labelimg->width*(labelimg->height-1));
	for(int i=0;i<labelimg->width;i++){
		ligneLabel[i]=labels[ligneLabel[i]];
		ligneLabelEnd[i]=labels[ligneLabelEnd[i]];
	}
	//count the number of black (resp white) pixel on each connected component border
	for(int i=1;i<labelimg->height-1;i++){
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		ligneImg = (unsigned char*)(img->imageData+img->widthStep*i);
		ligneImgPrev = (unsigned char*)(img->imageData+img->widthStep*(i-1));
		ligneImgNext = (unsigned char*)(img->imageData+img->widthStep*(i+1));
		for(int j=1;j<labelimg->width-1;j++){
			if(ligneImg[j]==128){
				root=labels[ligneLabel[j]];/*
				if(root>real_highest_label)
					cout<<"erreur"<<endl;
					*/
				//update the value of the label :
				nbVal[root]++;
				ligneLabel[j]=root;
				if(ligneImgPrev[j]!=128)
					probas[root]+=ligneImgPrev[j];
				if(ligneImgPrev[j]<128){
					nbWhite[root]++;
				}else{
					if(ligneImgPrev[j]>128){
						nbBlack[root]++;
					}
				}
				
				if(ligneImg[j-1]!=128)
					probas[root]+=ligneImg[j-1];
				if(ligneImg[j-1]<128){
					nbWhite[root]++;
				}else{
					if(ligneImg[j-1]>128){
						nbBlack[root]++;
					}
				}
				if(ligneImgNext[j]!=128)
					probas[root]+=ligneImgNext[j];
				if(ligneImgNext[j]<128){
					nbWhite[root]++;
				}else{
					if(ligneImgNext[j]>128){
						nbBlack[root]++;
					}
				}
				if(ligneImg[j+1]!=128)
					probas[root]+=ligneImg[j+1];
				if(ligneImg[j+1]<128){
					nbWhite[root]++;
				}else{
					if(ligneImg[j+1]>128){
						nbBlack[root]++;
					}
				}
			}
		}
	}

	for(int i=0;i<labelimg->height;i++){
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		ligneImg = (unsigned char*)(img->imageData+img->widthStep*i);
		for(int j=0;j<labelimg->width;j++){
			if(ligneImg[j]==128){
				root=ligneLabel[j];
				if(nbWhite[root]+nbBlack[root]>10){
					//double proba=(.5+((nbWhite[root]-1.5*nbBlack[root])/((nbWhite[root]+1.5*nbBlack[root])*4.0)));
					double proba=(probas[root]/(nbWhite[root]+nbBlack[root]));
					ligneImg[j]=proba;
					/*
					if(nbWhite[root]>=1.5*nbBlack[root])//+(nbWhite[root]+nbBlack[root])/2
						ligneImg[j]=192;
					else{
						ligneImg[j]=64;
					}*/
				}
			}
		}
	}
	delete nbWhite;
	delete nbBlack;
	delete nbVal;
	

}

IplImage * ConnectedComponents::relabel_image(const IplImage *img, CvMat *labelimg, bool updateTernaryImg)
{

	IplImage *out=cvCreateImage(cvSize(labelimg->width,labelimg->height),8,1);
	real_highest_label = 1;
	unsigned int* ligneLabel;
	unsigned int* ligneLabelEnd;
	unsigned char* ligneImg;
	unsigned char* ligneImgPrev;
	unsigned char* ligneImgNext;

	for(unsigned int id=1; id<labels.size(); id++)
  {
		if(labels[id]==id)
    {
			labels[id] = real_highest_label;
			real_highest_label++;
		}
    else
    {
			labels[id] = labels[labels[id]];
		}
	}
	double *nbVal=new double[real_highest_label];
	double *nbBlack=new double[real_highest_label];
	double *nbWhite=new double[real_highest_label];
	for(unsigned int i=0;i<real_highest_label;i++){
		nbVal[i]=0;
		nbBlack[i]=0;
		nbWhite[i]=0;
	}
	unsigned int root;
	for(int i=0;i<labelimg->height;i++){
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		ligneLabel[0]=labels[ligneLabel[0]];
		ligneLabel[labelimg->width-1]=labels[ligneLabel[labelimg->width-1]];
	}

	ligneLabel = (unsigned int*)(labelimg->data.i);
	ligneLabelEnd = (unsigned int*)(labelimg->data.i + labelimg->width*(labelimg->height-1));
	for(int i=0;i<labelimg->width;i++){
		ligneLabel[i]=labels[ligneLabel[i]];
		ligneLabelEnd[i]=labels[ligneLabelEnd[i]];
	}
	//count the number of black (resp white) pixel on each connected component border
	for(int i=1;i<labelimg->height-1;i++){
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		ligneImg = (unsigned char*)(img->imageData+img->widthStep*i);
		ligneImgPrev = (unsigned char*)(img->imageData+img->widthStep*(i-1));
		ligneImgNext = (unsigned char*)(img->imageData+img->widthStep*(i+1));
		for(int j=1;j<labelimg->width-1;j++){
			if(ligneImg[j]==128){
				root=labels[ligneLabel[j]];/*
				if(root>real_highest_label)
					cout<<"erreur"<<endl;
					*/
				//update the value of the label :
				nbVal[root]++;
				ligneLabel[j]=root;
				if(ligneImgPrev[j]<=50){
					nbBlack[root]++;
				}else{
					if(ligneImgPrev[j]>=200){
						nbWhite[root]++;
					}
				}
				if(ligneImg[j-1]<=50){
					nbBlack[root]++;
				}else{
					if(ligneImg[j-1]>=200){
						nbWhite[root]++;
					}
				}
				if(ligneImgNext[j]<=50){
					nbBlack[root]++;
				}else{
					if(ligneImgNext[j]>=200){
						nbWhite[root]++;
					}
				}
				if(ligneImg[j+1]<=50){
					nbBlack[root]++;
				}else{
					if(ligneImg[j+1]>=200){
						nbWhite[root]++;
					}
				}
			}
		}
	}
	/*
	//remove noises
	for(int i=1;i<labelimg->height-1;i++){
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*(i));
		ligneImg = (unsigned char*)(img->imageData+img->widthStep*i);
		for(int j=1;j<labelimg->width-1;j++){
			if(ligneImg[j]==0){//on est sur un blob.
				root=closeToCC(img,labelimg,i,j);
				if(root>0){
					if(nbWhite[root]>nbBlack[root])//it's background
						if(ligneImg[j]==0&&nbBlackClose(img,i,j)>=3){
							ligneLabel[j]=root;
							ligneImg[j]=128;
						}
				}
			}
		}
	}
	for(int i=labelimg->height-2;i>1;i--){
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*(i));
		ligneImg = (unsigned char*)(img->imageData+img->widthStep*i);
		for(int j=labelimg->width-2;j>1;j--){
			if(ligneImg[j]==0){//on est sur un blob.
				root=closeToCC(img,labelimg,i,j);
				if(root>0){
					if(nbWhite[root]>nbBlack[root])//it's background
						if(ligneImg[j]==0&&nbBlackClose(img,i,j)>=3){
							ligneLabel[j]=root;
							ligneImg[j]=128;
						}
				}
			}
		}
	}
	//*/

	// --->
	for(int i=1;i<labelimg->height-2;i++){
		ligneImg = (unsigned char*)(img->imageData+img->widthStep*i);
		for(int j=1;j<labelimg->width-2;j++){
			if(ligneImg[j]==0){//on est sur un blob.
				if(nbBlackClose(img,i,j)<=1||nbBlackClose(img,i,j,2)<=2){
					ligneImg[j]=254;
				}
			}
		}
	}
	/*
	// <---
	for(int i=labelimg->height-2;i>1;i--){
		ligneImg = (unsigned char*)(img->imageData+img->widthStep*i);
		for(int j=labelimg->width-2;j>1;j--){
			if(ligneImg[j]==0){//on est sur un blob.
				if(nbBlackClose(img,i,j)<=1||nbBlackClose(img,i,j,2)<=2){
					ligneImg[j]=254;
				}
			}
		}
	}//*/
	
	
	//find the biggest clic, and set to background :
	unsigned int idMax=0,nbMax=0;
	/*
	for(unsigned int id=1; id<real_highest_label; id++){
		if(nbMax<nbBlack[id]+nbWhite[id]){
			nbMax=nbBlack[id]+nbWhite[id];
			idMax=id;
		}
	}
	*/
	unsigned char* ligneOut;

	for(int i=0;i<labelimg->height;i++){
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		ligneImg = (unsigned char*)(img->imageData+img->widthStep*i);
		ligneOut = (unsigned char*)(out->imageData+img->widthStep*i);
		for(int j=0;j<labelimg->width;j++){
			if(ligneImg[j]!=128)//allready labelled
      {
				ligneOut[j]=ligneImg[j];
        if( updateTernaryImg&&ligneOut[j]==0 )
          ligneImg[j]=0;
      }
			else
      {
				root=ligneLabel[j];
				if(nbWhite[root]>nbBlack[root])// modif V7 : 1.5*nbBlack[root]
					ligneOut[j]=254;
				else{
					ligneOut[j]=0;
        if( updateTernaryImg )
          ligneImg[j]=0;
				}
			}
		}
	}
	delete nbWhite;
	delete nbBlack;
	delete nbVal;
	

	return out;
}

vector<StatisticCC> ConnectedComponents::computeStatisticsCC(const IplImage *ternaire,CvMat *labelimg){

	real_highest_label = 1;

	for(unsigned int id=1; id<labels.size(); id++){
		if(labels[id]==id){
			labels[id] = real_highest_label;
			real_highest_label++;
		}else{
			labels[id] = labels[labels[id]];
		}
	}

	vector<StatisticCC> out(real_highest_label);

	unsigned short lab;
	for(int i=0;i<labelimg->height;i++){
		for(int j=0;j<labelimg->width;j++){
			if(CV_GetReal2DMatI(labelimg,i,j)>0){
				lab=labels[CV_GetReal2DMatI(labelimg,i,j)];
				CV_SetReal2DMatI(labelimg,i,j,lab);
				UPDATE_STATISTICS(i,j,lab);
			}
		}
	}
	widthMean=0;
	heightMean=0;
	widthVar=0;
	heightVar=0;
	int nbVals=0;
	int width=0;
	int height=0;
	for(unsigned int index=1;index<real_highest_label;index++){
		width=out[index].pos1x-out[index].posx;
		height=out[index].pos1y-out[index].posy;
		double densite=(double)out[index].nbPixels/(width*height);

		if((width>=2)&&(height>=2)){
			widthMean+=(out[index].pos1x-out[index].posx);
			heightMean+=(out[index].pos1y-out[index].posy);
			nbVals++;
		}else{
			out[index].ccClass=11;
		}
	}

	widthMean/=nbVals;
	heightMean/=nbVals;
	for(unsigned int index=1;index<real_highest_label;index++){
		if(out[index].ccClass<10){
			widthVar+=pow(widthMean-(out[index].pos1x-out[index].posx),2);
			heightVar+=pow(heightMean-(out[index].pos1y-out[index].posy),2);
		}
	}
	widthVar=sqrt(widthVar/nbVals);
	heightVar=sqrt(heightVar/nbVals);
	return out;
}

int ConnectedComponents::searchCloseCCText(vector<StatisticCC> blobs,StatisticCC ref){
	int centerx=(ref.posx+ref.pos1x)/2;
	int centery=(ref.posy+ref.pos1y)/2;
	double sizeMin=9999999.0;
	int posMin=-1;
	for(unsigned int iter=1;iter<blobs.size();iter++){
		if(blobs[iter].ccClass==1){
			if(centerx<blobs[iter].pos1x&&centery<blobs[iter].pos1y)
				if(centerx>blobs[iter].posx&&centery>blobs[iter].posy){
					if(sizeMin>(blobs[iter].pos1x-blobs[iter].posx)*(blobs[iter].pos1y-blobs[iter].posy)){
						sizeMin=(blobs[iter].pos1x-blobs[iter].posx)*(blobs[iter].pos1y-blobs[iter].posy);
						posMin=iter;
					}
				}
		}
	}
	return posMin;
}
int ConnectedComponents::hasCloseC(vector<StatisticCC> blobs,int i){
	//compute the augmented box :
	int leftBox=(int)(blobs[i].posx-2*(blobs[i].pos1x-blobs[i].posx));
	int bottomBox=(int)(blobs[i].pos1y+2*(blobs[i].pos1y-1.0*blobs[i].posy));
	int rightBox=(int)(blobs[i].pos1x+2*(blobs[i].pos1x-1.0*blobs[i].posx));
	int topBox=(int)(blobs[i].posy-2*(blobs[i].pos1y-1.0*blobs[i].posy));
	for(unsigned int iter=1;iter<blobs.size();iter++){
		if(!blobs[iter].ccClass&&iter!=i){
			if(leftBox<=blobs[iter].pos1x&&blobs[i].posy<=blobs[iter].pos1y){
				if(blobs[i].pos1x>=blobs[iter].posx&&blobs[i].pos1y>=blobs[iter].posy)
					return iter;
			}
			if(blobs[i].posx<=blobs[iter].pos1x&&blobs[i].posy<=blobs[iter].pos1y){
				if(blobs[i].pos1x>=blobs[iter].posx&&bottomBox>=blobs[iter].posy)
					return iter;
			}
			if(blobs[i].posx<=blobs[iter].pos1x&&blobs[i].posy<=blobs[iter].pos1y){
				if(rightBox>=blobs[iter].posx&&blobs[i].pos1y>=blobs[iter].posy)
					return iter;
			}
		}
	}
	return 0;
}


void ConnectedComponents::filterCC(const IplImage *imgColor,const IplImage *imgBin, const IplImage *imgTern, CvMat *labelimgPrev){

	vector<unsigned int> oldLabels(labels);

  seedsExtands(imgTern,imgBin,imgColor);
  
  ////////////////////////////////For now, skip this step//////////////////////
  return;

	CvMat* labelimg = cvCreateMat(imgTern->height,imgTern->width,CV_32SC1);
  
	//first recompute the CC (white and black):
	label_image(imgTern,labelimg,0,true);
	unsigned int* ligneLabel;
	unsigned char* ligneBin;
	unsigned char* ligneBinPrev;
	unsigned char* ligneBinNext;
	unsigned char* ligneImg;
	unsigned char* ligneImgPrev;
	unsigned char* ligneImgNext;
	vector<StatisticCC> statistics=computeStatisticsCC(imgTern,labelimg);
	
	double *nbVal=new double[real_highest_label];
	double *nbBlack=new double[real_highest_label];
	double *nbWhite=new double[real_highest_label];
	for(unsigned int i=0;i<real_highest_label;i++){
		nbVal[i]=0;
		nbBlack[i]=0;
		nbWhite[i]=0;
	}
	unsigned int root;

//	cvShowImage("finalImg",imgBin);
//	cvWaitKey(0);

	//count the number of black (resp white) pixel on each connected component border
	for(int i=1;i<labelimg->height-2;i++)
  {
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		ligneImg = (unsigned char*)(imgTern->imageData+imgTern->widthStep*i);
		ligneImgPrev = (unsigned char*)(imgTern->imageData+imgTern->widthStep*(i-1));
		ligneImgNext = (unsigned char*)(imgTern->imageData+imgTern->widthStep*(i+1));
		ligneBin = (unsigned char*)(imgBin->imageData+imgBin->widthStep*i);
		ligneBinPrev = (unsigned char*)(imgBin->imageData+imgBin->widthStep*(i-1));
		ligneBinNext = (unsigned char*)(imgBin->imageData+imgBin->widthStep*(i+1));
		for(int j=1;j<labelimg->width-2;j++)
    {
			if(ligneImg[j]==0)
      {
				root=ligneLabel[j];

				//update the value of the label :
				nbVal[root]++;
				{
					if(ligneImgPrev[j]>=200)
						nbWhite[root]++;
          else if(ligneImgPrev[j]==0)
            nbBlack[root]++;
				}
				{
					if(ligneImg[j-1]>=200)
						nbWhite[root]++;
          else if(ligneImg[j-1]==0)
            nbBlack[root]++;
				}
				{
					if(ligneImgNext[j]>=200)
						nbWhite[root]++;
          else if(ligneImgNext[j]==0)
            nbBlack[root]++;
				}
				{
					if(ligneImg[j+1]>=200)
						nbWhite[root]++;
          else if(ligneImg[j+1]==0)
            nbBlack[root]++;
				}
			}
		}
	}
	
	//relabel strange CC:
	for(int i=labelimg->height-1;i>=0;i--){
		ligneBin = (unsigned char*)(imgBin->imageData+imgBin->widthStep*i);
		ligneImg = (unsigned char*)(imgTern->imageData+imgTern->widthStep*i);
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		for(int j=labelimg->width-1;j>=0;j--){
			if(ligneImg[j]==0){//on est sur un blob.
				root=ligneLabel[j];
				if(root>0){
					//is this label correct?
					if( nbWhite[root]>100*nbBlack[root] ){//nbWhite[root]<2*nbBlack[root] || 
						ligneImg[j]=128;
						ligneBin[j]=128;
					}//else{
					//	if(((statistics[root].pos1x>imgBin->width-10||statistics[root].posx<10)&&(statistics[root].pos1x-statistics[root].posx<40))||
					//		((statistics[root].pos1y>imgBin->height-10||statistics[root].posy<10)&&(statistics[root].pos1y-statistics[root].posy<40))){
					//			ligneImg[j]=254;
					//			ligneBin[j]=254;
					//	}
					//}
				}
			}
		}
	}
	
	cvReleaseMat(&labelimg);
}


float ConnectedComponents::computeColorMean(const IplImage* imgSrc,CvMat* labelimg,int width,int x,int y,int idLabel)
{
	float out=0;
	int nb=0;
	unsigned int* ligneLabel;
	unsigned char* ligneBin;
	int xDeb=x-width/2;
	int yDeb=y-width/2;
	if(xDeb<0) xDeb=0;
	if(yDeb<0) yDeb=0;
	if(xDeb+width>imgSrc->width) xDeb=imgSrc->width-width-1;
	if(yDeb+width>imgSrc->height) yDeb=imgSrc->height-width-1;
	int xFin=xDeb+width;
	int yFin=yDeb+width;

	//count the number of black (resp white) pixel on each connected component border
	for(int i=yDeb;i<yFin;i++){
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		ligneBin = (unsigned char*)(imgSrc->imageData+imgSrc->widthStep*i);
		for(int j=xDeb;j<xFin;j++){
			if(ligneLabel[j]==idLabel){
				out+=ligneBin[j];
				nb++;
			}
		}
	}
	if(nb==0)
		return 0;
	return out/nb;
}

void ConnectedComponents::filterCC(IplImage *out, CvMat *labelimg){

	//first recompute the CC (white and black):
	label_image(out,labelimg,0,true);
    unsigned int* ligneLabel;
    unsigned char* ligneImg;
	vector<StatisticCC> statistics=computeStatisticsCC(out,labelimg);
	unsigned int sizeTmp;
	unsigned int index;
	unsigned int width,height;
	int dist=0;
	CvFont font;
	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,0.5,0.5,0,2);

	//1- essaye de trouver la classe les CC : 0->inconnu,1->texte et 2-> bruit...

	for(unsigned int i=0;i<statistics.size();i++){
		width=(statistics[i].pos1x-statistics[i].posx);
		height=(statistics[i].pos1y-statistics[i].posy);
		sizeTmp=width*height;

		//too tiny... It's noise :
		if(statistics[i].ccClass==0){
			//look arround to estimate the class :
			statistics[i].ccClass=findText(statistics,i);
		}
	}
/*
	//2- les CC dont la classe n'est pas déterminée vont subir une contre expertise ;-)
	//on commence d'abord par retrouver la taille moyenne du bruit
	//(nous permettra de savoir par la suite à quoi nous avons a faire...)
	
	widthMeanN=0;
	heightMeanN=0;
	widthMean=0;
	heightMean=0;
	int nbNoise=0;
	int nbOK=0;
	densite=0;
	for(unsigned int i=1;i<statistics.size();i++){
		if(statistics[i].ccClass!=1){//c'est du bruit ou autre...
			if(statistics[i].ccClass!=10){//le bruit trop exceptionnel n'est pas a prendre en compte
				widthMeanN+=(statistics[i].pos1x-statistics[i].posx);
				heightMeanN+=(statistics[i].pos1y-statistics[i].posy);
				nbNoise++;
			}
		}else{
			widthMean+=(statistics[i].pos1x-statistics[i].posx);
			heightMean+=(statistics[i].pos1y-statistics[i].posy);
			nbOK++;
			densite+=(double)statistics[i].nbPixels/((statistics[i].pos1x-statistics[i].posx)*(statistics[i].pos1y-statistics[i].posy));
		}
	}
	widthMeanN/=nbNoise;
	heightMeanN/=nbNoise;
	widthMean/=nbOK;
	heightMean/=nbOK;
	densite/=nbOK;
	//cout<<"densite : "<<densite<<endl;
	//compute projection :
	projection(out,statistics);

	//3- Maintenant, nous regardons pour chacun des CC indeterminé le voisinage.
	//On essaye de déterminer la classe finale :

	int nbChanges=1;
	int classF;
	while(nbChanges>0){
		nbChanges=0;
		for(unsigned int i=1;i<statistics.size();i++){
			if(statistics[i].ccClass==0){
				classF=classFinale(statistics,i);
				if(classF!=0){
					statistics[i].ccClass=classF;
					nbChanges++;
				}
			}
		}
	}
	if(densite>0.3){
		finalise(statistics,out);
		nbChanges=1;
		while(nbChanges>0){
			nbChanges=0;
			for(unsigned int i=1;i<statistics.size();i++){
				if(statistics[i].ccClass==0){
					classF=classFinale(statistics,i);
					if(classF!=0){
						statistics[i].ccClass=classF;
						nbChanges++;
					}
				}
			}
		}
	}
*/

	for(int i=0;i<labelimg->height;i++){
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		ligneImg = (unsigned char*)(out->imageData + out->widthStep*i);
		for(int j=0;j<labelimg->width;j++){
			index=ligneLabel[j];
			if(index>0){
					double CCsize=(statistics[index].pos1x-statistics[index].posx)*(statistics[index].pos1y-statistics[index].posy);
				if(statistics[index].ccClass==0&&CCsize<1000||
					(statistics[index].ccClass>1)){//met du blanc car c'est du bruit !
						if(ligneImg[j]==0)
							ligneImg[j]=254;
				}
			}
		}
	}

//Now, the background CC :

	label_image(out,labelimg,254);
	double divise;
	//cout<<"densite "<<densite<<endl;
	densite<0.3?divise=254:divise=128.0;
	vector<StatisticCC> statisticsW=computeStatisticsCC(out,labelimg);
	for(unsigned int i=1;i<statisticsW.size();i++){
		int father=searchCloseCCText(statistics,statisticsW[i]);
		if(father>0){
			double dens=(double)statistics[father].nbPixels/((statistics[father].pos1x-statistics[father].posx)*(statistics[father].pos1y-statistics[father].posy));
			if(densite>0.3){
					if((statisticsW[i].pos1x-statisticsW[i].posx)*(statisticsW[i].pos1y-statisticsW[i].posy)<((double)(statistics[father].pos1x-statistics[father].posx)*(statistics[father].pos1y-statistics[father].posy))/divise){
						statisticsW[i].ccClass=1;
	//					cout<<"width : "<<(statistics[father].pos1x-statistics[father].posx)<<" height : "<<(statistics[father].pos1y-statistics[father].posy)<<endl;
	//					cout<<"width : "<<(statisticsW[i].pos1x-statisticsW[i].posx)<<" height : "<<(statisticsW[i].pos1y-statisticsW[i].posy)<<endl;
					}
			}
		}
	}
	
	for(int i=0;i<labelimg->height;i++){
		ligneLabel = (unsigned int*)(labelimg->data.i + labelimg->width*i);
		ligneImg = (unsigned char*)(out->imageData + out->widthStep*i);
		for(int j=0;j<labelimg->width;j++){
			index=ligneLabel[j];
			if(index>0){
				if(statisticsW[index].ccClass==1)
					ligneImg[j]==0?ligneImg[j]=254:ligneImg[j]=0;
			}
		}
	}
	/*
	for(unsigned int i=0;i<statistics.size();i++){
		char nbChar[200];
		double CCsize=(statistics[i].pos1x-statistics[i].posx)*(statistics[i].pos1y-statistics[i].posy);
		sprintf(nbChar, "%f", CCsize);
		//sprintf(nbChar, "%i", statistics[i].ccClass);

		if(statistics[i].ccClass!=1)
			cvPutText(out,nbChar,cvPoint(statistics[i].posx,statistics[i].posy),&font,cvScalar(128));
	}*/
	
	
}


//analyse le voisinage du CC i...
//retourne la classe du CC :
//0 inconnu
//1 c'est un caractère de texte (ce qu'il y a des plus dans un texte...)
//>9 c'est du bruit, sur à 100%
int ConnectedComponents::classFinale(vector<StatisticCC> blobs,int i){
	//have we to search ?
	if(blobs[i].ccClass>9)
		return blobs[i].ccClass;
	//init the var :
	int sizeCC=(blobs[i].pos1x-blobs[i].posx)*(blobs[i].pos1y-blobs[i].posy);
	double width=(double)(blobs[i].pos1x-blobs[i].posx);
	double height=(double)(blobs[i].pos1y-blobs[i].posy);

	//very small CC, it's noise or ponctuation :
	if((width<widthMeanN-widthMeanN*0.1)&&(height<heightMeanN-heightMeanN*0.1)&&densite>0.3){
		//compute the proba :
		double probaV=0.f,probaH=0.f;
		for(int j=blobs[i].posx;j<blobs[i].pos1x;j++){
			if(probaV<projVerticale[j])
				probaV=projVerticale[j];
		}
		for(int j=blobs[i].posy;j<blobs[i].pos1y;j++){
			if(probaH<projHorizontale[j])
				probaH=projHorizontale[j];
		}
		//attention aux ponctuations : elles ne se trouve pas directement sur les lignes de texte...
		if(probaV>0.2&&probaH>0.3)
			return 1;

		return 12;
	}
	//compute the augmented box :

	bool overlap=false;
	int nbNoise=0;
	int nextToText=0;
	int leftBox=(int)(blobs[i].posx-(blobs[i].pos1x-blobs[i].posx));
	if(leftBox<0)leftBox=0;
	int bottomBox=(int)(blobs[i].pos1y+(blobs[i].pos1y-1.0*blobs[i].posy));
	int rightBox=(int)(blobs[i].pos1x+(blobs[i].pos1x-1.0*blobs[i].posx));
	int topBox=(int)(blobs[i].posy+0.5*(blobs[i].pos1y-1.0*blobs[i].posy));
	if(topBox<0)topBox=0;

	if(fabs(height-heightMeanN)+fabs(width-widthMeanN)<fabs(height-heightMean)+fabs(width-widthMean)){

		//compute the proba :
		double probaV=0.f,probaH=0.f;
		int nbV=0,nbH=0;
		for(int j=blobs[i].posx;j<blobs[i].pos1x;j++){
			probaV+=projVerticale[j];
			nbV++;
		}
		for(int j=blobs[i].posy;j<blobs[i].pos1y;j++){
			probaH+=projHorizontale[j];
			nbH++;
		}
		probaV/=nbV;
		probaH/=nbH;
		if(probaV<0.1||probaH<0.2){
			for(unsigned int iter=1;iter<blobs.size();iter++){
				overlap=false;
				if(blobs[iter].ccClass==1&&iter!=i){
					if(leftBox<=blobs[iter].pos1x&&blobs[i].posy<=blobs[iter].pos1y){
						if(blobs[i].pos1x>=blobs[iter].posx&&blobs[i].pos1y>=blobs[iter].posy)
							overlap=true;
					}
					if(blobs[i].posx<=blobs[iter].pos1x&&blobs[i].posy<=blobs[iter].pos1y){
						if(blobs[i].pos1x>=blobs[iter].posx&&bottomBox>=blobs[iter].posy)
							overlap=true;
					}
					if(overlap)
						return 1;
				}
			}
			return 15;
		}
		return 1;
	}
	
	leftBox=(int)(blobs[i].posx-3.5*(blobs[i].pos1x-blobs[i].posx));
	if(leftBox<0)leftBox=0;
	bottomBox=(int)(blobs[i].pos1y+2.5*(blobs[i].pos1y-1.0*blobs[i].posy));
	topBox=(int)(blobs[i].posy-(blobs[i].pos1y-1.0*blobs[i].posy));
	if(topBox<0)topBox=0;

	//compute the proba :
	double probaV=0.f,probaH=0.f;
	int nbV=0,nbH=0;
	for(int j=blobs[i].posx;j<blobs[i].pos1x;j++){
		probaV+=projVerticale[j];
		nbV++;
	}
	for(int j=blobs[i].posy;j<blobs[i].pos1y;j++){
		probaH+=projHorizontale[j];
		nbH++;
	}
	probaV/=nbV;
	probaH/=nbH;

	//size correct...
	//if the CC is alone, it's noise...
	//if it's connected to text, it's OK
	//with this augmented box, we look around for box that overlap:
	for(unsigned int iter=1;iter<blobs.size();iter++){
		overlap=false;
		if(blobs[iter].ccClass==1&&iter!=i){

			if(leftBox<=blobs[iter].pos1x&&blobs[i].posy<=blobs[iter].pos1y){
				if(blobs[i].pos1x>=blobs[iter].posx&&blobs[i].pos1y>=blobs[iter].posy)
					overlap=true;
			}
			if(blobs[i].posx<=blobs[iter].pos1x&&blobs[i].posy<=blobs[iter].pos1y){
				if(blobs[i].pos1x>=blobs[iter].posx&&bottomBox>=blobs[iter].posy)
					overlap=true;
			}
			if(blobs[i].posx<=blobs[iter].pos1x&&blobs[i].posy<=blobs[iter].pos1y){
				if(rightBox>=blobs[iter].posx&&blobs[i].pos1y>=blobs[iter].posy)
					overlap=true;
			}
			if(blobs[i].posx<=blobs[iter].pos1x&&topBox<=blobs[iter].pos1y){
				if(blobs[i].pos1x>=blobs[iter].posx&&blobs[i].pos1y>=blobs[iter].posy)
					overlap=true;
			}

			if(overlap&&(probaV>0.01&&probaH>0.01))
				return 1;
		}
	}
	if(probaV<0.05||probaH<0.15)
		return 15;
	if(probaV>0.3||probaH>0.4)
		return 1;
	return 0;
}

//analyse le voisinage du CC i...
//retourne la classe du CC :
//0 inconnu
//1 c'est un caractère de texte (ce qu'il y a de plus dans un texte...)
//>9 c'est du bruit, sur à 100%
int ConnectedComponents::findText(vector<StatisticCC> blobs,int i){
	//have we to search ?
	if(blobs[i].ccClass>9)
		return blobs[i].ccClass;
	//init the var :
	int sizeCC=(blobs[i].pos1x-blobs[i].posx)*(blobs[i].pos1y-blobs[i].posy);
	int smaller=0;//number of CC smaller than i
	int same=0;//number of CC with same size than i (+- 40%...)
	int bigger=0;//number of CC bigger than i
	int bbigger=0;//number of CC >3*bigger than i
	//compute the augmented box :
	int leftBox=blobs[i].posx-5*(blobs[i].pos1x-blobs[i].posx);
	if(leftBox<0)	leftBox=0;
	int bottomBox=(int)(blobs[i].pos1y+2*(blobs[i].pos1y-1.0*blobs[i].posy));
	int rightBox=(int)(blobs[i].pos1x+3*(blobs[i].pos1x-1.0*blobs[i].posx));
	int topBox=(int)(blobs[i].posy-1*(blobs[i].pos1y-1.0*blobs[i].posy));

	//with this augmented box, we look around for box that overlap:
	bool overlap=false;
	for(unsigned int iter=1;iter<blobs.size();iter++){
		overlap=false;
		if(blobs[iter].ccClass<=1&&iter!=i){

			if(leftBox<=blobs[iter].pos1x&&topBox<=blobs[iter].pos1y){
				if(rightBox>=blobs[iter].posx&&bottomBox>=blobs[iter].posy)
					overlap=true;
			}

			if(overlap){//what is this connected componnent relative to i?
				int sizeTmp1=(blobs[iter].pos1x-blobs[iter].posx)*(blobs[iter].pos1y-blobs[iter].posy);
				if(sizeTmp1<sizeCC-(sizeCC*50/100))
					smaller++;
				else
					if(sizeTmp1>3*sizeCC)
						bbigger++;
					else
						if(sizeTmp1>sizeCC+(sizeCC*50/100))
							bigger++;
						else
							same++;
			}
		}
	}
	if(same+bigger+bbigger!=0)
	{
		
		if(sizeCC>25){
			double seuil=0.7;
			if(blobs[i].nbPixels>100)
				seuil-=0.08;
			if(blobs[i].nbPixels>150)
				seuil-=0.05;
			if(blobs[i].nbPixels>185)
				seuil-=0.03;
			if((double)blobs[i].nbPixels/(double)sizeCC>seuil){
				return 14;
			}
		}
		return 1;//it's text 100% (hope...)
	}

	//des voisins ?
	if(bbigger+bigger+same+smaller==0){
		return 15;//no : it's noise...
	}
	if((blobs[i].pos1x-blobs[i].posx)>3&&(blobs[i].pos1y-blobs[i].posy)>3){
		if(0.8<(double)blobs[i].nbPixels/((blobs[i].pos1x-blobs[i].posx)*(blobs[i].pos1y-blobs[i].posy))){
			return 16;
		}
	}
	if(sizeCC>25&&blobs[i].pos1x-blobs[i].posx>4){
		double seuil=0.7;
		if(seuil<(double)blobs[i].nbPixels/(double)sizeCC){
			return 17;
		}
	}

	return 0;
}

void ConnectedComponents::projection(const IplImage *img,const vector<StatisticCC> stats){
	this->projHorizontale=new double[img->height];
	this->projVerticale=new double[img->width];
	for(int cpt=0;cpt<img->width;cpt++)
		projVerticale[cpt]=0.f;
	for(int cpt=0;cpt<img->height;cpt++)
		projHorizontale[cpt]=0.f;
	for(unsigned int i=1;i<stats.size();i++){
		if(stats[i].ccClass==1||stats[i].ccClass==0){
			for(int j=stats[i].posx;j<stats[i].pos1x;j++)
				projVerticale[j]++;
			for(int j=stats[i].posy;j<stats[i].pos1y;j++)
				projHorizontale[j]++;
		}
	}
	double maxiH=0.f,maxiV=0.f;
	for(int cpt=0;cpt<img->width;cpt++){
		if(projVerticale[cpt]>maxiV)
			maxiV=projVerticale[cpt];
	}
	for(int cpt=0;cpt<img->height;cpt++){
		if(projHorizontale[cpt]>maxiH)
			maxiH=projHorizontale[cpt];
	}
	//passe en proba :
	for(int cpt=0;cpt<img->width;cpt++)
		projVerticale[cpt]=projVerticale[cpt]/maxiV;
	for(int cpt=0;cpt<img->height;cpt++)
		projHorizontale[cpt]=projHorizontale[cpt]/maxiH;
/*
	IplImage *test = cvCreateImage(cvSize(img->width,img->height),8,1);
	for(int i=0;i<img->height;i++){
		for(int j=0;j<img->width;j++){
//			CV_SetChar2D(test,i,j,((projHorizontale[i]+projVerticale[j])*127.5f));
//			CV_SetChar2D(test,i,j,(projHorizontale[i]*255.f));
			CV_SetChar2D(test,i,j,(projVerticale[j]*255.f));
		}
	}
	cvSaveImage("tmp.jpg",test);
*/
}
void ConnectedComponents::finalise(vector<StatisticCC> &blobs,IplImage *img){
	//d'abord trouver la position des CC :
	unsigned int haut=1;
	unsigned int bas=1;
	unsigned int droite=1;
	unsigned int gauche=1;
	unsigned int i=1;
	while(blobs[i].ccClass!=1)
		i++;
	haut=i;
	bas=i;
	droite=i;
	gauche=i;
	for(;i<blobs.size();i++){
		if(blobs[i].ccClass==1){
			if(blobs[haut].posy>blobs[i].posy)
				haut=i;
			if(blobs[bas].pos1y<blobs[i].pos1y)
				bas=i;
			if(blobs[droite].pos1x<blobs[i].pos1x)
				droite=i;
			if(blobs[gauche].posx>blobs[i].posx)
				gauche=i;
		}
	}
	//on a donc les CC les plus éloignées... Calculons le ratio largeur/hauteur des CC qui l'entoure :
	double ratioh=0,ratiob=0,ratiod=0,ratiog=0;
	unsigned int nbh=0,nbb=0,nbd=0,nbg=0;

	for(unsigned int i=1;i<blobs.size();i++){
		if(blobs[i].ccClass==1){
			//pour le haut :
			if(blobs[i].posy<blobs[haut].pos1y+(img->height*2)/100){
				//compute the ratio :
				ratioh+=((double)(blobs[i].pos1x-blobs[i].posx)/(blobs[i].pos1y-blobs[i].posy))*blobs[i].nbPixels;
				nbh+=blobs[i].nbPixels;
			}
			//pour le bas :
			if(blobs[i].pos1y>blobs[bas].posy-(img->height*2)/100){
				ratiob+=((double)(blobs[i].pos1x-blobs[i].posx)/(blobs[i].pos1y-blobs[i].posy))*blobs[i].nbPixels;
				nbb+=blobs[i].nbPixels;
			}
			//pour la droite :
			if(blobs[i].pos1x>blobs[droite].posx-(img->width*2)/100){
				ratiod+=((double)(blobs[i].pos1x-blobs[i].posx)/(blobs[i].pos1y-blobs[i].posy))*blobs[i].nbPixels;
				nbd+=blobs[i].nbPixels;
			}
			//pour la gauche :
			if(blobs[i].posx<blobs[gauche].pos1x+(img->width*2)/100){
				ratiog+=((double)(blobs[i].pos1x-blobs[i].posx)/(blobs[i].pos1y-blobs[i].posy))*blobs[i].nbPixels;
				nbg+=blobs[i].nbPixels;
			}
		}
	}
	ratioh/=nbh;
	ratiob/=nbb;
	ratiod/=nbd;
	ratiog/=nbg;
	/*
	cout<<"ratioh : "<<ratioh<<endl;
	cout<<"ratiob : "<<ratiob<<endl;
	cout<<"ratiod : "<<ratiod<<endl;
	cout<<"ratiog : "<<ratiog<<endl;
	*/
	bool hautIsNotGood=ratioh>3||nbh<=2;
	bool basIsNotGood=ratiob>3||nbb<=2;
	bool droiteIsNotGood=ratiod<0.33||nbd<=2;
	bool gaucheIsNotGood=ratiog<0.33||nbg<=2;
	for(unsigned int i=1;i<blobs.size();i++){
		if(blobs[i].ccClass==1){
			//pour le haut :
			if(hautIsNotGood&&blobs[i].posy<blobs[haut].pos1y+(img->height*2)/100){
				if((double)(blobs[i].pos1x-blobs[i].posx)/(blobs[i].pos1y-blobs[i].posy)>2.0)
					blobs[i].ccClass=16;
				else
					blobs[i].ccClass=0;
			}
			//pour le bas :
			if(basIsNotGood&&blobs[i].pos1y>blobs[bas].posy-(img->height*2)/100){
				if((double)(blobs[i].pos1x-blobs[i].posx)/(blobs[i].pos1y-blobs[i].posy)>2.0)
					blobs[i].ccClass=16;
				else
					blobs[i].ccClass=0;
			}
			//pour la droite :
			if(droiteIsNotGood&&blobs[i].pos1x>blobs[droite].posx-(img->width*2)/100){
				if((double)(blobs[i].pos1x-blobs[i].posx)/(blobs[i].pos1y-blobs[i].posy)<0.35)
					blobs[i].ccClass=16;
				else
					blobs[i].ccClass=0;
			}
			//pour la gauche :
			if(gaucheIsNotGood&&blobs[i].posx<blobs[gauche].pos1x+(img->width*2)/100){
				if((double)(blobs[i].pos1x-blobs[i].posx)/(blobs[i].pos1y-blobs[i].posy)<0.35)
					blobs[i].ccClass=16;
				else
					blobs[i].ccClass=0;
			}
		}
	}
}
