#include "OpImg.h"

#include <opencv2/features2d/features2d.hpp>

#include "EM_modif.h"

using namespace cv;

IplImage *OpImg::debugImg=NULL;
CvMat* OpImg::labelimg = NULL;


//fonction qui permet de définir la classe des pixels proche des contours
//out : image ternaire où:
//0 correspond à du texte
//254 correspond à du fond
//128 correspond à une classe indeterminée
//IplImage *OpImg::expandEdge(IplImage *img,PileCanny *pc,IplImage *e,int demiMaskKmean,char demiMaskAnalyse=2){
IplImage *OpImg::expandEdge(IplImage *img,PileCanny* p,IplImage *e,int demiMaskKmean){
  	char demiMaskAnalyse=demiMaskKmean;
	edge=e;

  /*/////////////////////////////////////////////////////////////////////////
  //distance from edges construction:
  Mat cannyMat(edge), distanceMapTMP, labelsTMP;
  Mat cannyInvert = 255 - cannyMat.clone();
  distanceTransform( cannyInvert, distanceMapTMP, labelsTMP, CV_DIST_L2, CV_DIST_MASK_PRECISE);
  /////////////////////////////////////////////////////////////////////////*/

	IplImage *out= cvCreateImage(cvSize(img->width,img->height),8,1);
	static IplImage *meanTxt= cvCreateImage(cvSize(img->width,img->height),8,1);
	static IplImage *meanBack= cvCreateImage(cvSize(img->width,img->height),8,1);
	if(debugImg==NULL){
		debugImg= cvCreateImage(cvSize(img->width,img->height),8,1);
		labelimg=cvCreateMat(img->height,img->width,CV_32FC1);
	}
	cvSet(debugImg,cvScalar(128));
	cvSet(labelimg,cvScalar(0));
	cvSet(meanTxt,cvScalar(254));
	cvSet(meanBack,cvScalar(0));

	demiMaskKmean+=1;//was 2
	uchar widthMask=2*demiMaskKmean+1;
	float *theWindow=new float[widthMask*widthMask];
	float mean[2];
	mean[0]=0;
	mean[1]=0;
	int m;

	unsigned char* ligneImg;
	unsigned char* ligneEdge;
	unsigned char* ligneOut;
	unsigned char* ligneDebug;
	unsigned char* ligneMeanT;
	unsigned char* ligneMeanF;
	float* ligneCalcul;
	float echange;
	float pixel;
	float mini=255,maxi=0;

	//first pass : only on edges :
	uchar* addrPixel;
	uchar* addrPixelTop;
  PILE_CANNY_POP(addrPixel);
  while(!PILE_CANNY_IS_END()){
		int indexPixel=(int)(addrPixel);

		int xLocal=indexPixel%meanTxt->widthStep;
		int yLocal=(int)(indexPixel/meanTxt->widthStep);

		if(yLocal<demiMaskKmean)
			yLocal=demiMaskKmean;
		if(yLocal>=meanTxt->height-demiMaskKmean)
			yLocal=meanTxt->height-demiMaskKmean-1;
		if(xLocal<demiMaskKmean)
			xLocal=demiMaskKmean;
		if(xLocal>=meanTxt->width-demiMaskKmean)
			xLocal=meanTxt->width-demiMaskKmean-1;

		//which pixel in meanT and meanF :
		ligneMeanT=((uchar*)meanTxt->imageData)+yLocal*meanTxt->widthStep+xLocal;//(uchar*)meanTxt->imageData+indexPixel;
		ligneMeanF=((uchar*)meanBack->imageData)+yLocal*meanBack->widthStep+xLocal;//(uchar*)meanBack->imageData+indexPixel;

		//try to find the two classes :
		m=0;
		mini=254;
		maxi=0;
		//top left :
		addrPixelTop=((uchar*)img->imageData)+(yLocal-demiMaskKmean)*img->widthStep+xLocal-demiMaskKmean;
		for(int x=0;x<widthMask;x++){
			for(int y=0;y<widthMask;y++){
				theWindow[m]=(float)addrPixelTop[y];
				if(theWindow[m]>maxi)
					maxi=theWindow[m];
				if(theWindow[m]<mini)
					mini=theWindow[m];
				m++;
			}
			addrPixelTop += img->widthStep;//next line...
		}
		mean[CLASS_BACKGROUND]=maxi;
		mean[CLASS_TEXT]=mini;
		//mean[CLASS_BACKGROUND]=0;
		//mean[CLASS_TEXT]=0;
		//OpImg::segmenteKMean(theWindow,m,mean);
		OpImg::newCenters(theWindow,m,mean);
		OpImg::newCenters(theWindow,m,mean);
		/*
		//mean[CLASS_BACKGROUND] is for the background, so it should be higher than the mean of text...
		if(mean[CLASS_BACKGROUND]<mean[CLASS_TEXT]){
			echange=mean[CLASS_BACKGROUND];
			mean[CLASS_BACKGROUND]=mean[CLASS_TEXT];
			mean[CLASS_TEXT]=echange;
		}
		*/

//		((unsigned char*)(meanBack->imageData+meanTxt->widthStep*i))[j]=mean[CLASS_BACKGROUND];
		//		((unsigned char*)(meanTxt->imageData+meanTxt->widthStep*i))[j]=mean[CLASS_TEXT];
		*ligneMeanT=mean[CLASS_TEXT];
		*ligneMeanF=mean[CLASS_BACKGROUND];
		PILE_CANNY_POP(addrPixel);
	}
	delete theWindow;

	PILE_CANNY_RELOAD();

  cvSet(out,cvScalar(0));
	//second pass : use previously computed means :
	float meanT,meanF;
	int nbmeanT,nbmeanF;
	widthMask=2*demiMaskAnalyse+1;
	PILE_CANNY_POP(addrPixel);
  while(!PILE_CANNY_IS_END()){
    int indexPixel=(int)(addrPixel);
    uchar* edgeVal = (uchar*)edge->imageData+indexPixel;
    bool isTrueEdge = (*edgeVal)>200;


		int xLocal=indexPixel%meanTxt->widthStep-demiMaskAnalyse;
		int yLocal=(int)(indexPixel/meanTxt->widthStep)-demiMaskAnalyse;

		if(yLocal<0)
			yLocal=0;
		if(yLocal>=meanTxt->height-widthMask)
			yLocal=meanTxt->height-widthMask-1;
		if(xLocal<0)
			xLocal=0;
		if(xLocal>=meanTxt->width-widthMask)
			xLocal=meanTxt->width-widthMask-1;

		//which pixel in meanT and meanF (move to top left) :

		ligneMeanT=(uchar*)meanTxt->imageData+	yLocal*meanTxt->widthStep	+xLocal;
		ligneMeanF=(uchar*)meanBack->imageData+	yLocal*meanBack->widthStep	+xLocal;
		ligneOut = (uchar*)out->imageData+		yLocal*out->widthStep		+xLocal;
		ligneImg = (uchar*)img->imageData+		yLocal*img->widthStep		+xLocal;
		ligneDebug=(uchar*)debugImg->imageData+	yLocal*debugImg->widthStep	+xLocal;
		ligneCalcul=(float*)(labelimg->data.fl+yLocal*labelimg->width		+xLocal);

		nbmeanT=0;
		nbmeanF=0;
		meanT=0.0;
		meanF=0.0;

		for(int x=0;x<widthMask;x++){
			for(int y=0;y<widthMask;y++){
				if(ligneMeanT[y]!=254){
					meanT+=ligneMeanT[y];
					meanF+=ligneMeanF[y];
					nbmeanT++;
					nbmeanF++;
				}
			}
			ligneMeanT += meanTxt->widthStep;//next line...
			ligneMeanF += meanBack->widthStep;
		}

		if(nbmeanT!=0)
			meanT/=nbmeanT;
		if(nbmeanF!=0)
			meanF/=nbmeanF;
		

		for(int x=0;x<=widthMask;x++){
			for(int y=0;y<=widthMask;y++){/*
				if(((meanF-ligneImg[y])/7.5)-((ligneImg[y]-meanT)/12.5)>0){//+14 !
					ligneOut[y]=0;
				}else{
					ligneOut[y]=254;
				}*/
				float diffF=meanF-ligneImg[y],diffT=ligneImg[y]-meanT;
				if(diffF<0){
					diffF=0;
				}
				if(diffT<0){
					diffT=0;
				}
				
        if(ligneOut[y]<254)
        {
          float valTmp = 2*sqrt(diffF)-1.5*sqrt(diffT);
          if( isTrueEdge||valTmp>0 )
          {
				    ligneCalcul[y]+=valTmp;
				    ligneOut[y]++;
          }
        }
			}
			ligneOut += out->widthStep;//next line...
			ligneImg += img->widthStep;
			ligneDebug+= debugImg->widthStep;
			ligneCalcul+=labelimg->width;
		}
		
		
		PILE_CANNY_POP(addrPixel);
	}
  

  ligneOut = (uchar*)out->imageData;
  ligneDebug= (uchar*)debugImg->imageData;
  ligneCalcul=labelimg->data.fl;
  for(int x=0;x<img->height;x++){
    for(int y=0;y<img->width;y++){
      if(ligneOut[y]>0){
        if( ligneCalcul[y]!=0.0 )
          ligneCalcul[y] = ligneCalcul[y] / ligneOut[y];
      }
    }
    ligneOut += meanTxt->widthStep;//next line...
    ligneCalcul+=labelimg->width;
    ligneDebug+= meanTxt->widthStep;
  }

  cvSet(out,cvScalar(128));
  ligneOut = (uchar*)out->imageData;
  ligneDebug= (uchar*)debugImg->imageData;
  ligneCalcul=labelimg->data.fl;
  for(int x=0;x<img->height;x++){
    //float *distance1 = &distanceMapTMP.at<float>(x,0);
    for(int y=0;y<img->width;y++){
      if(ligneCalcul[y]!=0){
        //float tmpDistance = sqrt(distance1[y]);

        if(ligneCalcul[y]>0){//tmpDistance-0.4
          ligneOut[y]=0;
        }else{
          ligneOut[y]=254;
        }

        if(ligneCalcul[y]>127)
          ligneDebug[y]=255;
        else
          if(ligneCalcul[y]<-127)
            ligneDebug[y]=0;
          else
            ligneDebug[y]=128+ligneCalcul[y];
      }
    }
    ligneOut += meanTxt->widthStep;//next line...
    ligneCalcul+=labelimg->width;
    ligneDebug+= meanTxt->widthStep;
  }
  /*

  ligneOut = (uchar*)out->imageData;
  ligneDebug= (uchar*)debugImg->imageData;
  ligneCalcul=labelimg->data.fl;
  for(int x=0;x<img->height;x++){
  for(int y=0;y<img->width;y++){
  if(ligneOut[y]>0){
  //ligneCalcul[y]=ligneCalcul[y]/ligneOut[y];//test a completer!!!
  if(abs(ligneCalcul[y])<.5)
  ligneCalcul[y]=0.;
  }
  }
  ligneOut += meanTxt->width;//next line...
  ligneCalcul+=labelimg->width;
  ligneDebug+= meanTxt->width;
  }

  //cvSet(out,cvScalar(128));
  ligneOut = (uchar*)out->imageData;
  ligneDebug= (uchar*)debugImg->imageData;
  ligneCalcul=labelimg->data.fl;
  for(int x=0;x<img->height;x++){
  for(int y=0;y<img->width;y++){
  if(ligneOut[y]>0 && abs(ligneCalcul[y])>.5){//(ligneCalcul[y]!=0){
  float lc=ligneCalcul[y];
  int nbVal=ligneOut[y];
  float diffLocale=(ligneCalcul[y]/ligneOut[y])*4;

  if(diffLocale>126)
  diffLocale=126;
  else
  if(diffLocale<-126)
  diffLocale=126;

  ligneDebug[y]=127+(int)diffLocale;
  if(abs(diffLocale)<5){//+14 !
  ligneOut[y]=128;
  }else{
  if(diffLocale<0)
  ligneOut[y]=254;
  else
  ligneOut[y]=0;
  }
  }else{
  ligneOut[y]=128;
  ligneDebug[y]=128;
  }
  }
  ligneMeanT += meanTxt->widthStep;//next line...
  ligneCalcul+=labelimg->width;
  ligneDebug+= meanTxt->widthStep;
  }*/
  /*
  cvShowImage("scanned",debugImg);
  cvSaveImage("0meanBack.bmp",meanBack);*/
  //cvSaveImage("0meanTxt.bmp",debugImg);

  PILE_CANNY_DELETE();
  //cvReleaseImage(&meanTxt);
  //cvReleaseImage(&meanBack);

  return out;
}

float sqrt2PI = sqrt(2. * CV_PI);

inline float dist_density( float mean, float var, float val, bool text )
{
  float sqrtVar = sqrt(var);
  float dist = val-mean;/*
  if( text )
    dist = MAX( dist, 0 );
  else
    dist = MAX( -dist, 0 );*/
  float expTmp = std::exp( -.5 * dist*dist / var );
  float divi = ( 1./( sqrtVar * sqrt2PI ) );

  return divi * expTmp;
}

typedef struct EM_params
{
  Mat meansText, meansBackground, weights, expect;
  float varText, varBackground;
} EM_params;

inline void computeKmeans( int i_, int j_, int winSize, Mat greyImg, EM_params* params )
{
  int demiMaskKmean = (winSize-1)/2;
  static float *theWindow=new float[winSize*winSize];
  //go to the left top corner:
  int j = (j_<demiMaskKmean) ? 0 : (j_>=greyImg.rows-demiMaskKmean) ? greyImg.rows-winSize-1 : j_-demiMaskKmean;
  int i = (i_<demiMaskKmean) ? 0 : (i_>=greyImg.cols-demiMaskKmean) ? greyImg.cols-winSize-1 : i_-demiMaskKmean;

  float meansTmp[2];
  int m = 0, mini = 255, maxi = 0;
  //compute the means:
  for(int y = 0; y < winSize; y++ )
  {
    uchar* ptrData = greyImg.ptr<uchar>( y+j );
    for( int x = 0; x<winSize; x++ )
    {
      theWindow[m]=(float)ptrData[i+x];
      if(theWindow[m]>maxi)
        maxi=theWindow[m];
      if(theWindow[m]<mini)
        mini=theWindow[m];
      m++;
    }
  }
  meansTmp[CLASS_BACKGROUND]=maxi;
  meansTmp[CLASS_TEXT]=mini;

  OpImg::newCenters(theWindow,m,meansTmp);
  OpImg::newCenters(theWindow,m,meansTmp);

  params->meansText.at<float>( j_, i_ ) = meansTmp[CLASS_TEXT];
  params->meansBackground.at<float>( j_, i_ ) = meansTmp[CLASS_BACKGROUND];
}

inline void computeExpectation( int i_, int j_, int winSize, Mat greyImg, EM_params* params )
{
  float probaText = ( params->weights.at<float>( j_, i_ ) ) * 
    dist_density( params->meansText.at<float>( j_, i_ ), params->varText, 
    greyImg.at<uchar>( j_, i_ ) , true );
  float probaBack = ( 1.0 - params->weights.at<float>( j_, i_ ) ) * dist_density( params->meansBackground.at<float>( j_, i_ ), params->varBackground, 
    greyImg.at<uchar>( j_, i_ ), false );

  params->expect.at<float>( j_, i_ ) = probaText / ( probaText + probaBack );
}

void computeExpectation( Mat greyImg, EM_params* params, PileCanny* p )
{
  uchar widthMask=7;

  //process the image:
  //first pass : only on edges :
  params->expect = Scalar ( -1. );
  uchar* addrPixel;
  uchar* addrPixelTop;
  PILE_CANNY_POP(addrPixel);
  while(!PILE_CANNY_IS_END()){
    int indexPixel=(int)(addrPixel);

    int xLocal=indexPixel%greyImg.step;
    int yLocal=(int)(indexPixel/greyImg.step);

    computeExpectation( xLocal, yLocal, widthMask, greyImg, params );
    PILE_CANNY_POP(addrPixel);
  }
  PILE_CANNY_RELOAD();
}

inline void computeMaximisation( int i_, int j_, int winSize, Mat greyImg, EM_params* params )
{
  int demiMaskKmean = (winSize-1)/2;
  static float *theWindow=new float[winSize*winSize];
  //go to the left top corner:
  int j = (j_<demiMaskKmean) ? 0 : (j_>=greyImg.rows-demiMaskKmean) ? greyImg.rows-winSize-1 : j_-demiMaskKmean;
  int i = (i_<demiMaskKmean) ? 0 : (i_>=greyImg.cols-demiMaskKmean) ? greyImg.cols-winSize-1 : i_-demiMaskKmean;

  float somme = 0, nbVals = 0;
  float meanT = 0, meanF = 0.;
  //compute the weight & means:
  for(int y = 0; y < winSize; y++ )
  {
    for( int x = 0; x<winSize; x++ )
    {
      float expectation = params->expect.at<float>( j+y, i+x );
      if(expectation>=0)
      {
        somme+=expectation;
        nbVals++;

        meanT += expectation*greyImg.at<uchar>( j+y, i+x );
        meanF += (1. - expectation)*greyImg.at<uchar>( j+y, i+x );
      }
    }
  }
  params->weights.at<float>( j_, i_) = somme/nbVals;

  meanT /= somme;
  meanF /= nbVals - somme;

  params->meansText.at<float>( j_, i_ ) = meanT;
  params->meansBackground.at<float>( j_, i_ ) = meanF;

}

void computeMaximisation( Mat greyImg, EM_params* params, PileCanny* p )
{
  uchar widthMask=7;

  //process the image:
  //first pass : only on edges :
  uchar* addrPixel;
  uchar* addrPixelTop;
  PILE_CANNY_POP(addrPixel);
  while(!PILE_CANNY_IS_END()){
    int indexPixel=(int)(addrPixel);

    int xLocal=indexPixel%greyImg.step;
    int yLocal=(int)(indexPixel/greyImg.step);

    computeMaximisation( xLocal, yLocal, widthMask, greyImg, params );
    PILE_CANNY_POP(addrPixel);
  }
  PILE_CANNY_RELOAD();
}

void runEM_once( Mat greyImage, PileCanny* p, EM_params* params )
{
    
  computeExpectation( greyImage, params, p );
  computeMaximisation( greyImage, params, p );

  PILE_CANNY_RELOAD();

  //now compute the variance:
  float nbVal = 0;
  uchar* addrPixel;
  PILE_CANNY_POP(addrPixel);
  while(!PILE_CANNY_IS_END()){
    int indexPixel=(int)(addrPixel);

    int xLocal=indexPixel%greyImage.step;
    int yLocal=(int)(indexPixel/greyImage.step);


    uchar tmpGreyVal = greyImage.at<uchar>( yLocal, xLocal );

    float tmp = (tmpGreyVal - params->meansText.at<float>( yLocal, xLocal ) );
    params->varText += tmp*tmp;
    tmp = (tmpGreyVal - params->meansBackground.at<float>( yLocal, xLocal ) );
    params->varBackground += tmp*tmp;
    nbVal++;
    PILE_CANNY_POP(addrPixel);
  }
  PILE_CANNY_RELOAD();

  params->varText = ( params->varText/nbVal );
  params->varBackground = ( params->varBackground/nbVal );
}

void runEM_Modif_local( int i_, int j_, int winSize, Mat greyImg, EM_params* params )
{
	
  int demiMaskKmean = (winSize-1)/2;
  int nsamples = winSize*winSize;
  Mat samples = Mat::zeros(nsamples, 1, CV_32FC1);
  float *theWindow = samples.ptr<float>( 0 );
  //go to the left top corner:
  int j = (j_<demiMaskKmean) ? 0 : (j_>=greyImg.rows-demiMaskKmean) ? greyImg.rows-winSize-1 : j_-demiMaskKmean;
  int i = (i_<demiMaskKmean) ? 0 : (i_>=greyImg.cols-demiMaskKmean) ? greyImg.cols-winSize-1 : i_-demiMaskKmean;

  float meansTmp[2];
  int m = 0, mini = 255, maxi = 0;
  //compute the means:
  for(int y = 0; y < winSize; y++ )
  {
    uchar* ptrData = greyImg.ptr<uchar>( y+j );
    for( int x = 0; x<winSize; x++ )
    {
      theWindow[m] = ptrData[i+x];
      if(theWindow[m]>maxi)
        maxi=theWindow[m];
      if(theWindow[m]<mini)
        mini=theWindow[m];
      m++;
    }
  }
  meansTmp[CLASS_BACKGROUND]=maxi;
  meansTmp[CLASS_TEXT]=mini;
  
  EM_modif em_model( 2, 1, params->varText, params->varBackground, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 1, FLT_EPSILON) );
	if (!em_model.train(samples)) {
		cerr << "error training the EM model" << endl;
		exit(-1);
	}

  const Mat& means = em_model.means;
  float mean1 = means.at<float>(0, 0);
  float mean2 = means.at<float>(1, 0);

  const Mat& weights = em_model.weights;
  float weight1 = weights.at<float>(0, 0);
  float weight2 = weights.at<float>(0, 1);
  if( mean1>mean2 )
  {
    float exchange = mean1;
    cout<<"swap"<<endl;
    CV_SWAP( mean1, mean2, exchange );
    CV_SWAP( weight1, weight2, exchange );
  }
  /*
	cout << "mean1 = " << mean1 << ", mean2 = " << mean2 << endl;
	cout << "scale1 = " << scale1 << ", scale2 = " << scale2 << endl;
   */

  params->meansText.at<float>( j_, i_ ) = mean1;
  params->meansBackground.at<float>( j_, i_ ) = mean2;

  params->weights.at<float>( j_, i_ ) = weight1;

  computeExpectation( i_, j_, winSize, greyImg, params );
}

void runEM_classic_local( int i_, int j_, int winSize, Mat greyImg, EM_params* params )
{
	
  int demiMaskKmean = (winSize-1)/2;
  int nsamples = winSize*winSize;
  Mat samples = Mat::zeros(nsamples, 1, CV_32FC1);
  float *theWindow = samples.ptr<float>( 0 );
  //go to the left top corner:
  int j = (j_<demiMaskKmean) ? 0 : (j_>=greyImg.rows-demiMaskKmean) ? greyImg.rows-winSize-1 : j_-demiMaskKmean;
  int i = (i_<demiMaskKmean) ? 0 : (i_>=greyImg.cols-demiMaskKmean) ? greyImg.cols-winSize-1 : i_-demiMaskKmean;

  float meansTmp[2];
  int m = 0, mini = 255, maxi = 0;
  //compute the means:
  for(int y = 0; y < winSize; y++ )
  {
    uchar* ptrData = greyImg.ptr<uchar>( y+j );
    for( int x = 0; x<winSize; x++ )
    {
      theWindow[m] = ptrData[i+x];
      if(theWindow[m]>maxi)
        maxi=theWindow[m];
      if(theWindow[m]<mini)
        mini=theWindow[m];
      m++;
    }
  }
  meansTmp[CLASS_BACKGROUND]=maxi;
  meansTmp[CLASS_TEXT]=mini;
  
  cv::EM em_model( 2, 1,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 4, FLT_EPSILON) );
	if (!em_model.train(samples)) {
		cerr << "error training the EM model" << endl;
		exit(-1);
	}

  const Mat& means = em_model.get<Mat>("means");
  int mean1 = means.at<float>(0, 0);
  int mean2 = means.at<float>(1, 0);
  
  const Mat& weights = em_model.get<Mat>("weights");
  float weight1 = weights.at<float>(0, 0);
  float weight2 = weights.at<float>(0, 1);
  if( mean1>mean2 )
  {
    float exchange = mean1;
    CV_SWAP( mean1, mean2, exchange );
    CV_SWAP( weight1, weight2, exchange );
  }
  /*
	cout << "mean1 = " << mean1 << ", mean2 = " << mean2 << endl;
	cout << "scale1 = " << scale1 << ", scale2 = " << scale2 << endl;
   */

  params->meansText.at<float>( j_, i_ ) = mean1;
  params->meansBackground.at<float>( j_, i_ ) = mean2;

  params->weights.at<float>( j_, i_ ) = weight1;

  computeExpectation( i_, j_, winSize, greyImg, params );
}

EM_params* initParams( Mat greyImage, PileCanny* p, int demiMaskKmean=3 );
EM_params* runEM_classic( Mat greyImage, PileCanny* p )
{
  //First init the params:
  EM_params *out = initParams( greyImage, p );
  //the temporary vector:
  uchar widthMask=7;

  cout<<out->varBackground<<endl;
  cout<<out->varText<<endl;

  //process the image:
  //first pass : only on edges :
  uchar* addrPixel;
  uchar* addrPixelTop;
  PILE_CANNY_POP(addrPixel);
  while(!PILE_CANNY_IS_END()){
    int indexPixel=(int)(addrPixel);

    int xLocal=indexPixel%greyImage.step;
    int yLocal=(int)(indexPixel/greyImage.step);

    runEM_classic_local( xLocal, yLocal, widthMask, greyImage, out );
    PILE_CANNY_POP(addrPixel);
  }
  PILE_CANNY_RELOAD();


  float sommeB = 0, sommeT = 0;
  //now compute the variance:
  PILE_CANNY_POP(addrPixel);
  while(!PILE_CANNY_IS_END()){
    int indexPixel=(int)(addrPixel);

    int xLocal=indexPixel%greyImage.step;
    int yLocal=(int)(indexPixel/greyImage.step);

    float expectation = out->expect.at<float>( yLocal, xLocal );
    if(expectation>=0)
    {
      sommeT+=expectation;
      sommeB+=(1.-expectation);

      float tmp = (*addrPixel - out->meansText.at<float>( yLocal, xLocal ) );
      out->varText += expectation*tmp*tmp;
      tmp = (*addrPixel - out->meansBackground.at<float>( yLocal, xLocal ) );
      out->varBackground += (1.-expectation)*tmp*tmp;
    }

    PILE_CANNY_POP(addrPixel);
  }
  PILE_CANNY_RELOAD();

  out->varText = ( out->varText/sommeT );
  out->varBackground = ( out->varBackground/sommeB );
  return out;
}
EM_params* runEM_Modif( Mat greyImage, PileCanny* p )
{
  //First init the params:
  EM_params *out = initParams( greyImage, p );
  //the temporary vector:
  uchar widthMask=7;

  for(int i=0; i<3; i++ )
  {
    //cout<<out->varBackground<<endl;
    //cout<<out->varText<<endl;

    //process the image:
    //first pass : only on edges :
    uchar* addrPixel;
    uchar* addrPixelTop;
    PILE_CANNY_POP(addrPixel);
    while(!PILE_CANNY_IS_END()){
      int indexPixel=(int)(addrPixel);

      int xLocal=indexPixel%greyImage.step;
      int yLocal=(int)(indexPixel/greyImage.step);

      runEM_Modif_local( xLocal, yLocal, widthMask, greyImage, out );
      PILE_CANNY_POP(addrPixel);
    }
    PILE_CANNY_RELOAD();

    float sommeB = 0, sommeT = 0;
    //now compute the variance:
    PILE_CANNY_POP(addrPixel);
    while(!PILE_CANNY_IS_END()){
      int indexPixel=(int)(addrPixel);

      int xLocal=indexPixel%greyImage.step;
      int yLocal=(int)(indexPixel/greyImage.step);

      float expectation = out->expect.at<float>( yLocal, xLocal );
      if(expectation>=0)
      {
        sommeT+=expectation;
        sommeB+=(1.-expectation);

        uchar tmpGreyVal = greyImage.at<uchar>( yLocal, xLocal );

        float tmp = (tmpGreyVal - out->meansText.at<float>( yLocal, xLocal ) );
        out->varText += expectation*tmp*tmp;
        tmp = (tmpGreyVal - out->meansBackground.at<float>( yLocal, xLocal ) );
        out->varBackground += (1.-expectation)*tmp*tmp;
      }

      PILE_CANNY_POP(addrPixel);
    }
    PILE_CANNY_RELOAD();

    out->varText = ( out->varText/sommeT );
    out->varBackground = ( out->varBackground/sommeB );
  }

  return out;
}
EM_params* initParams( Mat greyImage, PileCanny* p, int demiMaskKmean )
{
  //First create the params:
  EM_params *out = new EM_params;
  out->meansText = Mat( greyImage.size(), CV_32FC1 );
  out->meansBackground = Mat( greyImage.size(), CV_32FC1 );
  out->weights = Mat( greyImage.size(), CV_32FC1 );
  out->expect = Mat( greyImage.size(), CV_32FC1 );
  
  //the temporary vector:
  uchar widthMask=demiMaskKmean*2+1;

  //init:
  out->meansText = Scalar( -1. );
  out->meansBackground = Scalar( -1. );
  out->weights = Scalar( 0.5 );
  out->expect = Scalar( -1. );
  out->varBackground = out->varText = 0.;

  //process the image:
  //first pass : only on edges :
  uchar* addrPixel;
  PILE_CANNY_POP(addrPixel);
  int idT = 0;
  while(!PILE_CANNY_IS_END()){
    idT++;
    int indexPixel=(int)addrPixel;//(addrPixel-(uchar*)(greyImage.ptr(0)));

    int xLocal=indexPixel%greyImage.step;
    int yLocal=(int)(indexPixel/greyImage.step);

    computeKmeans( xLocal, yLocal, widthMask, greyImage, out );
    PILE_CANNY_POP(addrPixel);
  }
  PILE_CANNY_RELOAD();

  //now compute the variance:
  float nbVal = 0;
  PILE_CANNY_POP(addrPixel);
  while(!PILE_CANNY_IS_END()){
    int indexPixel=(int)addrPixel;

    int xLocal=indexPixel%greyImage.step;
    int yLocal=(int)(indexPixel/greyImage.step);

    uchar tmpGreyVal = greyImage.at<uchar>( yLocal, xLocal );

    float tmp = (tmpGreyVal - out->meansText.at<float>( yLocal, xLocal ) );
    out->varText += tmp*tmp;
    tmp = (tmpGreyVal - out->meansBackground.at<float>( yLocal, xLocal ) );
    out->varBackground += tmp*tmp;
    nbVal++;
    PILE_CANNY_POP(addrPixel);
  }
  PILE_CANNY_RELOAD();

  out->varText = ( out->varText/nbVal );
  out->varBackground = ( out->varBackground/nbVal );

  return out;
};

IplImage *OpImg::expandEdgeEM(IplImage *img,PileCanny* p,IplImage *e,int demiMaskKmean){

  EM_params* em_params = initParams( img, p );//KMeans
  //EM_params* em_params = runEM_Modif( img, p );
  //EM_params* em_params = runEM_classic( img, p );

  em_params->expect.release();
  em_params->weights.release();

  char demiMaskAnalyse=demiMaskKmean;
  edge=e;
  IplImage *out= cvCreateImage(cvSize(img->width,img->height),8,1);
  if(debugImg==NULL){
    debugImg= cvCreateImage(cvSize(img->width,img->height),8,1);
    labelimg=cvCreateMat(img->height,img->width,CV_32FC1);
  }
  if( debugImg->height!=img->height )
  {
    cvReleaseImage(&debugImg);
    cvReleaseMat(&labelimg);
    debugImg= cvCreateImage(cvSize(img->width,img->height),8,1);
    labelimg=cvCreateMat(img->height,img->width,CV_32FC1);
  }

  cvSet(debugImg,cvScalar(128));
  cvSet(labelimg,cvScalar(0));
  cvSet(out,cvScalar(0));


  //second pass : use previously computed means :
  float meanT,meanF;
  int nbmeanT,nbmeanF;
  int widthMask=2*demiMaskAnalyse+1;
  uchar* addrPixel;
  PILE_CANNY_POP(addrPixel);
  while(!PILE_CANNY_IS_END()){
    int indexPixel=(int)addrPixel;

    int xLocal=indexPixel%img->widthStep;
    int yLocal=(int)(indexPixel/img->widthStep);

    meanT=em_params->meansText.at<float>( yLocal, xLocal);
    meanF=em_params->meansBackground.at<float>( yLocal, xLocal);

    bool isFull = *((uchar*)edge->imageData+		yLocal*edge->widthStep		+xLocal)>150;

    yLocal = (yLocal<demiMaskAnalyse) ? yLocal=0 : (yLocal>=img->height-demiMaskAnalyse-1) ? img->height-widthMask-1 : yLocal - demiMaskAnalyse;
    xLocal = (xLocal<demiMaskAnalyse) ? xLocal=0 : (xLocal>=img->width-demiMaskAnalyse-1) ? img->width-widthMask-1 : xLocal - demiMaskAnalyse;
    uchar* ligneImg = (uchar*)img->imageData+		yLocal*img->widthStep		+xLocal;
    float* ligneCalcul=(float*)(labelimg->data.fl+yLocal*labelimg->width		+xLocal);
    uchar* ligneOut = (uchar*)out->imageData+		yLocal*out->widthStep		+xLocal;

    ////////////////////////////////////////////////////////////////////
    for(int x=0;x<=widthMask;x++){
      for(int y=0;y<=widthMask;y++){

        float diffF=dist_density(meanF, em_params->varBackground, ligneImg[y], false);
        float diffT=dist_density(meanT, em_params->varText, ligneImg[y], true);

        if(ligneOut[y]<254)
        {
          if(isFull)
          {
            //ligneCalcul[y]+=(diffF)-.0*(diffT);//10*sqrt(diffF)-4*sqrt(diffT);//
            ligneCalcul[y] += (diffT - diffF);
            ligneOut[y]++;
          }
          else
          {
            if( (diffT - diffF)>0 )
            {
              //ligneCalcul[y]+=(diffF)-.0*(diffT);//10*sqrt(diffF)-4*sqrt(diffT);//
              ligneCalcul[y] += (diffT - diffF);
              ligneOut[y]++;
            }
          }
        }
      }
      ligneOut += out->widthStep;//next line...
      ligneImg += img->widthStep;
      ligneCalcul+=labelimg->width;
    }

    PILE_CANNY_POP(addrPixel);
  }
  
  uchar* ligneOut = (uchar*)out->imageData;
  float* ligneCalcul=labelimg->data.fl;
  for(int x=0;x<img->height;x++){
    for(int y=0;y<img->width;y++){
      if(ligneOut[y]>0){
        //ligneCalcul[y]=ligneCalcul[y]/ligneOut[y];//test a completer!!!
        ligneCalcul[y] = ligneCalcul[y] / ligneOut[y];
      }
    }
    ligneOut += out->widthStep;//next line...
    ligneCalcul+=labelimg->width;
  }

  cvSet(out,cvScalar(128));
  cvSet(debugImg,cvScalar(128));
  ligneOut = (uchar*)out->imageData;
  uchar* ligneDebug= (uchar*)debugImg->imageData;
  ligneCalcul=labelimg->data.fl;
  for(int x=0;x<img->height;x++){
    for(int y=0;y<img->width;y++){
      if(ligneCalcul[y]!=0){
        ligneCalcul[y] = ligneCalcul[y]*1000;
        if(ligneCalcul[y]>0){//+14 !
          ligneOut[y]=0;
        }else{
          ligneOut[y]=254;
        }

        if(ligneCalcul[y]>127)
          ligneDebug[y]=255;
        else
          if(ligneCalcul[y]<-127)
            ligneDebug[y]=0;
          else
            ligneDebug[y]=128+ligneCalcul[y];
      }
    }
    ligneOut += out->widthStep;//next line...
    ligneCalcul+=labelimg->width;
    ligneDebug+= debugImg->widthStep;
  }
  /*
  cv::imwrite( "0Back.bmp", testBack );
  cv::imwrite( "0Text.bmp", testText );
  cv::imwrite( "0Diff.bmp", Mat(debugImg) );*/
  //cout<<"Variances : Text : "<<em_params->varText<<" Back : "<<em_params->varBackground<<endl;

  PILE_CANNY_DELETE();
  //cvReleaseImage(&meanTxt);
  //cvReleaseImage(&meanBack);
  delete em_params;
  return out;
}


//fonction qui permet de définir la classe des pixels proche des contours
//out : image ternaire où:
//0 correspond à du texte
//254 correspond à du fond
//128 correspond à une classe indeterminée
IplImage *OpImg::expandEdge(IplImage *img,PileCanny *p,IplImage *e,int demiMaskKmean,char demiMaskAnalyse){
  edge=e;
  IplImage *out= cvCreateImage(cvSize(img->width,img->height),8,1);
  IplImage *meanTxt= cvCreateImage(cvSize(img->width,img->height),8,1);
  IplImage *meanBack= cvCreateImage(cvSize(img->width,img->height),8,1);
  cvSet(out,cvScalar(128));
  cvSet(meanTxt,cvScalar(254));
  cvSet(meanBack,cvScalar(0));

  float *theWindow=new float[(demiMaskKmean*2+1)*(demiMaskKmean*2+1)];
  float mean[2];
  mean[0]=0;
  mean[1]=0;
  int m;

  unsigned char* ligneImg;
  unsigned char* ligneEdge;
  unsigned char* ligneOut;
  unsigned char* ligneMeanT;
  unsigned char* ligneMeanF;
  float echange;
  float pixel;
  float mini=255,maxi=0;

  //first pass : only on edges :
  int i,j;
  unsigned char *addrPixel;
  PILE_CANNY_POP(addrPixel);
  while(!PILE_CANNY_IS_END()){
    int indexPixel=(int)(addrPixel);

    j = indexPixel%meanTxt->widthStep;
    i = (int)(indexPixel/meanTxt->widthStep);

    if(i-demiMaskKmean<0) i=demiMaskKmean;
    if(i+demiMaskKmean>=out->height)
      i=out->height-demiMaskKmean-1;
    if(j-demiMaskKmean<0) j=demiMaskKmean;
    if(j+demiMaskKmean>=out->width)
      j=out->width-demiMaskKmean-1;

    //try to find the two classes :
    m=0;
    ligneImg = (unsigned char*)(img->imageData+img->widthStep*(i-demiMaskKmean));
    mini=254;
    maxi=0;
    for(int x=-demiMaskKmean;x<=demiMaskKmean;x++){
      for(int y=j-demiMaskKmean;y<=j+demiMaskKmean;y++){
        theWindow[m]=ligneImg[y];
        if(theWindow[m]>maxi)
          maxi=theWindow[m];
        if(theWindow[m]<mini)
          mini=theWindow[m];
        m++;
      }
      ligneImg += img->widthStep;//newt line...
    }
    mean[CLASS_BACKGROUND]=maxi;
    mean[CLASS_TEXT]=mini;
    //mean[CLASS_BACKGROUND]=0;
    //mean[CLASS_TEXT]=0;
    //OpImg::segmenteKMean(theWindow,m,mean);
    OpImg::newCenters(theWindow,m,mean);
    OpImg::newCenters(theWindow,m,mean);

    //mean[CLASS_BACKGROUND] is for the background, so it should be higher than the mean of text...
    //if(mean[CLASS_BACKGROUND]<mean[CLASS_TEXT]){
    //	echange=mean[CLASS_BACKGROUND];
    //	mean[CLASS_BACKGROUND]=mean[CLASS_TEXT];
    //	mean[CLASS_TEXT]=echange;
    //}


    ((unsigned char*)(meanBack->imageData+meanTxt->widthStep*i))[j]=mean[CLASS_BACKGROUND];
    ((unsigned char*)(meanTxt->imageData+meanTxt->widthStep*i))[j]=mean[CLASS_TEXT];
    PILE_CANNY_POP( addrPixel );
  }
  delete theWindow;

  PILE_CANNY_RELOAD();

  //second pass : use previously computed means :
  float meanT,meanF;
  int nbmeanT,nbmeanF;
  PILE_CANNY_POP(addrPixel);
  while(!PILE_CANNY_IS_END()){
    int indexPixel=(int)(addrPixel);

    j = indexPixel%meanTxt->widthStep;
    i = (int)(indexPixel/meanTxt->widthStep);

    //petite boucle interieur :
    nbmeanT=0;
    nbmeanF=0;
    meanT=0.0;
    meanF=0.0;

    if(i-demiMaskAnalyse<0) i=demiMaskAnalyse;
    if(i+demiMaskAnalyse>=out->height)
      i=out->height-demiMaskAnalyse-1;
    if(j-demiMaskAnalyse<0) j=demiMaskAnalyse;
    if(j+demiMaskAnalyse>=out->width)
      j=out->width-demiMaskAnalyse-1;

    ligneMeanT = (unsigned char*)(meanTxt->imageData+meanTxt->widthStep*(i-demiMaskAnalyse));
    ligneMeanF = (unsigned char*)(meanBack->imageData+meanBack->widthStep*(i-demiMaskAnalyse));
    for(int x=-demiMaskAnalyse;x<=demiMaskAnalyse;x++){
      for(int y=j-demiMaskAnalyse;y<=j+demiMaskAnalyse;y++){
        if(ligneMeanT[y]!=254){
          meanT+=ligneMeanT[y];
          meanF+=ligneMeanF[y];
          nbmeanT++;
          nbmeanF++;
        }
      }
      ligneMeanT += meanTxt->widthStep;
      ligneMeanF += meanBack->widthStep;
    }

    meanT/=nbmeanT;
    meanF/=nbmeanF;


    for(int x=i-demiMaskAnalyse;x<=i+demiMaskAnalyse;x++){
      ligneOut = (unsigned char*)(out->imageData+img->widthStep*x);
      ligneImg = (unsigned char*)(img->imageData+img->widthStep*x);
      for(int y=j-demiMaskAnalyse;y<=j+demiMaskAnalyse;y++){
        if(ligneOut[y]==128){
          pixel=(unsigned char)ligneImg[y];
          if((pixel-meanT)/12.5<(meanF-pixel)/7.5){//+14 !
            ligneOut[y]=0;
          }else{
            ligneOut[y]=254;
          }
        }
      }
      PILE_CANNY_POP( addrPixel );

    }
  }

  PILE_CANNY_DELETE();
  cvReleaseImage(&meanTxt);
  cvReleaseImage(&meanBack);
  return out;
}


/*
//fonction qui permet de définir la classe des pixels proche des contours
//out : image ternaire où:
//0 correspond à du texte
//254 correspond à du fond
//128 correspond à une classe indeterminée
IplImage *OpImg::expandEdge(IplImage *i,IplImage *e,int demiMask,char marge=4){
img=i;
edge=e;
IplImage *out= cvCreateImage(cvSize(img->width,img->height),8,1);
cvSet(out,cvScalar(128));

float *theWindow=new float[(demiMask*2+1)*(demiMask*2+1)];
float mean[2];
mean[0]=0;
mean[1]=0;
int m;

unsigned char* ligneImg;
unsigned char* ligneOut;
float echange;
float pixel;

for(int i=0;i<img->height;i++){
ligneOut = (unsigned char*)(out->imageData+img->widthStep*i);
for(int j=0;j<img->width;j++){
if(proximaEdge(edge,i,j,marge)){//if an edge is close to i,j
//try to find the two classes :
m=0;
for(int x=i-demiMask;x<=i+demiMask;x++){
ligneImg = (unsigned char*)(img->imageData+img->widthStep*x);
if(x>=0&&x<img->height)
for(int y=j-demiMask;y<=j+demiMask;y++){
if(y>=0&&y<img->width){
theWindow[m]=ligneImg[y];
m++;
}
}
}
mean[0]=0;
mean[1]=0;
OpImg::segmenteKMean(theWindow,m,mean);
//mean[CLASS_BACKGROUND] is for the background, so it should be higher than the mean of text...
if(mean[CLASS_BACKGROUND]<mean[CLASS_TEXT]){
echange=mean[CLASS_BACKGROUND];
mean[CLASS_BACKGROUND]=mean[CLASS_TEXT];
mean[CLASS_TEXT]=echange;
}
//some time, there is only one class...
if(mean[CLASS_BACKGROUND]<0){
mean[CLASS_BACKGROUND]=254;
mean[CLASS_TEXT]=0;
}
//if the means have correct values :
if((mean[CLASS_BACKGROUND]>mean[CLASS_TEXT]+16.0)&&mean[CLASS_BACKGROUND]<253.0&&mean[CLASS_TEXT]>1.0){
pixel=(float)CV_GetReal2D(img,i,j);
if(fabs(mean[CLASS_BACKGROUND]-pixel)+8.0<fabs(mean[CLASS_TEXT]-pixel)){
ligneOut[j]=254;
}else{
ligneOut[j]=0;
}
}else{
ligneOut[j]=defineClass(edge,i,j);
if(ligneOut[j]==128)
ligneOut[j]=defineClassOnEdge(out,img,i,j);
mean[0]=0;
mean[1]=0;
}
}
}
}
return out;
}

//fonction qui permet de définir la classe des pixels proche des contours
//out : image ternaire où:
//0 correspond à du texte
//254 correspond à du fond
//128 correspond à une classe indeterminée
IplImage *OpImg::expandEdge(IplImage *i,IplImage *e,int demiMask,char marge=8){
img=i;
edge=e;
IplImage *out= cvCreateImage(cvSize(img->width,img->height),8,1);
cvSet(out,cvScalar(128));

unsigned char* ligneImg;
unsigned char* ligneOutP;
unsigned char* ligneOut;
float echange;
float pixel;

//colorie autour des contours
for(int i=1;i<img->height-1;i++){
ligneOut = (unsigned char*)(out->imageData+img->widthStep*i);
for(int j=1;j<img->width-1;j++){
if(CV_GetReal2D(edge,i,j)==0){
ligneOut[j]=defineClass(edge,i,j);
}
}
}

//colorie les contours
for(int i=0;i<img->height;i++){
ligneOut = (unsigned char*)(out->imageData+img->widthStep*i);
for(int j=0;j<img->width;j++){
if(CV_GetReal2D(edge,i,j)!=0){
ligneOut[j]=defineClassOnEdge(out,img,i,j);
}
}
}

IplImage *test= cvCreateImage(cvSize(img->width,img->height),8,1);
unsigned char* ligneTest;
cvCopy(out,test);

//augmente encore un peu autour des bords :
int nb=1;
int cptVal;
int val;
while(nb<marge){

for(int i=nb;i<img->height-nb;i++){
ligneTest = (unsigned char*)(test->imageData+img->widthStep*i);
ligneOut = (unsigned char*)(out->imageData+img->widthStep*i);
ligneOutP = (unsigned char*)(out->imageData+img->widthStep*(i-1));
ligneImg = (unsigned char*)(out->imageData+img->widthStep*(i+1));//c'est maintenant la ligne suivante..
for(int j=nb;j<img->width-nb;j++){
if(ligneOut[j]==128){//&&proximaEdge(edge,i,j,nb+1)){
cptVal=0;
val=0;
if(ligneOutP[j-1]!=128){val+=ligneOutP[j-1];cptVal++;};
if(ligneOutP[j]!=128){val+=ligneOutP[j];cptVal++;};
if(ligneOutP[j+1]!=128){val+=ligneOutP[j+1];cptVal++;};
if(ligneImg[j-1]!=128){val+=ligneImg[j-1];cptVal++;};
if(ligneImg[j]!=128){val+=ligneImg[j];cptVal++;};
if(ligneImg[j+1]!=128){val+=ligneImg[j+1];cptVal++;};
if(ligneOut[j-1]!=128){val+=ligneOut[j-1];cptVal++;};
if(ligneOut[j+1]!=128){val+=ligneOut[j+1];cptVal++;};
if(cptVal>1){
val=val/cptVal;
if(val>200){
ligneTest[j]=254;
}else{
if(val<64)
ligneTest[j]=defineClassOnEdge(out,img,i,j);
}
}
}
}
}

cvCopy(test,out);
nb++;
}

return out;
}
*/

//fonction qui permet de définir la classe des pixels proche des contours
//out : image ternaire où:
//0 correspond à du texte
//254 correspond à du fond
//128 correspond à une classe indeterminée
IplImage *OpImg::SeuilSauvolaOptimiz(IplImage *gray_image,int whalf,float k=0.2){
  IplImage *out= cvCreateImage(cvSize(gray_image->width,gray_image->height),8,1);

  int w=whalf*2+1;
  // fprintf(stderr,"[sauvola %g %d]\n",k,w);
  if(!(k>=0.05 && k<=0.95))
    k=0.15;
  if(!(w>0 && k<1000))
    w=15;

  int image_width  = gray_image->height;
  int image_height = gray_image->width;

  // Calculate the integral image, and integral of the squared image
  CvMat* integral_image,*rowsum_image;
  CvMat* integral_sqimg,*rowsum_sqimg;

  int xmin,ymin,xmax,ymax;
  float diagsum,idiagsum,diff,sqdiagsum,sqidiagsum,sqdiff,area;
  float mean,std,threshold;

  integral_image=cvCreateMat(gray_image->height,gray_image->width,CV_32FC1);
  integral_sqimg=cvCreateMat(gray_image->height,gray_image->width,CV_32FC1);
  int matStep=integral_image->step/(sizeof(float));

  rowsum_sqimg=cvCreateMat(gray_image->height,gray_image->width,CV_32FC1);
  rowsum_image=cvCreateMat(gray_image->height,gray_image->width,CV_32FC1);


  for(int j=0; j<image_height; j++){
    CV_SetReal2DMatD(rowsum_image,0,j,CV_GetReal2D(gray_image,0,j));
    CV_SetReal2DMatD(rowsum_sqimg,0,j,CV_GetReal2D(gray_image,0,j)*CV_GetReal2D(gray_image,0,j));
  }
  for(int i=1; i<image_width; i++){
    for(int j=0; j<image_height; j++){
      CV_SetReal2DMatD(rowsum_image,i,j,CV_GetReal2DMatD(rowsum_image,i-1,j)+CV_GetReal2D(gray_image,i,j));
      CV_SetReal2DMatD(rowsum_sqimg,i,j,CV_GetReal2DMatD(rowsum_sqimg,i-1,j)+CV_GetReal2D(gray_image,i,j)*CV_GetReal2D(gray_image,i,j));
    }
  }

  for(int i=0; i<image_width; i++){
    CV_SetReal2DMatD(integral_image,i,0,CV_GetReal2DMatD(rowsum_image,i,0));
    CV_SetReal2DMatD(integral_sqimg,i,0,CV_GetReal2DMatD(rowsum_sqimg,i,0));
  }
  for(int i=0; i<image_width; i++){
    for(int j=1; j<image_height; j++){
      CV_SetReal2DMatD(integral_image,i,j,CV_GetReal2DMatD(integral_image,i,j-1)+CV_GetReal2DMatD(rowsum_image,i,j));
      CV_SetReal2DMatD(integral_sqimg,i,j,CV_GetReal2DMatD(integral_sqimg,i,j-1)+CV_GetReal2DMatD(rowsum_sqimg,i,j));
    }
  }
  cvReleaseMat(&rowsum_image);
  cvReleaseMat(&rowsum_sqimg);


  //Calculate the mean and standard deviation using the integral image

  for(int i=0; i<image_width; i++){
    for(int j=0; j<image_height; j++){
      xmin = max(0,i-whalf);
      ymin = max(0,j-whalf);
      xmax = min(image_width-1,i+whalf);
      ymax = min(image_height-1,j+whalf);
      area = (xmax-xmin+1)*(ymax-ymin+1);
      // area can't be 0 here
      // proof (assuming whalf >= 0):
      // we'll prove that (xmax-xmin+1) > 0,
      // (ymax-ymin+1) is analogous
      // It's the same as to prove: xmax >= xmin
      // image_width - 1 >= 0         since image_width > i >= 0
      // i + whalf >= 0               since i >= 0, whalf >= 0
      // i + whalf >= i - whalf       since whalf >= 0
      // image_width - 1 >= i - whalf since image_width > i
      // --IM
      if(!xmin && !ymin){ // Point at origin
        diff   = CV_GetReal2DMatD(integral_image,xmax,ymax);
        sqdiff = CV_GetReal2DMatD(integral_sqimg,xmax,ymax);
      }
      else if(!xmin && ymin){ // first column
        diff   = CV_GetReal2DMatD(integral_image,xmax,ymax) - CV_GetReal2DMatD(integral_image,xmax,ymin-1);
        sqdiff = CV_GetReal2DMatD(integral_sqimg,xmax,ymax) - CV_GetReal2DMatD(integral_sqimg,xmax,ymin-1);
      }
      else if(xmin && !ymin){ // first row
        diff   = CV_GetReal2DMatD(integral_image,xmax,ymax) - CV_GetReal2DMatD(integral_image,xmin-1,ymax);
        sqdiff = CV_GetReal2DMatD(integral_sqimg,xmax,ymax) - CV_GetReal2DMatD(integral_sqimg,xmin-1,ymax);
      }
      else{ // rest of the image
        diagsum    = CV_GetReal2DMatD(integral_image,xmax,ymax) + CV_GetReal2DMatD(integral_image,xmin-1,ymin-1);
        idiagsum   = CV_GetReal2DMatD(integral_image,xmax,ymin-1) + CV_GetReal2DMatD(integral_image,xmin-1,ymax);
        diff       = diagsum - idiagsum;
        sqdiagsum  = CV_GetReal2DMatD(integral_sqimg,xmax,ymax) + CV_GetReal2DMatD(integral_sqimg,xmin-1,ymin-1);
        sqidiagsum = CV_GetReal2DMatD(integral_sqimg,xmax,ymin-1) + CV_GetReal2DMatD(integral_sqimg,xmin-1,ymax);
        sqdiff     = sqdiagsum - sqidiagsum;
      }

      mean = diff/area;
      std  = sqrt((sqdiff - diff*diff/area)/(area-1));
      threshold = mean*(1.0+k*((std/128.0)-1.0));
      if(CV_GetReal2D(gray_image,i,j) < threshold)
        CV_SetReal2D(out,i,j,0);
      else
        CV_SetReal2D(out,i,j,254);
    }
  }
  cvReleaseMat(&integral_image);
  cvReleaseMat(&integral_sqimg);
  return out;
}
//fonction qui permet de définir la classe des pixels proche des contours
//out : image ternaire où:
//0 correspond à du texte
//254 correspond à du fond
//128 correspond à une classe indeterminée
IplImage *OpImg::SeuilSauvola(IplImage *i,int demiMask,float param=0.2){
  img=i;
  IplImage *out= cvCreateImage(cvSize(img->width,img->height),8,1);
  cvSet(out,cvScalar(128));

  float mean=0,vari=0;
  int m;

  unsigned char* ligneImg;
  unsigned char* ligneOut;

  for(int i=0;i<img->height;i++){
    ligneOut = (unsigned char*)(out->imageData+img->widthStep*i);
    for(int j=0;j<img->width;j++){
      m=0;
      for(int x=i-demiMask;x<=i+demiMask;x++){
        ligneImg = (unsigned char*)(img->imageData+img->widthStep*x);
        if(x>=0&&x<img->height)
          for(int y=j-demiMask;y<=j+demiMask;y++){
            if(y>=0&&y<img->width){
              mean+=ligneImg[y];
              m++;
            }
          }
      }
      mean/=m;
      m=0;
      for(int x=i-demiMask;x<=i+demiMask;x++){
        ligneImg = (unsigned char*)(img->imageData+img->widthStep*x);
        if(x>=0&&x<img->height)
          for(int y=j-demiMask;y<=j+demiMask;y++){
            if(y>=0&&y<img->width){
              vari+=(ligneImg[y]-mean)*(ligneImg[y]-mean);
              m++;
            }
          }
      }
      vari=sqrt(vari/m);
      if(mean*(1+param*((vari/128)-1))<CV_GetReal2D(img,i,j)){
        ligneOut[j]=254;
      }else{
        ligneOut[j]=0;
      }
    }
  }
  return out;
}

void OpImg::finalize(IplImage *img){
  static unsigned char* ligneImg;
  for(int i=0;i<img->height;i++){
    ligneImg = (unsigned char*)(img->imageData+img->widthStep*i);
    for(int j=0;j<img->width;j++){
      if(ligneImg[j]==128)
        ligneImg[j]=254;
    }
  }
}


/*
The Kmeans :
*/
void OpImg::segmenteKMean(float *data,int nbVal,float *centers){
  if(centers[0]==0&&centers[1]==0){
    float maxi=0;
    float mini=254;

    // now we create 2 pseudo-clusters :
    for(int i=0;i<nbVal;i++){
      if(maxi<data[i])
        maxi=data[i];
      if(mini>data[i])
        mini=data[i];
    }

    float nbClass[2];
    nbClass[0] = 0;
    nbClass[1] = 0;
    for (int i = 0; i <nbVal; i++) {
      if (fabs(maxi-data[i])>fabs(mini-data[i])) {
        centers[1] += data[i];
        nbClass[1]++;
      } else {
        centers[0] += data[i];
        nbClass[0]++;
      }
    }
    centers[0]/=nbClass[0];
    centers[1]/=nbClass[1];
  }

  //real KMEAN :
  float ancienC = -1;
  float ancienC1 = -1;
  uchar iterMx=0;
  while ((ancienC != centers[0])&&(ancienC1 != centers[1])&&iterMx<5) {
    ancienC = centers[0];
    ancienC1 = centers[1];
    newCenters(data,nbVal, centers);
    iterMx++;
  }
};

void OpImg::newCenters(float *data,int nbVal, float *centers) {
  // new centers :
  float newCenter1=0;
  float newCenter2=0;
  int nbCenter0 = 0;
  int nbCenter1 = 0;
  for (int i = 0; i < nbVal; i++) {
    if (fabs(centers[0]-data[i]) > fabs(centers[1]-data[i])){
      newCenter2 += data[i];
      nbCenter1++;
    }else{
      newCenter1 += data[i];
      nbCenter0++;
    }
  }
  centers[0] =newCenter1/ nbCenter0;
  centers[1] =newCenter2/ nbCenter1;
}

#define LOOK_AROUND(canny,i,j) {\
  endLine=false;\
  if(CV_GetReal2D(canny,i-1,j)==0&&CV_GetReal2D(canny,i-1,j-1)==0&&CV_GetReal2D(canny,i-1,j+1)==0&&\
  CV_GetReal2D(canny,i,j-1)==0&&CV_GetReal2D(canny,i,j+1)==0)\
  if(CV_GetReal2D(canny,i+1,j-1)==0||CV_GetReal2D(canny,i+1,j)==0||CV_GetReal2D(canny,i+1,j+1)==0)\
  endLine=true;\
  if(CV_GetReal2D(canny,i,j-1)==0&&CV_GetReal2D(canny,i-1,j-1)==0&&CV_GetReal2D(canny,i+1,j-1)==0&&\
  CV_GetReal2D(canny,i-1,j)==0&&CV_GetReal2D(canny,i+1,j)==0)\
  if(CV_GetReal2D(canny,i-1,j+1)==0||CV_GetReal2D(canny,i,j+1)==0||CV_GetReal2D(canny,i+1,j+1)==0)\
  endLine=true;\
  if(CV_GetReal2D(canny,i+1,j)==0&&CV_GetReal2D(canny,i+1,j-1)==0&&CV_GetReal2D(canny,i+1,j+1)==0&&\
  CV_GetReal2D(canny,i,j-1)==0&&CV_GetReal2D(canny,i,j+1)==0)\
  if(CV_GetReal2D(canny,i-1,j+1)==0||CV_GetReal2D(canny,i-1,j)==0||CV_GetReal2D(canny,i-1,j-1)==0)\
  endLine=true;\
  if(CV_GetReal2D(canny,i,j+1)==0&&CV_GetReal2D(canny,i-1,j+1)==0&&CV_GetReal2D(canny,i+1,j+1)==0&&\
  CV_GetReal2D(canny,i-1,j)==0&&CV_GetReal2D(canny,i+1,j)==0)\
  if(CV_GetReal2D(canny,i-1,j-1)==0||CV_GetReal2D(canny,i,j-1)==0||CV_GetReal2D(canny,i+1,j-1)==0)\
  endLine=true;\
}

typedef struct{
  unsigned short int x;
  unsigned short int y;
}position;


void OpImg::closeContours(IplImage *canny,IplImage *img){
  //first look for end of lines :

  vector<position> bouts(2000);
  bool endLine=false;
  unsigned int nbBouts=0;
  for(int i=1;i<img->height-1;i++){
    for(int j=1;j<img->width-1;j++){
      if(CV_GetReal2D(canny,i,j)==254){//we are on contour...

        LOOK_AROUND(canny,i,j);

        if(endLine){
          if(nbBouts+1 > bouts.capacity()){
            bouts.reserve(nbBouts*2);
          }
          bouts.resize(nbBouts+1);
          bouts[nbBouts].x=i;
          bouts[nbBouts].y=j;
          nbBouts++;
        }
      }
    }
  }
  //try to connect the open CC :
  unsigned int distMin=99999999;
  char distMax=(char)(img->height+img->width)/200;
  int position=0;
  for(unsigned int i=0;i<nbBouts;i++){
    distMin=999999;
    for(unsigned int j=0;j<nbBouts;j++){
      if(i!=j){
        if(distMin>sqrt((float)(bouts[i].x-bouts[j].x)*(bouts[i].x-bouts[j].x)+(bouts[i].y-bouts[j].y)*(bouts[i].y-bouts[j].y))){
          distMin=(unsigned int)sqrt((float)(bouts[i].x-bouts[j].x)*(bouts[i].x-bouts[j].x)+(bouts[i].y-bouts[j].y)*(bouts[i].y-bouts[j].y));
          position=j;
        }
      }
    }
    //if(distMin<distMax)
    cvLine(canny,cvPoint(bouts[i].y,bouts[i].x),cvPoint(bouts[position].y,bouts[position].x),cvScalar(254));
  }

};


static IplImage * Threshold_Otsu( IplImage * image )
{

  IplImage *out= cvCreateImage(cvSize(image->width,image->height),8,1);
  //cout<<"Attention, penser  a diminuer de 30% le temps de calcul (optimisation non prise en compte)"<<endl;

  int hist_size[] = {255};
  CvHistogram* hist;

  hist = cvCreateHist( 1, hist_size, CV_HIST_ARRAY);
  cvCalcHist( &image, hist, 0, 0 );
  float max_val = 0;

  int i, count;
  const float* h;
  float sum = 0, mu = 0;
  bool uniform = false;
  float low = 0, high = 0, delta = 0;
  float* nu_thresh = 0;
  float mu1 = 0, q1 = 0;
  float max_sigma = 0;

  count = hist->mat.dim[0].size;
  h = (float*)cvPtr1D( hist->bins, 0 );

  if( !CV_HIST_HAS_RANGES(hist) || CV_IS_UNIFORM_HIST(hist) )
  {
    if( CV_HIST_HAS_RANGES(hist) )
    {
      low = hist->thresh[0][0];
      high = hist->thresh[0][1];
    }
    else
    {
      low = 0;
      high = count;
    }

    delta = (high-low)/count;
    low += delta*0.5;
    uniform = true;
  }
  else
    nu_thresh = hist->thresh2[0];

  for( i = 0; i < count; i++ )
  {
    sum += h[i];
    if( uniform )
      mu += (i*delta + low)*h[i];
    else
      mu += (nu_thresh[i*2] + nu_thresh[i*2+1])*0.5*h[i];
  }

  sum = fabs(sum) > FLT_EPSILON ? 1./sum : 0;
  mu *= sum;

  mu1 = 0;
  q1 = 0;

  for( i = 0; i < count; i++ )
  {
    float p_i, q2, mu2, val_i, sigma;

    p_i = h[i]*sum;
    mu1 *= q1;
    q1 += p_i;
    q2 = 1. - q1;

    if( MIN(q1,q2) < FLT_EPSILON || MAX(q1,q2) > 1. - FLT_EPSILON )
      continue;

    if( uniform )
      val_i = i*delta + low;
    else
      val_i = (nu_thresh[i*2] + nu_thresh[i*2+1])*0.5;

    mu1 = (mu1 + val_i*p_i)/q1;
    mu2 = (mu - q1*mu1)/q2;
    sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
    if( sigma > max_sigma )
    {
      max_sigma = sigma;
      max_val = val_i;
    }
  }

  unsigned char* ligneImg;
  unsigned char* ligneEdge;
  unsigned char* ligneOut;
  float echange;
  float pixel;

  //first pass : only on edges :
  for(int i=0;i<image->height;i++){
    ligneImg = (unsigned char*)(image->imageData+image->widthStep*i);
    ligneOut = (unsigned char*)(out->imageData+out->widthStep*i);
    for(int j=0;j<image->width;j++){
      if(ligneImg[j]>max_val)
        ligneOut[j]=254;
      else
        ligneOut[j]=0;
    }
  }

  cvReleaseHist(&hist);
  return out;
}


