#ifndef _OPIMG_H
#define _OPIMG_H

#pragma warning(disable:4251)
#include <cv.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <math.h>
#include <vector>
#include "connected.h"

using namespace std;

#include "macro.h"

#define CLASS_BACKGROUND 0
#define CLASS_TEXT 1
struct PileCanny{
	uchar **PC_stack_top, **PC_stack_bottom, **PC_stack_current;
	int PC_maxsize;
};
#define PILE_CANNY_INIT(max)			p->PC_maxsize=max;p->PC_stack_top=p->PC_stack_bottom=p->PC_stack_current= new unsigned char*[p->PC_maxsize]
#define PILE_CANNY_DELETE()				delete [] p->PC_stack_bottom;
#define PILE_CANNY_DELETE_PTR(ptr)	delete [] ptr->PC_stack_bottom;
#define PILE_CANNY_PUSH(d)				*p->PC_stack_top++ = (d);p->PC_stack_current=p->PC_stack_top
#define PILE_CANNY_POP(d)				(d) = *--p->PC_stack_current
#define PILE_CANNY_IS_END()				(p->PC_stack_current<p->PC_stack_bottom)
#define PILE_CANNY_RELOAD()				p->PC_stack_current=p->PC_stack_top
#define PILE_CANNY_RESIZE(nbValAdd)    if( (p->PC_stack_top - p->PC_stack_bottom) + nbValAdd > p->PC_maxsize ){\
	unsigned char** new_stack_bottom;\
	p->PC_maxsize = MAX( p->PC_maxsize * 3.0/2.0, p->PC_maxsize + nbValAdd );\
	new_stack_bottom = new unsigned char*[p->PC_maxsize];\
	for(int cpt=0;cpt<(p->PC_stack_top - p->PC_stack_bottom);cpt++) new_stack_bottom[cpt]=p->PC_stack_bottom[cpt];\
	p->PC_stack_top = new_stack_bottom + (p->PC_stack_top - p->PC_stack_bottom);\
	p->PC_stack_current=p->PC_stack_top;\
	delete [] p->PC_stack_bottom;\
	p->PC_stack_bottom = new_stack_bottom;}
/*
class PileCanny{
	unsigned short *stack_top, *stack_bottom, *stack_current;
	int maxsize;
public:
	PileCanny(int max){
		maxsize=2*max;
		this->stack_top = this->stack_bottom = this->stack_current = (unsigned short*)cvAlloc( this->maxsize*sizeof(unsigned short));
	}
	~PileCanny() 
	{
		cvFree( &this->stack_bottom );
	}
	inline void reload(){
		stack_current=stack_top;
	}
	inline void push(int i,int j){
		*stack_top++ = (i); *stack_top++ = (j);stack_current=stack_top;
	}
	inline bool pop(int &i,int &j){
		j = *--stack_current;i = *--stack_current;
		return stack_current>=this->stack_bottom;
	}
	inline void shouldResize(int nbValAdd){
		if( (this->stack_top - this->stack_bottom) + nbValAdd > this->maxsize )
		{
			unsigned short* new_stack_bottom;
			this->maxsize = MAX( this->maxsize * 3/2, maxsize + nbValAdd );
			new_stack_bottom = (unsigned short*)cvAlloc( this->maxsize * sizeof(unsigned short));
			memcpy( new_stack_bottom, this->stack_bottom, (this->stack_top - this->stack_bottom)*sizeof(unsigned short) );
			this->stack_top = new_stack_bottom + (this->stack_top - this->stack_bottom);
			cvFree( &this->stack_bottom );
			this->stack_bottom = new_stack_bottom;
		}
	}
};
*/

class OpImg{
protected:
	CvMat* ccTmp;
	int sommeVal[50];//nb max of clics... arbitrary set to 50...
	int nbSomme[50];
	int otherLabel[50];
	inline bool proximaEdge(IplImage *img,int i,int j,int demiMask){
		unsigned char* ligneImg;
		if(j-demiMask<0)
			return false;
		if(j+demiMask>=img->width)
			return false;
		if(i-demiMask<0)
			return false;
		if(i+demiMask>=img->height)
			return false;
		for(int x=i-demiMask;x<=i+demiMask;x++){
			ligneImg = (unsigned char*)(img->imageData+img->widthStep*x);
			for(int y=j-demiMask;y<=j+demiMask;y++){
				if(ligneImg[y]!=0)//c'est un contour !
					return true;
			}
		}
		return false;
	};

	inline unsigned char defineClass(IplImage *img,int i,int j){
		if(j-1<0)
			j++;
		if(j+1>=img->width)
			j--;
		if(i-1<0)
			i++;
		if(i+1>=img->height)
			i--;
		unsigned char* ligneImg;
		for(int x=i-1;x<=i+1;x++){
			ligneImg = (unsigned char*)(img->imageData+img->widthStep*x);
			for(int y=j-1;y<=j+1;y++){
				if(ligneImg[y]>0){//c'est un contour !
					if(ligneImg[y]==32){//contour vers le bas
						if(x>i)//on est au dessus du contour
							return 0;
						if(x<i)//on est dessous le contour
							return 254;
					}
					if(ligneImg[y]==64){//contour vers le haut
						if(x>i)//on est au dessus du contour
							return 254;
						if(x<i)//on est dessous le contour
							return 0;
					}
					if(ligneImg[y]==96){//contour vers la droite
						if(y>j)//on est a droite du contour
							return 0;
						if(y<j)//on est a gauche du contour
							return 254;
					}
					if(ligneImg[y]==128){//contour vers la gauche
						if(y>j)//on est a droite du contour
							return 254;
						if(y<j)//on est a gauche du contour
							return 0;
					}
				}
			}
		}
		return 128;

	};
	inline unsigned char defineClassOnEdge(IplImage *edge,IplImage *img,int i,int j){
		if(j-2<0)
			j=2;
		if(j+2>=img->width)
			j=img->width-3;
		if(i-2<0)
			i=2;
		if(i+2>=img->height)
			i=img->height-3;
		unsigned char* ligneImg;
		unsigned char* ligneEdge;
		float moyT,moyF;
		float nbT,nbF;
		moyT=0;
		moyF=0;
		nbT=0;
		nbF=0;
		for(int x=i-2;x<=i+2;x++){
			ligneImg = (unsigned char*)(img->imageData+img->widthStep*x);
			ligneEdge = (unsigned char*)(edge->imageData+img->widthStep*x);
			for(int y=j-2;y<=j+2;y++){
				if(ligneEdge[y]==0){//c'est du texte
					moyT+=ligneImg[y];
					nbT++;
				}else{
					if(ligneEdge[y]==254){//c'est du fond
						moyF+=ligneImg[y];
						nbF++;
					}
				}
			}
		}
		if(nbT>1&&nbF>1){
			moyT=moyT/nbT;
			moyF=moyF/nbF;
			if(abs(cvGetReal2D(img,i,j)-moyT)>abs(cvGetReal2D(img,i,j)-moyF)+9){
				return 254;
			}else{
				return 0;
			}
		}
		return 128;
	};

	//inline CvMat* connected(IplImage *img,int i,int j,int demiMask){
	inline void connected(IplImage *edge,IplImage *imgNG,int i,int j,int demiMask,float *moy){
		unsigned char oldEdge=cvGetReal2D(edge,i,j);
		cvSetReal2D(edge,i,j,254);//c'est un contour
		if(ccTmp==NULL){
			ccTmp=cvCreateMat(demiMask*2+1,demiMask*2+1,CV_32SC1);
		};
		for(int tmpiter=0;tmpiter<50;tmpiter++){
			nbSomme[tmpiter]=0;
			sommeVal[tmpiter]=0;
			otherLabel[tmpiter]=tmpiter;
		}

		int idCC=0;
		unsigned char* ligneImg;
		int tmp,tmp1;

		//Premiere case :
		if(cvGetReal2D(edge,i-demiMask,j-demiMask)>128){
			cvSetReal2D(ccTmp,0,0,-1);//c'est un contour
			idCC--;//pour qu'après ça ne passe pas direct à 1...
		}else{
			cvSetReal2D(ccTmp,0,0,idCC);
			nbSomme[idCC]++;
			sommeVal[idCC]+=cvGetReal2D(imgNG,i-demiMask,j-demiMask);
		}

		//première ligne :
		for(int y=j-demiMask+1,y1=1;y<=j+demiMask;y++,y1++){
			if(cvGetReal2D(edge,i-demiMask,y)>128)
				cvSetReal2D(ccTmp,0,y1,-1);//c'est un contour
			else{
				if(cvGetReal2D(ccTmp,0,y1-1)>=0){
					tmp=cvGetReal2D(ccTmp,0,y1-1);
					cvSetReal2D(ccTmp,0,y1,tmp);
					nbSomme[tmp]++;
					sommeVal[tmp]+=cvGetReal2D(imgNG,i-demiMask,y);
				}else{//nouvelle CC :
					idCC++;
					cvSetReal2D(ccTmp,0,y1,idCC);
					nbSomme[idCC]++;
					sommeVal[idCC]+=cvGetReal2D(imgNG,i-demiMask,y);
				}
			}
		}


		//reste de l'image :
		for(int x=i-demiMask+1,x1=1;x<=i+demiMask;x++,x1++){
			ligneImg = (unsigned char*)(imgNG->imageData+imgNG->widthStep*x);
			if(x>0&&x<imgNG->height){
				//premier pixel de la ligne :
				if(cvGetReal2D(edge,x,j-demiMask)>128)
					cvSetReal2D(ccTmp,x1,0,-1);//c'est un contour
				else{
					//juste à regarder le pixel du dessus :
					if(cvGetReal2D(ccTmp,x1-1,0)>=0){//c'est un label, donc on récupère son num et on s'ajoute :
						tmp=cvGetReal2D(ccTmp,x1-1,0);
						cvSetReal2D(ccTmp,x1,0,tmp);
						nbSomme[tmp]++;
						sommeVal[tmp]+=ligneImg[j-demiMask];
					}else{//nouvelle CC :
						idCC++;
						cvSetReal2D(ccTmp,x1,0,idCC);
						nbSomme[idCC]++;
						sommeVal[idCC]+=ligneImg[j-demiMask];
					}
				}
				for(int y=j-demiMask+1,y1=1;y<=j+demiMask;y++,y1++){
					if(y>=0&&y<edge->width){

						//est on sur un contour ?
						if(cvGetReal2D(edge,x,y)>128){
							cvSetReal2D(ccTmp,x1,y1,-1);//c'est un contour
						}else{

							//deux choses à tester : au-dessus et à gauche :
							tmp=cvGetReal2D(ccTmp,x1-1,y1);
							tmp1=cvGetReal2D(ccTmp,x1,y1-1);
							if((tmp>=0)&&(tmp1>=0)){//On a deux labels
								tmp=otherLabel[tmp];
								tmp1=otherLabel[tmp1];
								if(tmp!=tmp1){//ils sont différents
									if(tmp>tmp1){//celui de gauche est plus grand (on garde)
										cvSetReal2D(ccTmp,x1,y1,tmp);
										nbSomme[tmp]++;
										sommeVal[tmp]+=ligneImg[y];

										nbSomme[tmp]+=nbSomme[tmp1];
										nbSomme[tmp1]=0;
										sommeVal[tmp]+=sommeVal[tmp1];
										sommeVal[tmp1]=0;
										otherLabel[tmp1]=tmp;
									}else{
										cvSetReal2D(ccTmp,x1,y1,tmp1);
										nbSomme[tmp1]++;
										sommeVal[tmp1]+=ligneImg[y];

										nbSomme[tmp1]+=nbSomme[tmp];
										nbSomme[tmp]=0;
										sommeVal[tmp1]+=sommeVal[tmp];
										sommeVal[tmp]=0;
										otherLabel[tmp]=tmp1;
									}
								}else{//tmp==tmp1
									cvSetReal2D(ccTmp,x1,y1,tmp);
									nbSomme[tmp]++;
									sommeVal[tmp]+=ligneImg[y];
								}
							}else{//il y a un contour au moins :
								if(tmp>=0){//c'est le pixel du haut qui est bon
									tmp=otherLabel[tmp];
									cvSetReal2D(ccTmp,x1,y1,tmp);
									nbSomme[tmp]++;
									sommeVal[tmp]+=ligneImg[y];
								}else{
									if(tmp1>=0){//c'est le pixel de gauche qui est bon
										tmp=otherLabel[tmp1];
										cvSetReal2D(ccTmp,x1,y1,tmp1);
										nbSomme[tmp1]++;
										sommeVal[tmp1]+=ligneImg[y];
									}else{//il n'y a que des contours... Il faut créer un nouveau label !
										idCC++;
										cvSetReal2D(ccTmp,x1,y1,idCC);
										nbSomme[idCC]++;
										sommeVal[idCC]+=ligneImg[y];
									}
								}
							}
						}
					}
				}
			}
		}
		cvSetReal2D(edge,i,j,oldEdge);//remet la bonne couleur...

		//on a accummulé toutes les CC... trouvons les bonnes :
		int nbCC=0;
		for(int iCC=0;iCC<=idCC;iCC++){
			if(nbSomme[iCC]>3){//nous avons un candidat (avec assez de valeurs) ;-)
				moy[nbCC]=(float)sommeVal[iCC]/nbSomme[iCC];
				nbCC++;
			}
		}
		if(nbCC<=1){//pas assez de valeurs
			moy[0]=0;
			moy[1]=0;
			return;
		}
		if(nbCC==2)
			return;//nous avons nos deux moyennes !

		if(nbCC==3){//fusion de la moyenne en trop :
			if((abs(moy[0]-moy[1])<abs(moy[0]-moy[2]))&&(abs(moy[0]-moy[1])<abs(moy[1]-moy[2]))){//moy[0] et moy[1] sont les plus proches !
				moy[0]=(moy[0]+moy[1])/2;
				moy[1]=moy[2];
			}else{
				if((abs(moy[0]-moy[2])<abs(moy[0]-moy[1]))&&(abs(moy[0]-moy[2])<abs(moy[1]-moy[2]))){//moy[0] et moy[2] sont les plus proches !
					moy[0]=(moy[0]+moy[2])/2;
				}else{//moy[1] et moy[2] sont les plus proches !
					moy[1]=(moy[1]+moy[2])/2;
				}
			}
			return;//nous avons nos deux moyennes !
		}


		//on a trop de CC... On clusterise en utilisant les K-Means :
		float *moyTmp=new float[nbCC];
		for(int iCC=0;iCC<nbCC;iCC++){
			moyTmp[iCC]=moy[iCC];
		}
		segmenteKMean(moyTmp,nbCC,moy);
		delete moyTmp;

		/*
		IplImage* img1=cvCreateImage(cvSize(demiMask*2+1,demiMask*2+1),8,1);
		CvMat* out=cvCreateMat(demiMask*2+1,demiMask*2+1,CV_32SC1);
		//recopie les valeurs :
		unsigned char* ligneImg;
		for(int x=i-demiMask,cpt=0;x<=i+demiMask;x++,cpt++){
			ligneImg = (unsigned char*)(img->imageData+img->widthStep*x);
			if(x>0&&x<img->height)
				for(int y=j-demiMask,cpt1=0;y<=j+demiMask;y++,cpt1++){
					if(y>=0&&y<img->width){
						int ttmp=CV_GetReal2D(img,x,y);
						cvSetReal2D(img1,cpt,cpt1,CV_GetReal2D(img,x,y));
					}
				}
		}
		ConnectedComponents cc(10);
		cc.connectedComp(img1,out);
		return out;
		/*/
	}
public:
	
	static IplImage *debugImg;
	static CvMat* labelimg;
	IplImage *img;
	IplImage *edge;
	IplImage *SeuilSauvola(IplImage *,int,float);
  IplImage *SeuilSauvolaOptimiz(IplImage *,int,float);
  IplImage *expandEdge(IplImage *,PileCanny*,IplImage *,int=3);
  IplImage *expandEdgeEM(IplImage *,PileCanny*,IplImage *,int=3);
  IplImage *expandEdge(IplImage *,PileCanny *,IplImage *,int,char);
	IplImage *expandEdge(IplImage *,IplImage *,int,char);
	OpImg(){ccTmp=NULL;};
	void finalize(IplImage *);
	static void segmenteKMean(float *data,int nbVal,float *centers);
	inline static void newCenters(float *data,int nbVal, float *centers);
	inline static void closeContours(IplImage *,IplImage *);

	void binarize( IplImage * dest) 
	{
		unsigned char* ligneOut = (unsigned char*)(dest->imageData);
		unsigned char* ligneOutP;
		ligneOut[0]=254;

		// label the first row :
		for(int c=1; c<dest->width; c++) {
			if(ligneOut[c]==128){
				ligneOut[c]=ligneOut[c-1];
			}
		}


		int cpt=0;
		for(int i=1;i<dest->height;i++){
			ligneOut = (unsigned char*)(dest->imageData+dest->widthStep*i);
			ligneOutP = (unsigned char*)(dest->imageData+dest->widthStep*(i-1));

			if(ligneOut[0]==128)
				ligneOut[0]=ligneOutP[0];
			for(int j=1;j<dest->width;j++){
				if(ligneOut[j]==128){
					cpt=ligneOut[j-1]+ligneOutP[j-1]+ligneOutP[j]+ligneOutP[j+1];
					if(cpt<128)
						ligneOut[j]=0;
					else{
						if(cpt>750)
							ligneOut[j]=254;
					}
				}
			}
		}
	}


};


#endif // _OPIMG_H