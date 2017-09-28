#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<malloc.h>
//#define BUFFERSIZE 1024*1024


void input(int row,int col,int row1,int col1,int n,int n1, float **i1,float **o1,float **w1,float **w2)
{
	float b1=0.35,b2=0.60;
	int i,j;

	printf("\n The input entered is \n");
	 for(i=0;i<row;i++)
	{	
	for(j=0;j<col;j++)
	{
	printf("%f\t",i1[i][j]);
	}
	printf("\n");
	}

	printf("\n The target output entered is \n");
	 for(i=0;i<row1;i++)
	{	
	for(j=0;j<col1;j++)
	{
	printf("%f\t",o1[i][j]);
	}
	printf("\n");
	}
	
	printf("\n The biases are \t");
	printf("\nb1=%f\t b2=%f\n",b1,b2);
	
	printf("\n The weights entered are(hidden layer) \n");
	 for(i=0;i<n;i++)
	{
	for(j=0;j<n;j++)
	{
	printf("%f\t",w1[i][j]);
	}
	printf("\n");
        }

	printf("\n The weights entered are(output layer)\n");
	 for(i=0;i<n1;i++)
	{
	for(j=0;j<n1;j++)
	{
	printf("%f\t",w2[i][j]);
	}
	printf("\n");
	}
	
}

void forward(int row,int col,int row1,int col1,int n,int n1,float **i1,float **o1,float **w1,float **w2,float **neth,float **outh,float **neto,float **outo,float **E)
{
	float b1=0.35,b2=0.60;
	int i,j;
	float x=0;
	printf("\n*******************Forward Pass*************************************\n");

	neth=(float**)malloc(row*sizeof(float*));
	for(i=0;i<row;i++)
	{
	neth[i]=(float*)malloc(col*sizeof(float));
	}
	
		
	n=row*col;
	for(i=0;i<n;i++)
	{
	for(j=0;j<n;j++)
	{	
	neth[i][j]=w1[i][j]*i1[i][j]+b1*1;
	}
	}

	printf("\nThe total net input (hidden layer)  \n");
	for(i=0;i<n;i++)
	{for(j=0;j<n;j++) 
	{printf("\tneth=%f\t",neth[i][j]);
	}
	printf("\n");
	}
	
	outh=(float**)malloc(row*sizeof(float*));
	for(i=0;i<row;i++)
	{
	outh[i]=(float*)malloc(col*sizeof(float));
	}

	for(i=0;i<n;i++)
	{
	for(j=0;j<n;j++)
	{	
	outh[i][j]=1/(1+(exp(x-neth[i][j])));
	}
	}

	printf("\nThe output for hidden layer is \n");
	for(i=0;i<n;i++)
	{for(j=0;j<n;j++) 
	{	printf("\touth=%f\t",outh[i][j]);
	}
	printf("\n");
	}


	neto=(float**)malloc(row1*sizeof(float*));
	for(i=0;i<row1;i++)
	{
	neto[i]=(float*)malloc(col1*sizeof(float));
	}
		n1=row1*col1;
	for(i=0;i<n1;i++)
	{
	for(j=0;j<n1;j++)
	{	
	neto[i][j]=w2[i][j]*i1[i][j]+b2*1;
	}
	}

	printf("\nThe total net input (output layer)  \n");
	for(i=0;i<n1;i++)
	{for(j=0;j<n1;j++) 
	{printf("\tneto=%f\t",neto[i][j]);
	}
	printf("\n");
	}
	
	outo=(float**)malloc(row1*sizeof(float*));
	for(i=0;i<row1;i++)
	{
	outo[i]=(float*)malloc(col1*sizeof(float));
	}

	for(i=0;i<n1;i++)
	{
	for(j=0;j<n1;j++)
	{	
	outo[i][j]=1/(1+(exp(x-neto[i][j])));
	}
	}

	printf("\nThe output for output layer is \n");
	for(i=0;i<n1;i++)
	{for(j=0;j<n1;j++) 
	{	printf("\touto=%f\t",outo[i][j]);
	}
	printf("\n");
	}
	
	E=(float**)malloc(row1*sizeof(float*));
	for(i=0;i<row1;i++)
	{
	E[i]=(float*)malloc(col1*sizeof(float));
	}
	
	for(i=0;i<n1;i++)
	{
	for(j=0;j<n1;j++)
	{
	E[i][j]=((o1[i][j]-outo[i][j])*(o1[i][j]-outo[i][j]))/2;
	}
	}
	printf("\n");
	for(i=0;i<n1;i++)
	{
	for(j=0;j<n1;j++)
	{
	printf(" Etotal=%f\t",E[i][j]);
	}
	printf("\n");
	}

}

void backward(int row,int col,int row1,int col1,int n,int n1,float **i1,float **o1,float **w1,float **w2,float **neth,float **outh,float **neto,float **outo,float **E, float **de,float **douto,float **dneto,float **dE,float **dEo11,float **dEo1,float **dEo22,float **dEo2,float **dEt,float **douth1, float **dneth1,float **dE1)
{
 	float b1=0.35,b2=0.60;
	int i,j;
	float x=0,eta=0.5;
 	float dEo111[20][20],dneto11[20][20],dneto12[20][20], dEo222[20][20];
 	printf("\n*********************Backward Pass******************************\n");
 	printf("\n Output Layer \n");
	

	de=(float**)malloc(row1*sizeof(float*));
	for(i=0;i<row1;i++)
	{
	de[i]=(float*)malloc(col1*sizeof(float));
	}
	
	n1=row1*col1;
	for(i=0;i<n1;i++){
	for(j=0;j<n1;j++){
	
	de[i][j]=-(o1[i][j]-outo[i][j]);
	
	 }
	 }
	
	printf("\n dEtotal/douto1 is \n");
	 for(i=0;i<n1;i++)
	 {for(j=0;j<n1;j++){
	 printf("\t de=%f\t",de[i][j]);
	 }
	printf("\n");
	 }

	douto=(float**)malloc(row1*sizeof(float*));
	for(i=0;i<row1;i++)
	{
	douto[i]=(float*)malloc(col1*sizeof(float));
	}
	
	for(i=0;i<n1;i++)
	{for(j=0;j<n1;j++)
	 { douto[i][j]=outo[i][j]*(1-outo[i][j]);}
	printf("\n");}

	printf("\n douto1/dneto1 is \n");
	for(i=0;i<n1;i++){
	for(j=0;j<n1;j++){
	printf("\t douto1=%f\t",douto[i][j]);}
	printf("\n");	}
	
	dneto=(float**)malloc(row1*sizeof(float*));
	for(i=0;i<row1;i++)
	{
	dneto[i]=(float*)malloc(col1*sizeof(float));
	}
	for(i=0;i<n1;i++){
	for(j=0;j<n1;j++){
	dneto[i][j]=outh[i][j];
			}
			}
	
	printf("\n dneto1/dw5 is \n");
	for(i=0;i<n1;i++){
	for(j=0;j<n1;j++){
	 printf("\tdneto1=%f\t",dneto[i][j]);				
	}printf("\n");
	}
	
	dE=(float**)malloc(row1*sizeof(float*));
	for(i=0;i<row1;i++)
	{
	dE[i]=(float*)malloc(col1*sizeof(float));
	}
	
	for(i=0;i<n1;i++){
	for(j=0;j<n1;j++){
	dE[i][j]=de[i][j]*douto[i][j]*dneto[i][j];
	}
	}
	printf("\n The value of dEtotal/dw5 is \n");
	for(i=0;i<n1;i++) 
	{for(j=0;j<n1;j++)
	{printf("\t dE=%f\t",dE[i][j]);
	}
	printf("\n");
	}


	printf("\nUpdating value of weights in output layer to hidden layer \n");
	for(i=0;i<n1;i++) 
	{for(j=0;j<n1;j++)
	{ w2[i][j]=w2[i][j]-(eta*dE[i][j]);}
	}
	for(i=0;i<n1;i++){
	for(j=0;j<n1;j++){  
	printf("\t w=%f\t",w2[i][j]); 
	}
	printf("\n");
	}
	printf("\n*************************************************\n");

	/** HIDDEN LAYER**/
	printf("\nHidden Layer \n");
	n=row*col;
	
	for(i=0;i<n;i++)
	{for(j=0;j<n;j++)
	{dEo111[i][j]=-(o1[i][j]-outo[i][j]);}
	}
	
	
	dEo11=(float**)malloc(row*sizeof(float*));
	for(i=0;i<row;i++)
	{
	dEo11[i]=(float*)malloc(col*sizeof(float));
	}
	
	for(i=0;i<n;i++)
	{
	for(j=0;j<n;j++)
	{ dEo11[i][j]=dEo111[i][j]*douto[i][j];}
	}

	printf("\n dEo1/dneto1 is \n");
	for(i=0;i<n;i++){
	for(j=0;j<n;j++){printf("\t dE011=%f\t",dEo11[i][j]);}
	printf("\n");
	}

	for(i=0;i<n;i++)
	{for(j=0;j<n;j++)
	{dneto11[i][j]=w2[i][j];}
	}

	dEo1=(float**)malloc(row*sizeof(float*));
	for(i=0;i<row;i++)
	{
	dEo1[i]=(float*)malloc(col*sizeof(float));
	}

	for(i=0;i<n;i++)
	{for(j=0;j<n;j++)
	{dEo1[i][j]=dEo11[i][j]*dneto11[i][j];}
	}

	printf("\n dEo1/douth1 is \n");
	for(i=0;i<n;i++)
	{for(j=0;j<n;j++)
	{printf("\t dEo1=%f\t", dEo1[i][j]);}
	printf("\n");
	}

	for(i=0;i<n;i++)
	{for(j=0;j<n;j++)
	{dneto12[i][j]=w2[i][j];}
	}

	for(i=0;i<n;i++){
	for(j=0;j<n;j++){dEo222[i][j]=-(o1[i][j]-outo[i][j]);}
	}

	for(i=0;i<n;i++){
	for(j=0;j<n;j++){
	douto[i][j]=outo[i][j]*(1-outo[i][j]);}
	}

	dEo22=(float**)malloc(row*sizeof(float*));
	for(i=0;i<row;i++)
	{
	dEo22[i]=(float*)malloc(col*sizeof(float));
	}

	for(i=0;i<n;i++)
	{for(j=0;j<n;j++)
	{dEo22[i][j]=dEo222[i][j]*douto[i][j];}
	}

	printf("\n dEo2/dneto2 is \n");
	for(i=0;i<n;i++)
	{
	for(j=0;j<n;j++)
	{
	printf("\t dEo22=%f\t",dEo22[i][j]);
	}
	printf("\n");
	}



	dEo2=(float**)malloc(row*sizeof(float*));
	for(i=0;i<row;i++)
	{
	dEo2[i]=(float*)malloc(col*sizeof(float));
	}
	for(i=0;i<n;i++)
	{for(j=0;j<n;j++)
	{dEo2[i][j]=dEo22[i][j]*dneto12[i][j];}}
	
	printf("\n dEo2/douth1 is \n");
	for(i=0;i<n;i++)
	{for(j=0;j<n;j++)
	{printf("\t dEo2=%f\t",dEo2[i][j]);
	}printf("\n");
	}
	printf("\n**********************************************\n");


	dEt=(float**)malloc(row*sizeof(float*));
	for(i=0;i<row;i++)
	{
	dEt[i]=(float*)malloc(col*sizeof(float));
	}
	for(i=0;i<n;i++){
	for(j=0;j<n;j++){
	dEt[i][j]=dEo1[i][j]+dEo2[i][j];}
	}

	printf("\n dEtotal/douth1 is \n");
	for(i=0;i<n;i++)
	{for(j=0;j<n;j++){
	printf("\t  dEt1=%f\t",dEt[i][j]);}
	printf("\n");
	}

	douth1=(float**)malloc(row*sizeof(float*));
	for(i=0;i<row;i++)
	{
	douth1[i]=(float*)malloc(col*sizeof(float));
	}	
	for(i=0;i<n;i++)
	{for(j=0;j<n;j++)	
	{douth1[i][j]=outh[i][j]*(1-outh[i][j]);}}
	
	printf("\n douth1/dneth1 is \n");
	for(i=0;i<n;i++){
	for(j=0;j<n;j++){printf("\t  douth1=%f\t",douth1[i][j]);}
	printf("\n");
	}


	dneth1=(float**)malloc(row*sizeof(float*));
	for(i=0;i<row;i++)
	{
	dneth1[i]=(float*)malloc(col*sizeof(float));
	}
	for(i=0;i<n;i++)
	{for(j=0;j<n;j++){
	dneth1[i]=i1[i];}
	}
	printf("\n  dneth1/dw1 is \n");
	for(i=0;i<n;i++){
	for(j=0;j<n;j++){
	printf("\t dneth1=%f\t",dneth1[i][j]);}
	printf("\n");
	}


	dE1=(float**)malloc(row*sizeof(float*));
	for(i=0;i<row;i++)
	{
	dE1[i]=(float*)malloc(col*sizeof(float));
	}
	for(i=0;i<n;i++){
	for(j=0;j<n;j++){
	dE1[i][j]=dEt[i][j]*douth1[i][j]*dneth1[i][j];}}

	printf("\n dEtotal/dw1 is \n");
	for(i=0;i<n;i++){
	for(j=0;j<n;j++){
	printf("\t  dE1=%f\t",dE1[i][j]);}
	printf("\n");
	}


	printf("\nUpdating value of weights in hidden layer to input layer  \n");
	for(i=0;i<n;i++)
	{for(j=0;j<n;j++)
	{w1[i][j]=w1[i][j]-(eta*dE1[i][j]);}}

	for(i=0;i<n;i++)
	{for(j=0;j<n;j++)
	{ printf("\t W=%f\t",w1[i][j]); }
	printf("\n");
	}

	printf("\n**********************************************\n");
	printf("\nUpdated all the weights\n");
	for(i=0;i<n;i++)
	{
	for(j=0;j<n;j++)
	{
	printf("\t w= %f\t",w1[i][j]);}
	printf("\n");
	}
	 for(i=0;i<n1;i++)
	{
	for(j=0;j<n1;j++)
	{
	printf("\t w= %f\t",w2[i][j]);}
	printf("\n");
	}

	printf("\n**********************************************\n");

}





int main()
{
float b1=0.35,b2=0.60;
int row,col,row1,col1;
int i,j,k,n,n1;
float **i1,**o1;
float **w1, **w2;
float **neth,**neto,**outh,**outo,**E;
float **de,**douto,**dneto,**dE,**dEo11,**dEo1,**dEo22,**dEo2,**dEt,**douth1,**dneth1,**dE1;

printf("\n Enter the number of rows and columns for input\n");
scanf("%d%d",&row,&col);

printf("\n Enter the number of rows and columns for output\n");
scanf("%d%d",&row1,&col1);

i1=(float**)malloc(row*sizeof(float*));
for(i=0;i<row;i++)
{
i1[i]=(float*)malloc(col*sizeof(float));
}
printf("\n Enter the inputs \n ");
for(i=0;i<row;i++)
{
for(j=0;j<col;j++)
{
scanf("%f",&i1[i][j]);
}
}


o1=(float**)malloc(row1*sizeof(float*));
for(i=0;i<row1;i++)
{
o1[i]=(float*)malloc(col1*sizeof(float));
}
printf("\n Enter the target ouputs \n ");
for(i=0;i<row1;i++)
{
for(j=0;j<col1;j++)
{
scanf("%f",&o1[i][j]);
}
}


	n=row*col;
	n1=row1*col1;
	 
	w1=(float**)malloc(n*sizeof(float*));
	for(i=0;i<n;i++)
	{
	w1[i]=(float*)malloc(n*sizeof(float));
	}
	printf("\n Enter the number of weights(hidden layer) ");
	for(i=0;i<n;i++)
	{
	for(j=0;j<n;j++)
	{
	scanf("%f",&w1[i][j]);
	}
	}
	
	w2=(float**)malloc(n1*sizeof(float*));
	for(i=0;i<n1;i++)
	{
	w2[i]=(float*)malloc(n1*sizeof(float));
	}

	printf("\n Enter the number of weights(output layer) ");
	for(i=0;i<n1;i++)
	{
	for(j=0;j<n;j++)
	{
	scanf("%f",&w2[i][j]);
	}
	}

	

printf("\n FUNCTION CALLING STARTED \n");
input(row,col,row1,col1,n,n1,i1,o1,w1,w2);
forward(row,col,row1,col1,n,n1,i1,o1,w1,w2,neth,outh,neto,outo,E);
backward(row,col,row1,col1,n,n1,i1,o1,w1,w2,neth,outh,neto,outo,E,de, douto, dneto, dE,dEo11, dEo1, dEo22, dEo2, dEt, douth1,dneth1, dE1);

/*deallocate memory*/


free(i1);
free(o1);
free(w1);
free(w2);
free(neth);
free(outh);
free(neto);
free(outo);
free(E);
free(de);
free(douto);
free(dneto);
free(dE);
free(dEo11);
free(dEo1);
free(dEo22);
free(dEo2);
free(douth1);
free(dneth1);
free(dE1);



return 0;
}
