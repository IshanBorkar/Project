#include<stdio.h>
#include<omp.h>
#include<math.h>
#include<stdlib.h>
#include<sys/time.h>
#define COL 95100

int readresults(FILE *results, float value[]);
int forward(FILE *results,float value[],float neth[], float outh[], float neto[],float outo[],float E[]);
int backward(FILE *results,float value[],float neth[], float outh[], float neto[],float outo[],float E[],float de[],float douto[],float dneto[],float dE[],float dEo11[],float dEo1[],float dEo22[],float dEo2[],float dEt[],float douth1[],float dneth1[],float dE1[],float dEo111[],float dneto11[],float dneto12[],float dEo222[]);

int main()
{
        FILE *f=fopen("/home/ruchika/Desktop/click_24.txt", "r");
	int  col, item;        
	float value[COL];
	float neth[COL],outh[COL],neto[COL],outo[COL],E[COL];
	float de[COL],douto[COL],dneto[COL], dE[COL],dEo11[COL], dEo1[COL], dEo22[COL], dEo2[COL],dEt[COL],douth1[COL], dneth1[COL];
	float dE1[COL],dEo111[COL],dneto11[COL],dneto12[COL],dEo222[COL];
	float b1=0.35,b2=0.60;
	float x=0,eta=0.5;

	int 	nthreads,tid;
	struct 	timeval  TimeValue_Start;
        struct 	timezone TimeZone_Start;
        struct 	timeval  TimeValue_Final;
        struct 	timezone TimeZone_Final;
        long    time_start, time_end;
        double  time_overhead;

	gettimeofday(&TimeValue_Start, &TimeZone_Start);
 
	#pragma omp parallel private(tid) shared(col)
	{
	tid = omp_get_thread_num();
  	printf("\n Running thread = %d\n", tid);
 	if (tid == 0) 
     {
     nthreads = omp_get_num_threads();
     printf("\n Number of threads = %d\n", nthreads);
     }
       }

	
	#pragma omp parallel for shared(col)
	for(col=0;col<11;col++)
	{
   readresults(f,  value);
   forward(f,value,neth,outh,neto,outo,E);
   backward(f,value,neth,outh,neto,outo,E,de,douto,dneto,dE,dEo11,dEo1,dEo22,dEo2,dEt,douth1,dneth1,dE1,dEo111,dneto11,dneto12,dEo222);
	}
	
	gettimeofday(&TimeValue_Final, &TimeZone_Final);
        time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
        time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
        time_overhead = (time_end - time_start)/1000000.0;

	printf("\nTime Taken in Seconds for Parallel Computation  for 24 MB dataset for %d threads    :%lf \n  ", time_overhead,nthreads);


return 0;
}


 
readresults(FILE *results,float value[])
{
        int  col, item;
		
	#pragma omp parallel for shared(col)
        for(col=0; col<COL; col++)
        {	 
           
                item = fscanf(results, "%f%f%f%f%f", &value[col], &value[col+1],&value[col+2],&value[col+3],&value[col+4]);
		
                printf("Display-Id[%d]=%f , Clicked(o1)[%d]=%f , Input(i1)[%d]=%f , Input_weight(w1)[%d]=%f , Output_weight(w2)[%d]=%f \n", col, value[col],col+1,value[col+1],col+2, value[col+2],col+3, value[col+3], col+4, value[col+4]);
		col=col+4;
          
        }
	
        return 0;
}



forward(FILE *results,float value[],float neth[], float outh[], float neto[],float outo[],float E[])
{ 	
	int col,item;
	float b1=0.35,b2=0.60,x=0;
	
	printf("\n*******************Forward Pass**************************\n");

     		 #pragma omp parallel for shared(col) 
      
	for(col=0;col<COL;col++)
	{ 

          neth[col]=(value[col+2]*value[col+3])+b1*1;
          col=col+4;
	}
		#pragma omp parallel for shared(col) 
	for(col=0;col<COL;col++)
	{
        	outh[col]=1/(1+(exp(x-neth[col])));
		col=col+4;	
	}
	
	printf("\n");
		 #pragma omp parallel for shared(col) 
     
	for(col=0;col<COL;col++)
	{
        neto[col]=(value[col+2]*value[col+4])+b2*1;
	col=col+4;
	}
	 	#pragma omp parallel for shared(col) 
 	for(col=0;col<COL;col++)
	{
        outo[col]=1/(1+(exp(x-neto[col])));
	col=col+4;	
	}

	
	printf("\n");
	 	#pragma omp parallel for shared(col) 
     
	for(col=0;col<COL;col++)
	{
	E[col]=((value[col+1]-outo[col])*(value[col+1]-outo[col]))/2;
	col=col+4;
	}
    	
	printf("\n");
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{ 
	printf("\t neth[%d]=%f \t outh[%d]=%f \n",col,neth[col],col,outh[col]);
	col=col+4;
	printf("\t neto[%d]=%f \t outo[%d]=%f\n ",col,neto[col],col,outo[col]);
	col=col+4;
	printf("\t Etotal[%d]=%f \n ",col,E[col]);
	col=col+4;
	}
	
return 0;
}


backward(FILE *results,float value[],float neth[], float outh[], float neto[],float outo[],float E[],float de[],float douto[],float dneto[],float dE[],float dEo11[],float dEo1[],float dEo22[],float dEo2[],float dEt[],float douth1[],float dneth1[],float dE1[],float dEo111[],float dneto11[],float dneto12[],float dEo222[])
{	int col,item;
	float b1=0.35,b2=0.60;
	float x=0,eta=0.5;
	
	printf("\n*********************Backward Pass******************************\n");
	
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{ 	
	de[col]=-(value[col+1]-outo[col]);
	col=col+4;
	}
		
	printf("\tdEtotal/douto1 is \n");
		#pragma omp parallel for shared(col)
	 for(col=0;col<COL;col++)
	{ 
	printf("\t de=%f \n",de[col]);
	col=col+4;
	}
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{ 
	douto[col]=outo[col]*(1-outo[col]);
	col=col+4;
	}
		
	//printf("\tdouto1/dneto1 is \n");
	
	 	#pragma omp parallel for  
	for(col=0;col<COL;col++)
	{ 
       	  dneto[col]=outh[col];
	col=col+4;
	}
		
	//printf("\tdneto1/dw5 is \n");
	
	 	 #pragma omp parallel for  
	for(col=0;col<COL;col++)
	{
	dE[col]=de[col]*douto[col]*dneto[col];
	col=col+4;
	}
		
	//printf("\tThe value of dEtotal/dw5 is \n");
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{ 
	printf("\t douto1=%f \n",douto[col]);
	col=col+4;
	 printf("\t dneto1=%f \n",dneto[col]);
	col=col+4;
	printf("\t dE=%f \n",dE[col]);
	col=col+4;
	}
	
	 printf("\nUpdating value of weights in output layer to hidden layer \n");
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{ 
	 value[col]=(value[col+4]-(eta*dE[col]));
	col=col+4;	
	}
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{ 
	printf("\t Updated_Output_weight(W2)=%f\n",value[col]); 
	col=col+4;
	}
	
	
	printf("\n*************************************************\n");

	/** HIDDEN LAYER**/
	printf("\nHidden Layer \n");
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	dEo111[col]=-(value[col+1]-outo[col]);
	col=col+4;	
	}
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{ dEo11[col]=dEo111[col]*douto[col];
	col=col+4;	
	}
		
	//printf("\n dEo1/dneto1 is \n");

		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	dneto11[col]=value[col+4];
	col=col+4;
	}
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	dEo1[col]=dEo11[col]*dneto11[col];
	col=col+4;	
	}
	
	//printf("\n dEo1/douth1 is \n");
	
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	dneto12[col]=value[col+4];
	col=col+4;	
	}
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	dEo222[col]=-(value[col+1]-outo[col]);
	col=col+4;
	}
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	douto[col]=outo[col]*(1-outo[col]);
	col=col+4;
	}
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	dEo22[col]=dEo222[col]*douto[col];
	col=col+4;
	}
		
	//printf("\n dEo2/dneto2 is \n");
	
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	dEo2[col]=dEo22[col]*dneto12[col];
	col=col+4;	
	}
		
	//printf("\n dEo2/douth1 is \n");
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	printf("\t dEo11=%f\n",dEo11[col]);
	col=col+4;
	printf("\t dEo1=%f\n", dEo1[col]);
	col=col+4;
	printf("\t dEo22=%f\n",dEo22[col]);
	col=col+4;
	printf("\t dEo2=%f\n",dEo2[col]);
	col=col+4;	
	}
	
	printf("\n**********************************************\n");
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	dEt[col]=dEo1[col]+dEo2[col];
	col=col+4;	
	}
		
	printf("\n dEtotal/douth1 is \n");
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	printf("\t  dEt1=%f\n",dEt[col]);
	col=col+4;	
	}
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	douth1[col]=outh[col]*(1-outh[col]);
	col=col+4;
	}
		
	printf("\n douth1/dneth1 is \n");
		#pragma omp parallel for shared(col)	
	for(col=0;col<COL;col++)
	{
	printf("\t  douth1=%f\n",douth1[col]);
	col=col+4;	
	}
	
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	dneth1[col]=value[col];
	col=col+4;
	}
		
	printf("\n  dneth1/dw1 is \n");
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	printf("\t dneth1=%f\n",dneth1[col]);
	col=col+4;	
	}
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	dE1[col]=dEt[col]*douth1[col]*dneth1[col];
	col=col+4;	
	}
		
	printf("\n dEtotal/dw1 is \n");
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	printf("\t  dE1=%f\n",dE1[col]);
	col=col+4;	
	}
	 
	printf("\nUpdating value of weights in hidden layer to input layer  \n");
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	value[col]=(value[col+3]-(eta*dE1[col]));
	col=col+4;	
	}
		#pragma omp parallel for shared(col)
	for(col=0;col<COL;col++)
	{
	 printf("\t Updated_Input_weight(W1)=%f\n",value[col]); 
	col=col+4;	
	}
	
	printf("\n**********************************************\n");

return 0;
}











