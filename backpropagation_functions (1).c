#include<stdio.h>
//#include<omp.h>
#include<math.h>
#include<stdlib.h>

void input(int n, float i1[20],float o1[20],float w1[20],float w2[20])
{

        float b1=0.35,b2=0.60;
	int i,n1,n2;
	printf("\n The input entered is \n");
	 for(i=0;i<n;i++)
	{	
		printf("%f\t",i1[i]);
	}

	printf("\n The target output entered is \n");
	 for(i=0;i<n;i++)
	{
	printf("%f\t",o1[i]);
	}

	printf("\nThe biases are \t");
	printf("\nb1=%f\t b2=%f\n",b1,b2);
	n1=pow(n,2);
	printf("\n The weights entered are(hidden layer) \n");
	 for(i=0;i<n1;i++)
	{
	printf("%f\t",w1[i]);
	}
        n2=pow(n,2);
	printf("\n The weights entered are(output layer)\n");
	 for(i=0;i<n2;i++)
	{
	printf("%f\t",w2[i]);
	}
}

void forward()
{
	//int nthreads,tid;
	float i1[20],o1[20];
	float b1=0.35,b2=0.60;
	float w1[20],w2[20],neth[20],neto[20],outh[20],outo[20],E[20];
	int i,n,n1;
	float x=0,eta=0.5;
	float de[20],douto[20],dneto[20],dE[20];
	float dEo111[20],dEo11[20],dneto11[20],dEo1[20],dneto12[20];
	float dEo222[20],dEo22[20],dEo2[20];
	float dEt[20],douth1[20],dneth1[20],dE1[20];
	
	for(i=0;i<n;i++)
	{
	neth[i]=w1[i]*i1[i]+b1*1;
	}
	printf("\nThe total net input (hidden layer)  \n");
	for(i=0;i<n;i++) 
	printf("\tneth=%f\t",neth[i]);

	for(i=0;i<n;i++)
	{
	outh[i]=1/(1+(exp(x-neth[i])));
	} 
	printf("\nThe output for hidden layer is \n");
	for(i=0;i<n;i++) 
	printf("\touth=%f\t",outh[i]);

	for(i=0;i<n;i++)
	{
	neto[i]=w2[i]*i1[i]+b2*1;
	}
	printf("\nThe total net input (output layer)  \n");
	for(i=0;i<n;i++) 
	printf("\tneto=%f\t",neto[i]);

	for(i=0;i<n;i++)
	{
	outo[i]=1/(1+(exp(x-neto[i])));
	} 
	printf("\nThe output for output layer is \n");
	for(i=0;i<n;i++) 
	printf("\touto=%f\t",outo[i]);

	for(i=0;i<n;i++)
	{
	E[i]=((o1[i]-outo[i])*(o1[i]-outo[i]))/2;
	}
	printf("\n");
	for(i=0;i<n;i++)
	printf(" Etotal=%f\t",E[i]);
}


void backward()
{
	//int nthreads,tid;
	float i1[20],o1[20];
	float b1=0.35,b2=0.60;
	float w1[20],w2[20],neth[20],neto[20],outh[20],outo[20],E[20];
	int i,n,n1;
	float x=0,eta=0.5;
	float de[20],douto[20],dneto[20],dE[20];
	float dEo111[20],dEo11[20],dneto11[20],dEo1[20],dneto12[20];
	float dEo222[20],dEo22[20],dEo2[20];
	float dEt[20],douth1[20],dneth1[20],dE1[20];
	
	for(i=0;i<n;i++)
	de[i]=-(o1[i]-outo[i]);
	 
	printf("\n dEtotal/douto1 is ");
	 for(i=0;i<n;i++)
	 printf("\n de=%f\t",de[i]);

	for(i=0;i<n;i++)
	 douto[i]=outo[i]*(1-outo[i]);


	printf("\n douto1/dneto1 is ");
	for(i=0;i<n;i++)
	printf("\n douto1=%f\t",douto[i]);

	for(i=0;i<n;i++)
	dneto[i]=outh[i];

	for(i=0;i<n;i++)
	 printf("\ndneto1/dw5 is dneto1=%f",dneto[i]);

	for(i=0;i<n;i++){
	dE[i]=de[i]*douto[i]*dneto[i];
	}
	for(i=0;i<n;i++) 
	printf("\nThe value of dEtotal/dw5 is dE=%f\n",dE[i]);

	 printf("\nUpdating value of weights in output layer to hidden layer \n");
	for(i=0;i<n;i++) 
	 w2[i]=w2[i]-(eta*dE[i]);
	for(i=0;i<n;i++)  
	printf("\nw=%f\n",w2[i]); 
	printf("\n*************************************************\n");

	/** HIDDEN LAYER**/
	printf("\nHidden Layer \n");
	for(i=0;i<n;i++)
	dEo111[i]=-(o1[i]-outo[i]);
	for(i=0;i<n;i++)
	dEo11[i]=dEo111[i]*douto[i];
	for(i=0;i<n;i++){
	printf("\ndEo1/dneto1 is  dE011=%f",dEo11[i]);
	}

	for(i=0;i<n;i++)
	dneto11[i]=w2[i];
	for(i=0;i<n;i++)
	dEo1[i]=dEo11[i]*dneto11[i];

	for(i=0;i<n;i++)
	{printf("\n dEo1/douth1 is dEo1=%f", dEo1[i]);
	}

	for(i=0;i<n;i++)
	dneto12[i]=w2[i];

	for(i=0;i<n;i++)
	dEo222[i]=-(o1[i]-outo[i]);

	for(i=0;i<n;i++)
	douto[i]=outo[i]*(1-outo[i]);

	for(i=0;i<n;i++)
	dEo22[i]=dEo222[i]*douto[i];

	for(i=0;i<n;i++)
	{
	printf("\n dEo2/dneto2 is dEo22=%f",dEo22[i]);
	}

	for(i=0;i<n;i++)
	dEo2[i]=dEo22[i]*dneto12[i];

	for(i=0;i<n;i++)
	{printf("\n dEo2/douth1 is dEo2=%f",dEo2[i]);
	}
	printf("\n**********************************************\n");

	for(i=0;i<n;i++)
	dEt[i]=dEo1[i]+dEo2[i];

	for(i=0;i<n;i++)
	{
	printf("\n dEtotal/douth1 is dEt1=%f",dEt[i]);
	}

	for(i=0;i<n;i++)
	douth1[i]=outh[i]*(1-outh[i]);

	for(i=0;i<n;i++){
	printf("\n douth1/dneth1 is douth1=%f",douth1[i]);
	}

	for(i=0;i<n;i++)
	dneth1[i]=i1[i];

	for(i=0;i<n;i++){
	printf("\n dneth1/dw1 is dneth1=%f",dneth1[i]);
	}

	for(i=0;i<n;i++)
	dE1[i]=dEt[i]*douth1[i]*dneth1[i];

	for(i=0;i<n;i++){
	printf("\n dEtotal/dw1 is dE1=%f",dE1[i]);
	}
	 
	printf("\nUpdating value of weights in hidden layer to input layer  \n");
	for(i=0;i<n;i++)
	w1[i]=w1[i]-(eta*dE1[i]);

	for(i=0;i<n;i++)
	{ printf("\nW=%f\n",w1[i]); 
	}
	printf("\n**********************************************\n");
	printf("\nUpdated all the weights");
	for(i=0;i<n;i++)
	{
	printf("\n w= %f \n",w1[i]);
	}
	 for(i=0;i<n;i++)
	{
	printf("\n w= %f \n",w2[i]);
	}

	printf("\n**********************************************\n");

}


int main()
{
float i1[20],o1[20],b1=0.35,b2=0.60;
int i,n,n1,n2;
float w1[20], w2[20];

printf("\nEnter the limit \n ");
scanf("%d",&n);
printf("\n Enter the inputs \n ");
for(i=0;i<n;i++)
{
scanf("%f",&i1[i]);
}
printf("\n Enter the target outputs \n");
for(i=0;i<n;i++)
{
scanf("%f",&o1[i]);
}
n1=pow(n,2);
printf("\nEnter the number of weights(hidden layer) ");
for(i=0;i<n1;i++)
{
scanf("%f",&w1[i]);
}


n2=pow(n,2);
printf("\nEnter the number of weights(output layer) ");
for(i=0;i<n2;i++)
{
scanf("%f",&w2[i]);
}

printf("\nFUNCTION CALLING STARTED\n");
input(n,i1,o1,w1,w2);
//forward();
//backward();

return 0;
}
