#include<stdio.h>
//#include<omp.h>
#include<math.h>
#include<stdlib.h>

int main(int argc , char *argv[])
{
int nthreads,tid;
float i1[20],o1[20];
float b1=0.35,b2=0.60;
float w1[20],w2[20],neth[20],neto[20],outh[20],outo[20],E[20];
int i,n,n1;
float x=0,eta=0.5;
float de[20],douto[20],dneto[20],dE[20];
float dEo111[20],dEo11[20],dneto11[20],dEo1[20],dneto12[20];
float dEo222[20],dEo22[20],dEo2[20];
float dEt[20],douth1[20],dneth1[20],dE1[20];
/*#pragma omp parallel 
{
tid = omp_get_thread_num();
   printf("\n Running thread = %d\n", tid);
 if (tid == 0) 
     {
     nthreads = omp_get_num_threads();
     printf("\n Number of threads = %d\n", nthreads);
     }
}*/
/**INPUT LAYER**/
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

printf("\n The input entered is \n");
 for(i=0;i<n;i++)
{
printf("\ti1=%f\t",i1[i]);
}

printf("\n The target output entered is \n");
 for(i=0;i<n;i++)
{
printf("\to1=%f\t",o1[i]);
}

printf("\nThe biases are \t");
printf("\nb1=%f\t b2=%f\n",b1,b2);

/**HIDDEN LAYER**/
n=pow(n,2);
printf("\nEnter the number of weights(hidden layer) ");
for(i=0;i<n;i++)
{
scanf("%f",&w1[i]);
}

printf("\n The weights entered are(hidden layer) \n");
 for(i=0;i<n;i++)
{
printf("\t w=%f\t",w1[i]);
}

/**OUTPUT LAYER**/
n1=pow(n,2);
printf("\nEnter the number of weights(output layer) ");
for(i=0;i<n;i++)
{
scanf("%f",&w2[i]);
}

printf("\n The weights entered are(output layer)\n");
 for(i=0;i<n;i++)
{
printf("\t w=%f\t",w2[i]);
}

printf("\n*******************Forward Pass*************************************\n");

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
printf("\t Etotal=%f\t",E[i]);
 printf("\n*********************Backward Pass******************************\n");


/** OUTPUT LAYER**/
for(i=0;i<n;i++)
de[i]=-(o1[i]-outo[i]);
 
printf("\n dEtotal/douto1 is \n ");
 for(i=0;i<n;i++)
 printf("\t de=%f\t",de[i]);

for(i=0;i<n;i++)
 douto[i]=outo[i]*(1-outo[i]);


printf("\n douto1/dneto1 is \n");
for(i=0;i<n;i++)
printf("\t douto1=%f\t",douto[i]);

for(i=0;i<n;i++)
dneto[i]=outh[i];

printf("\n dneto1/dw5 is \n");
for(i=0;i<n;i++)
 printf("\t dneto1=%f\t",dneto[i]);

for(i=0;i<n;i++){
dE[i]=de[i]*douto[i]*dneto[i];
}

printf("\n The value of dEtotal/dw5 is \n");
for(i=0;i<n;i++) 
printf("\t dE=%f\t",dE[i]);

 printf("\nUpdating value of weights in output layer to hidden layer \n");
for(i=0;i<n;i++) 
 w2[i]=w2[i]-(eta*dE[i]);
for(i=0;i<n;i++)  
printf("\t w=%f\t",w2[i]); 
printf("\n*************************************************\n");

/** HIDDEN LAYER**/
printf("\nHidden Layer \n");
for(i=0;i<n;i++)
dEo111[i]=-(o1[i]-outo[i]);
for(i=0;i<n;i++)
dEo11[i]=dEo111[i]*douto[i];

printf("\n dEo1/dneto1 is \n");
for(i=0;i<n;i++){
printf("\t dE011=%f\t",dEo11[i]);
}

for(i=0;i<n;i++)
dneto11[i]=w2[i];
for(i=0;i<n;i++)
dEo1[i]=dEo11[i]*dneto11[i];

printf("\n dEo1/douth1 is \n");
for(i=0;i<n;i++)
{printf("\t dEo1=%f\t", dEo1[i]);
}

for(i=0;i<n;i++)
dneto12[i]=w2[i];

for(i=0;i<n;i++)
dEo222[i]=-(o1[i]-outo[i]);

for(i=0;i<n;i++)
douto[i]=outo[i]*(1-outo[i]);

for(i=0;i<n;i++)
dEo22[i]=dEo222[i]*douto[i];


printf("\n dEo2/dneto2 is \n");
for(i=0;i<n;i++)
{
printf("\t dEo22=%f\t",dEo22[i]);
}

for(i=0;i<n;i++)
dEo2[i]=dEo22[i]*dneto12[i];

printf("\n dEo2/douth1 is \n");
for(i=0;i<n;i++)
{printf("\t  dEo2=%f\t",dEo2[i]);
}
printf("\n**********************************************\n");

for(i=0;i<n;i++)
dEt[i]=dEo1[i]+dEo2[i];

printf("\n dEtotal/douth1 is \n");
for(i=0;i<n;i++)
{
printf("\t dEt1=%f\t",dEt[i]);
}


for(i=0;i<n;i++)
douth1[i]=outh[i]*(1-outh[i]);

printf("\n douth1/dneth1 is \n");
for(i=0;i<n;i++){
printf("\t douth1=%f\t",douth1[i]);
}

for(i=0;i<n;i++)
dneth1[i]=i1[i];

printf("\n dneth1/dw1 is \n");
for(i=0;i<n;i++){
printf("\t  dneth1=%f\t",dneth1[i]);
}

for(i=0;i<n;i++)
dE1[i]=dEt[i]*douth1[i]*dneth1[i];

printf("\n dEtotal/dw1 is \n");
for(i=0;i<n;i++){
printf("\t dE1=%f\t",dE1[i]);
}
 
printf("\nUpdating value of weights in hidden layer to input layer  \n");
for(i=0;i<n;i++)
w1[i]=w1[i]-(eta*dE1[i]);

for(i=0;i<n;i++)
{ printf("\t W=%f\t",w1[i]); 
}
printf("\n**********************************************\n");
printf("\nUpdated all the weights \n");
for(i=0;i<n;i++)
{
printf("\t w= %f\t",w1[i]);
}
 for(i=0;i<n;i++)
{
printf("\t w= %f\t",w2[i]);
}

printf("\n**********************************************\n");

return 0;
}
