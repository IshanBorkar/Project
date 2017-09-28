#include <stdio.h>
#include<time.h>
#include<stdlib.h>
#include <string.h>
#define n 1000000
int main()
{
    FILE *fp = fopen("/home/mtech/Desktop/Project codes/dataset/click_train.csv","r");
    const char s[2] = ", ";
    char *token;
    int i,a[n];
    if(fp != NULL)
    {
        char line[20];
        while(fgets(line, sizeof line, fp) != NULL)
        {
            token = strtok(line, s);
            for(i=0;i<2;i++)
            {
                if(i==0)
                {   
                    printf("%s \t",token);
                    token = strtok(NULL,s);
                } else {
                    printf("%d\n",atoi(token));
                }       
            }
        }
        fclose(fp);
    } else {
        perror("click_train.csv");
    } 
printf("\nThe inputs are \n"); 
s=a[n];
for(i=0;i<n;i++)
{
printf("%d\n",atoi(token));
}
    return 0; 

}  
