
/* 
    parann
    Copyright (C) 2015 Katelyn McNair

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include "mpi.h"

#define rando() ((double)rand()/RAND_MAX)

	int num_in;               /* 1. num input neurons */
	int num_hn;               /* 2. num hidden neurons */
	int num_on;               /* 3. num output neurons */
	int max_epochs;           /* 4. num max epochs */
	double step_size;         /* 5. step size */
	int num_train_patterns;   /* 6. num training patterns */
	char file_train[20];      /* 7. name of file for 'training' set */
	char file_train_des[20];  /* 8. name of file for 'training-desired' set */
	int num_test_patterns;    /* 9. num of testing patterns */
	char file_test[20];       /* 10. name of file for 'testing' set */
	char file_test_des[20];   /* 11. name of file for 'testing-desired' set */
	char file_err_epoch[20];  /* 12. name of file to write epoch error values */
	char file_output[20];     /* 13. name of file to write output for testing set */

	char c[20];

	FILE *fPtr; /* file ptr */
	FILE *error_file, *train_file, *train_file_des, *test_file, *test_file_des, *output_file;

	double **Input, **Target, **trainset, **traindesset, **testset, **testdesset;
	double **SumH, **Hidden, **SumO, **Output, **WeightIH, **DeltaWeightIH, **WeightHO, **DeltaWeightHO;
	double *DeltaO, *SumDOW, *DeltaH;
	double Error, smallwt = 0.5;

double **allocate_array(int row_dim, int col_dim)
{
  double **result;
  int i;

  /* Allocate an array of pointers to hold pointers to the rows of the
         array */
  result=(double **)malloc(row_dim*sizeof(double *));

  /* The first pointer is in fact a pointer to the entire array */
  result[0]=(double *)malloc(row_dim*col_dim*sizeof(double));

  /* The remaining pointers are just pointers into this array, offset
         in units of col_dim */
  for(i=1; i<row_dim; i++)
        result[i]=result[i-1]+col_dim;

  return result;
}

int main( int argc, char **argv )
{
	int    i, j, k, p, epoch, local_num_patterns, start_patt, end_patt;
	double Out;
	double t1,t2;
	struct timeval tp;
	int rtn;
	/*------------------------------MPI Parameters---------------------------*/
	int rank, size;
	MPI_Status status;
	struct {int numIN,numHN,numON,numEP; double stepsize;} values;
	MPI_Datatype mystruct;
	int blocklens[2]={4,1};
	MPI_Aint base;
	MPI_Aint indices[2];
	MPI_Datatype old_types[2]={MPI_INT,MPI_DOUBLE};
	MPI_Init( &argc, &argv );
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	MPI_Comm_size( MPI_COMM_WORLD, &size );

	MPI_Address( &values.numIN, &indices[0] );
	MPI_Address( &values.stepsize, &indices[1] );
	base=indices[0];
	for(i=0;i<2;i++)
		indices[i]-=base;

	MPI_Type_struct( 2, blocklens, indices, old_types, &mystruct );
	MPI_Type_commit( &mystruct );
	if (rank == 0){
		values.numIN=3;
		values.numHN=4;
		values.numON=5;
		values.numEP=6;
		values.stepsize=0.2;
	}
	/*------------------------------Time Parameters---------------------------*/
	rtn=gettimeofday(&tp, NULL);
	t1=(double)tp.tv_sec+(1.e-6)*tp.tv_usec;
	/*------------------------------Parameter Initializations---------------------------*/
	if ( ( fPtr = fopen("input.par", "r") )  == NULL )
		printf("File could not be opened\n");
	else {
        	fscanf( fPtr, "%d", &num_in ); 
        	fscanf( fPtr, "%d", &num_hn ); 
	        fscanf( fPtr, "%d", &num_on );
        	fscanf( fPtr, "%d", &max_epochs ); 
	        fscanf( fPtr, "%lf", &step_size ); 
        	fscanf( fPtr, "%d", &num_train_patterns ); 
	        fscanf( fPtr, "%s", file_train ); 
	        fscanf( fPtr, "%s", file_train_des ); 
	        fscanf( fPtr, "%d", &num_test_patterns ); 
	        fscanf( fPtr, "%s", file_test ); 
	        fscanf( fPtr, "%s", file_test_des ); 
	        fscanf( fPtr, "%s", file_err_epoch ); 
	        fscanf( fPtr, "%s", file_output ); 
	}
        fclose( fPtr ); /* close file  */
	/*--------------------------- Read in Training Patterns -------------------------------*/	
	trainset = (double **) calloc(1+num_train_patterns,sizeof(double));
	for(i=0;i<=num_train_patterns;i++){
		trainset[i]=(double *) calloc(1+num_in,sizeof(double));
	}
        if ( (train_file=fopen(file_train,"r")) == NULL )
                printf("Training pattern file could not be opened\n");
	for(p=1;p<=num_train_patterns;p++){
		trainset[p][0]=1.0;
		for(i=1;i<=num_in;i++){
			if(fgets(c, 10, train_file)==NULL)
				printf("Error reading in training pattern at line:%d\n",i);
			trainset[p][i]=atof(c);
		}
	}
	fclose( train_file );
	/*--------------------- Read in Training Patterns Desired -----------------------------*/	
        traindesset = (double **) calloc(1+num_train_patterns,sizeof(double));
        for(i=0;i<=num_train_patterns;i++){
                traindesset[i]=(double *) calloc(1+num_on,sizeof(double));
        }
        if ( (train_file_des=fopen(file_train_des,"r")) == NULL )
                printf("Training pattern desired file could not be opened\n");
        for(p=1;p<=num_train_patterns;p++){
                traindesset[p][0]=1.0;
                for(i=1;i<=num_on;i++){
                        if(fgets(c, 10, train_file_des)==NULL)
                                printf("Error reading in training pattern desired at line:%d\n",i);
                        traindesset[p][i]=atof(c);
                }
        }
	fclose( train_file_des );
        /*--------------------------- Read in Testing Patterns -------------------------------*/
        testset = (double **) calloc(1+num_test_patterns,sizeof(double));
        for(i=0;i<=num_test_patterns;i++){
                testset[i]=(double *) calloc(1+num_in,sizeof(double));
        }
        if ( (test_file=fopen(file_test,"r")) == NULL )
                printf("Testing pattern file could not be opened\n");
        for(p=1;p<=num_test_patterns;p++){
                testset[p][0]=1.0;
                for(i=1;i<=num_in;i++){
                        if(fgets(c, 10, test_file)==NULL)
                                printf("Error reading in testing pattern at line:%d\n",i);
                        testset[p][i]=atof(c);
                }
        }
	fclose( test_file );
        /*--------------------- Read in Testing Patterns Desired -----------------------------*/
        testdesset = (double **) calloc(1+num_test_patterns,sizeof(double));
        for(i=0;i<=num_test_patterns;i++){
                testdesset[i]=(double *) calloc(1+num_on,sizeof(double));
        }
        if ( (test_file_des=fopen(file_test_des,"r")) == NULL )
                printf("Testing pattern desired file could not be opened\n");
        for(p=1;p<=num_test_patterns;p++){
                testdesset[p][0]=1.0;
                for(i=1;i<=num_on;i++){
                        if(fgets(c, 10, test_file_des)==NULL)
                                printf("Error reading in testing pattern desired at line:%d\n",i);
                        testdesset[p][i]=atof(c);
                }
        }
	fclose( test_file_des );
	/*--------------------Open Error and Output File For Root Process Only---------------------------*/
	if(rank==0){
		if ( (error_file=fopen(file_err_epoch,"w+")) == NULL )
			printf("Error file could not be opened\n");
	        if ( (output_file=fopen(file_output,"w+")) == NULL )
        	        printf("Error file could not be opened\n");
	}
	/*-------------------------------Initialize Network----------------------------------------------*/
	SumH = (double **) calloc(1+num_train_patterns,sizeof(double));
	Hidden = (double **) calloc(1+num_train_patterns,sizeof(double));
	SumO = (double **) calloc(1+num_train_patterns,sizeof(double));
        Output = (double **) calloc(1+num_train_patterns,sizeof(double));
	for(i=0;i<=num_train_patterns;i++){
                SumH[i]=(double *) calloc(1+num_hn,sizeof(double));
               	Hidden[i]=(double *) calloc(1+num_hn,sizeof(double));
		SumO[i]=(double *) calloc(1+num_on,sizeof(double));
               	Output[i]=(double *) calloc(1+num_on,sizeof(double));
	}
	WeightHO=allocate_array(num_hn+1,num_on+1);
	WeightIH=allocate_array(num_in+1,num_hn+1);
	DeltaWeightHO=allocate_array(num_hn+1,num_on+1);
	DeltaWeightIH=allocate_array(num_in+1,num_hn+1);
        DeltaO = (double *) calloc(1+num_on,sizeof(double));
        SumDOW = (double *) calloc(1+num_hn,sizeof(double));
        DeltaH = (double *) calloc(1+num_hn,sizeof(double));

	/*-------------------------------Initialize Network----------------------------------------------*/
	if(rank==0){
	for( j = 1 ; j <= num_hn ; j++ ){    /* initialize WeightIH and DeltaWeightIH */
		for( i = 0 ; i <= num_in; i++ ){ 
			WeightIH[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
			DeltaWeightIH[i][j] = 0.0 ;
             	}
        }
	for( k = 1 ; k <= num_on; k ++ ) {    /* initialize WeightHO and DeltaWeightHO */
        	for( j = 0 ; j <= num_hn ; j++ ) {
            		WeightHO[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ;
       			DeltaWeightHO[j][k] = 0.0 ;
        	}
    	}
	}

	/*-------------------------------Main Routine----------------------------------------------*/
	Input=trainset;
	Target=traindesset;
	
	for( epoch = 0 ; epoch < max_epochs ; epoch++ ){    /* iterate weight updates */
		MPI_Bcast(*WeightHO,(num_hn+1)*(num_on+1),MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(*WeightIH,(num_in+1)*(num_hn+1),MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if(rank!=0){
			for( j = 1 ; j <= num_hn ; j++ ){    /* initialize DeltaWeightIH */
        			for( i = 0 ; i <= num_in ; i++ ){
            				DeltaWeightIH[i][j] = 0.0 ;
				}
			}
       			for( k = 1 ; k <= num_on; k ++ ){    /* initialize DeltaWeightHO */
       				for( j = 0 ; j <= num_hn ; j++ ){
       					DeltaWeightHO[j][k] = 0.0 ;
				}
			}
	        	Error = 0.0;
			/*------------------------Set Up where each processes pattern set is---------------*/
			local_num_patterns=ceil((float)num_train_patterns/(size-1));
                        start_patt=(rank-1)*local_num_patterns+1;
                        end_patt=start_patt+local_num_patterns-1;
                        if(end_patt>num_train_patterns) end_patt=num_train_patterns;

        		for( p = start_patt ; p <= local_num_patterns; p++ ){ /* repeat for all the training patterns */
				for( j = 1 ; j <= num_hn ; j++ ){ /* compute hidden unit activations */
					SumH[p][j] = 0;
        	        		for( i = 0 ; i <= num_in; i++ ) {
		        	            SumH[p][j] += Input[p][i] * WeightIH[i][j];
        		        	}
                			Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j]));
				}
				Hidden[p][0] = 1.0; 
				for( k = 1 ; k <= num_on; k++ ) {    /* compute output unit activations and errors */
					SumO[p][k] = 0 * WeightHO[0][k] ;
        	        		for( j = 0 ; j <= num_hn ; j++ ) {
						SumO[p][k] += Hidden[p][j] * WeightHO[j][k] ;
					}
	                		Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k])) ;   /* Sigmoidal Outputs */
			/*		Output[p][k] = SumO[p][k];  Linear Outputs */
					Error += 0.5 * (Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k]) ;   /* SSE */
			/*		Error -= (Target[p][k]*log(Output[p][k])+(1.0-Target[p][k])*log(1.0-Output[p][k]));Cross-Entropy Error */
					DeltaO[k] = (Target[p][k]-Output[p][k])*Output[p][k]*(1.0-Output[p][k]); /* Sigmoidal Outputs, SSE */
			/*		DeltaO[k] = Target[p][k] - Output[p][k];   Sigmoidal Outputs, Cross-Entropy Error */
			/*		DeltaO[k] = Target[p][k] - Output[p][k];   Linear Outputs, SSE */
				}
	        		for( j = 1 ; j <= num_hn ; j++ ) {    /* 'back-propagate' errors to hidden layer */
					SumDOW[j] = 0.0 ;
					for( k = 1 ; k <= num_on; k++ ) {
        	        	    		SumDOW[j] += WeightHO[j][k] * DeltaO[k] ;
	        	        	}
        	        		DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
                	                for( i = 0 ; i <= num_in; i++ ){
                        	                DeltaWeightIH[i][j] += step_size * Input[p][i] * DeltaH[j];
                                	}
	                        }
        	                for( k = 0 ; k <= num_on; k ++ ){    /* update weights WeightHO */
                	                for( j = 1 ; j <= num_hn ; j++ ){
                        	                DeltaWeightHO[j][k] += step_size * Hidden[p][j] * DeltaO[k] ;
                                	}
	                        }
			}//end of PATTERN
			/*-------------------Send the Delta Weights, and the error hitches a ride--------------------*/
			DeltaWeightHO[0][0]=Error; 
			MPI_Send(*DeltaWeightIH,(num_in+1)*(num_hn+1),MPI_DOUBLE,0,13,MPI_COMM_WORLD);
			MPI_Send(*DeltaWeightHO,(num_hn+1)*(num_on+1),MPI_DOUBLE,0,13,MPI_COMM_WORLD);
		}else{
			Error=0.0;
			for(p=1;p<size;p++){
				/*-------------------Recieve the Delta Weights, and the error-------------------------*/
				MPI_Recv(*DeltaWeightIH,(num_in+1)*(num_hn+1),MPI_DOUBLE,p,13,MPI_COMM_WORLD,&status);
				MPI_Recv(*DeltaWeightHO,(num_hn+1)*(num_on+1),MPI_DOUBLE,p,13,MPI_COMM_WORLD,&status);
				Error+=DeltaWeightHO[0][0];	
				for( k = 0 ; k <= num_on; k ++ ){    /* update weights for batch mode */
					for( j = 1 ; j <= num_hn ; j++ ){
						WeightHO[j][k] += DeltaWeightHO[j][k] ;
					}
				} 
				for( j = 1 ; j <= num_hn ; j++ ){    /* update weights for batch mode */
					for( i = 0 ; i <= num_in; i++ ){
						WeightIH[i][j] += DeltaWeightIH[i][j] ;
					}
				}
			}
			fprintf(error_file,"%d\t%f\n",epoch,Error);
		}
	}//end of EPOCH
	/*-----------------------------------Now that the network is up run the test patterns though------------------*/
	if(rank==0){
		fclose(error_file);
        	if ( (output_file=fopen(file_output,"w+")) == NULL )
                	printf("Error file could not be opened\n");
	        Input=testset;
        	Target=testdesset;
		for(p=1;p<=num_test_patterns;p++){
		        for( j = 1 ; j <= num_hn ; j++ ){
			        SumH[p][j] = 0;
                		for( i = 0 ; i <= num_in; i++ ) {
                	        	SumH[p][j] += Input[p][i] * WeightIH[i][j];
	                	}
				Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j]));
			}
	        	Hidden[p][0] = 1.0;
        		for( k = 1 ; k <= num_on; k++ ){
                		SumO[p][k] = 0 * WeightHO[0][k] ;
                		for( j = 0 ; j <= num_hn ; j++ ) {
                        		SumO[p][k] += Hidden[p][j] * WeightHO[j][k] ;
 		               	}
                		Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k]));
				fprintf(output_file,"%f\t%f\n",Target[p][k],Output[p][k]);
	        	}
		}
	}
	/*-------------------------------------------Time---------------------------------------------*/
	rtn=gettimeofday(&tp, NULL);
	t2=(double)tp.tv_sec+(1.e-6)*tp.tv_usec;
	printf("Time:%5.5f\n",t2-t1);
	return 1 ;
}

