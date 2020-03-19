#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#define CRITICALTHRESHOLD	3
#define WARNINGTHRESHOLD	6
#define CRITICAL	2
#define WARNING		1
#define OK			0
#define MB 1024*1024

char* checkCuda(cudaError_t result){
	if (result != cudaSuccess) {
		printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
	}
	return 0;
}

int main(){
	int res = OK;
	

	int devices = 0;
	checkCuda(cudaGetDeviceCount(&devices));
		
	char pcibus[32];
	int bytes = 1024 * MB;
	int tests = 3;
	for(int i = 0; i < devices; i++){
		cudaDeviceProp prop;
		checkCuda( cudaGetDeviceProperties(&prop, 0) );
		if(strstr(prop.name, "M40") != NULL || strstr(prop.name, "K40") != NULL){
			printf("Does not support %s.\n", prop.name);
			continue;
		}
	}
	
	for(int i = 0; i < devices; i++){
		for(int j = 0; j < devices; j++){
			if(i == j) continue;

			int accessible_i_j = 0;
			checkCuda(cudaDeviceCanAccessPeer(&accessible_i_j, i, j));
			if(!accessible_i_j){
				printf("GPU %d is not able to P2P access to GPU %d.\n", i, j);
				continue;
			}
			
			int accessible_j_i = 0;
			checkCuda(cudaDeviceCanAccessPeer(&accessible_j_i, j, i));
			if(!accessible_j_i){
				printf("GPU %d is not able to P2P access to GPU %d.\n", j, i);
				continue;
			}
			double i_to_j = 0;
			double j_to_i = 0;
			checkCuda(cudaSetDevice(i));
			cudaDeviceEnablePeerAccess(j, 0);
			checkCuda(cudaSetDevice(j));
			cudaDeviceEnablePeerAccess(i, 0);
			char* d_mem_i;
			char* d_mem_j;
			checkCuda(cudaSetDevice(i));
			checkCuda(cudaMalloc( (void**)&d_mem_i, sizeof(char)*bytes));
			checkCuda(cudaSetDevice(j));
			checkCuda(cudaMalloc( (void**)&d_mem_j, sizeof(char)*bytes));
			cudaEvent_t start, stop;
			checkCuda(cudaEventCreate(&start));
			checkCuda(cudaEventCreate(&stop));
			checkCuda(cudaEventRecord(start, 0));
			for(int j = 0; j < tests; j++)
				checkCuda(cudaMemcpy( d_mem_i, d_mem_j, sizeof(char)*bytes, cudaMemcpyDefault ));
			checkCuda(cudaEventRecord(stop, 0));
			checkCuda(cudaEventSynchronize(stop));
			float t;
			checkCuda(cudaEventElapsedTime(&t, start, stop));
			i_to_j = (double)bytes*1e-6/(t/tests);
			checkCuda(cudaEventRecord(start, 0));
			for(int j = 0; j < tests; j++)
				checkCuda(cudaMemcpy( d_mem_j, d_mem_i, sizeof(char)*bytes, cudaMemcpyDefault ));
			checkCuda(cudaEventRecord(stop, 0));
			checkCuda(cudaEventSynchronize(stop));
			checkCuda(cudaEventElapsedTime(&t, start, stop));
			j_to_i = (double)bytes*1e-6/(t/tests);
			printf("GPU %d P2P GPU %d: %2.2fGB/s  GPU %d P2P GPU %d: %2.2fGB/s\n", i, j, i_to_j, j, i, j_to_i);
			checkCuda(cudaSetDevice(i));
			checkCuda(cudaFree(d_mem_i));
			checkCuda(cudaSetDevice(j));
			checkCuda(cudaFree(d_mem_j));
			checkCuda( cudaEventDestroy(start) );
			checkCuda( cudaEventDestroy(stop) );
		}
		
	}
	
	
	return res;
}
