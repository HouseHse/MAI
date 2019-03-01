#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include <clocale>

cudaError_t powWithCuda(unsigned int *c, const int *a, unsigned int size);

// ����� ����� � GPU
__global__ void powKernel(unsigned int *c, const int *a)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] * a[i];
}

// ����� ����� � ����������
int main()
{
	setlocale(LC_CTYPE, "rus");

	const int arraySize = 8000;
	int a[arraySize] = { 0 };
	unsigned int c[arraySize] = { 0 };

	for (int i = 0; i < arraySize; i++)
	{
		a[i] = i + 1;
	}

	srand(time(0));

	// Add vectors in parallel.
	cudaError_t cudaStatus = powWithCuda(c, a, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "powWithCuda failed!");
		return 1;
	}
	clock_t device_time = clock();

	printf("������ ������ ������ ����� � ��������� �� �������� \n\n");

	printf("����� ������ 10 � ��������� 5 ����������� ���������� ������� ����� � ������� ���������� �� %d ���������: \n\n", arraySize);
	printf("{1,2,3,4,5,6,7,8,9,10} = \n{%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}\n\n",
		c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]);
	printf("{%d,%d,%d,%d,%d} = \n{%d,%d,%d,%d,%d}\n",
		a[arraySize - 5], a[arraySize - 4], a[arraySize - 3], a[arraySize - 2], a[arraySize - 1],
		c[arraySize - 5], c[arraySize - 4], c[arraySize - 3], c[arraySize - 2], c[arraySize - 1]);

	printf("\n*******************************************\n\n");

	//������ �� �����
	srand(time(0));

	for (int i = 0; i < arraySize; i++)
	{
		c[i] = a[i] * a[i];
	}
	clock_t host_time = clock();

	printf("����� ������ �� ���������� ���������: %d ��� \n", device_time * 1000);
	printf("����� ������ �� ����� ���������: %d ��� \n", host_time * 1000);
	printf("������� ������� �� ���������� ��������: %d ��� \n", (host_time - device_time) * 1000);

	printf("\n*******************************************\n\n");

	printf("�������� ������ �� ����������: \n\n");

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	for (int device = 0; device < deviceCount; device++) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("����� ����������: %d\n", device);
		printf("��� ����������: %s\n", deviceProp.name);
		printf("����� ���������� ������: %d\n", deviceProp.totalGlobalMem);
		printf("����� shared-������ � ����� : %d\n", deviceProp.sharedMemPerBlock);
		printf("����� ����������� ������: %d\n", deviceProp.regsPerBlock);
		printf("������ warp'a: %d\n", deviceProp.warpSize);
		printf("������ ���� ������: %d\n", deviceProp.memPitch);
		printf("���� ���������� ������� � �����: %d\n", deviceProp.maxThreadsPerBlock);

		printf("������������ ����������� ������: x = %d, y = %d, z = %d\n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);

		printf("������������ ������ �����: x = %d, y = %d, z = %d\n",
			deviceProp.maxGridSize[0],
			deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);

		printf("�������� �������: %d\n", deviceProp.clockRate);
		printf("����� ����� ����������� ������: %d\n", deviceProp.totalConstMem);
		printf("�������������� ��������: %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("�������� ����������� ������������ : %d\n", deviceProp.textureAlignment);
		printf("���������� �����������: %d\n\n", deviceProp.multiProcessorCount);
	}
	printf("\n*******************************************\n\n");


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t powWithCuda(unsigned int *c, const int *a, unsigned int size)
{
	int *dev_a = 0; // dev - ��������� �� GPU
	unsigned int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0); // ���������, ��� �������� �� "0"-� �����, �.�. ���������
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(unsigned int)); // �������� ������ �� ����������
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int)); // �������� ������ �� ����������
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice); // �������� �������� ���������� � ����� �� GPU 
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	// Launch a kernel on the GPU with one thread for each element.
	dim3 block(512, 1);
	dim3 grid((size / 512 + 1), 1);
	powKernel << <grid, block >> > (dev_c, dev_a); // ������ ������� � ����������� (size - ������ �������)


														  // Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mulKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching mulKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);

	return cudaStatus;
}

