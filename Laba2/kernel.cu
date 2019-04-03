#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <clocale>

#define BLOCK_SIZE 16
/////������� ������ ������� � ����
__host__ void SaveMatrixToFile(char* fileName, int* matrix, int width, int height) {
	FILE* file = fopen(fileName, "wt");
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			fprintf(file, "%d\t", matrix[y * width + x]);
		}
		fprintf(file, "\n");
	}
	fclose(file);
}

/////������� ���������������� ������
__global__ void transpose(int* inputMatrix, int* outputMatrix, int width, int height) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	for (int x = 0; x < width; x++)
		for (int y = 0; y < height; y++)
			outputMatrix[x * height + y] = inputMatrix[y * width + x];
}

/////������� ���������������� ������ �� �����
__host__ void transponse_on_host(int* inputMatrix, int* outputMatrix, int width, int height) {
	for (int y = 0; y < width; y++) {
		for (int x = 0; x < height; x++) {
			outputMatrix[x * width + y]  = inputMatrix[y * width + x];
		}
	}
}

__host__ int main() {
	setlocale(LC_CTYPE, "rus");
	int width;     //������ �������
	int height;    //������ �������
	printf("������� ���-�� ��������: ");
	scanf("%d", &width);
	printf("������� ���-�� �����: ");
	scanf("%d", &height);
	int N = width*height;

	//��������� ������ ��� ������� �� �����
	int* A;  //�������� ������� 
	A = (int *)malloc(sizeof(int) * N);
	int* A_t; //����������������� �������
	A_t = (int *)malloc(sizeof(int) * N);
	int* A_t_host; //����������������� ������� �� �����
	A_t_host = (int *)malloc(sizeof(int) * N);

	//��������� ������� �������
	for (int i = 0; i < N; i++) {
		A[i] = i + 1;
	}


	//���������� �������� ������� � ����
	SaveMatrixToFile("matrix.txt", A, width, height);

	//��������� ������ ��� ������� �� �������
	int* A_dev; //�������� ������� 
	int* A_t_dev; //����������������� ������� 

				  //�������� ���������� ������ ��� ������ ������ �� �������
	cudaMalloc((void**)&A_dev, N * sizeof(int));
	cudaMalloc((void**)&A_t_dev, N * sizeof(int));

	//�������� �������� ������� � ����� �� ������
	cudaMemcpy(A_dev, A, N * sizeof(int), cudaMemcpyHostToDevice);


	dim3 gridSize = dim3(width / 8, height / 8, 1);
	dim3 blockSize = dim3(8, 8, 1);

	//������ ������� �������
	srand(time(0));

	//������ ���� 
	transpose << <gridSize, blockSize >> >(A_dev, A_t_dev, width, height);

	//��������� ������ ����, ��������� ������� 
	clock_t device_time = clock();


	//�������� ��������� � ������� �� ����
	cudaMemcpy(A_t, A_t_dev, N * sizeof(int), cudaMemcpyDeviceToHost);

	//���������� ����������������� ������� � ����
	SaveMatrixToFile("transpose matrix.txt", A_t, height, width);

	//�������� �������� � ����������
	cudaFree(A_dev);
	cudaFree(A_t_dev);


	srand(time(0));
	transponse_on_host(A, A_t_host, width, height);

	//���������� ����������������� ������� � ����
	SaveMatrixToFile("transpose matrix on host.txt", A_t_host, height, width);
	clock_t host_time = clock();


	printf("����� ������ �� ���������� ���������: %d ���\n", device_time * 1000);
	printf("����� ������ �� ����� ���������: %d ���\n", host_time * 1000);
	printf("������� ������� �� ���������� ��������: %d ���\n", (host_time - device_time) * 1000);
	getch();

	//�������� �������� � �����
	delete[] A;
	delete[] A_t;

	return 0;
}