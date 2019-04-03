#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <clocale>

#define BLOCK_SIZE 16
/////Функция записи матрицы в файл
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

/////Функция транспонирования матицы
__global__ void transpose(int* inputMatrix, int* outputMatrix, int width, int height) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	for (int x = 0; x < width; x++)
		for (int y = 0; y < height; y++)
			outputMatrix[x * height + y] = inputMatrix[y * width + x];
}

/////Функция транспонирования матицы на хосте
__host__ void transponse_on_host(int* inputMatrix, int* outputMatrix, int width, int height) {
	for (int y = 0; y < width; y++) {
		for (int x = 0; x < height; x++) {
			outputMatrix[x * width + y]  = inputMatrix[y * width + x];
		}
	}
}

__host__ int main() {
	setlocale(LC_CTYPE, "rus");
	int width;     //Ширина матрицы
	int height;    //Высота матрицы
	printf("Введите кол-во столбцов: ");
	scanf("%d", &width);
	printf("Введите кол-во строк: ");
	scanf("%d", &height);
	int N = width*height;

	//Выделение памяти под матрицы на хосте
	int* A;  //Исходная матрица 
	A = (int *)malloc(sizeof(int) * N);
	int* A_t; //Транспонированная матрица
	A_t = (int *)malloc(sizeof(int) * N);
	int* A_t_host; //Транспонированная матрица на хосте
	A_t_host = (int *)malloc(sizeof(int) * N);

	//Заполняем матрицу данными
	for (int i = 0; i < N; i++) {
		A[i] = i + 1;
	}


	//Записываем исходную матрицу в файл
	SaveMatrixToFile("matrix.txt", A, width, height);

	//Выделение памяти под матрицы на девайсе
	int* A_dev; //Исходная матрица 
	int* A_t_dev; //Транспонированная матрица 

				  //Выделяем глобальную память для храния данных на девайсе
	cudaMalloc((void**)&A_dev, N * sizeof(int));
	cudaMalloc((void**)&A_t_dev, N * sizeof(int));

	//Копируем исходную матрицу с хоста на девайс
	cudaMemcpy(A_dev, A, N * sizeof(int), cudaMemcpyHostToDevice);


	dim3 gridSize = dim3(width / 8, height / 8, 1);
	dim3 blockSize = dim3(8, 8, 1);

	//Начать отсчета времени
	srand(time(0));

	//Запуск ядра 
	transpose << <gridSize, blockSize >> >(A_dev, A_t_dev, width, height);

	//Окончание работы ядра, остановка времени 
	clock_t device_time = clock();


	//Копируем результат с девайса на хост
	cudaMemcpy(A_t, A_t_dev, N * sizeof(int), cudaMemcpyDeviceToHost);

	//Записываем транспонированную матрицу в файл
	SaveMatrixToFile("transpose matrix.txt", A_t, height, width);

	//Удаление ресурсов с видеокарты
	cudaFree(A_dev);
	cudaFree(A_t_dev);


	srand(time(0));
	transponse_on_host(A, A_t_host, width, height);

	//Записываем транспонированную матрицу в файл
	SaveMatrixToFile("transpose matrix on host.txt", A_t_host, height, width);
	clock_t host_time = clock();


	printf("Время работы на устройстве составило: %d мкс\n", device_time * 1000);
	printf("Время работы на хосте составило: %d мкс\n", host_time * 1000);
	printf("Выигрыш времени на устройстве составил: %d мкс\n", (host_time - device_time) * 1000);
	getch();

	//Удаление ресурсов с хоста
	delete[] A;
	delete[] A_t;

	return 0;
}
