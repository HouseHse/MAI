
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include <clocale>
#include <windows.h>

#define CUDA_DEBUG

#ifdef CUDA_DEBUG

#define CUDA_CHECK_ERROR(err)           \
if (err != cudaSuccess) {          \
printf("Cuda error: %s\n", cudaGetErrorString(err));    \
printf("Error in file: %s, line: %i\n", __FILE__, __LINE__);  \
}                 \

#else

#define CUDA_CHECK_ERROR(err)

#endif

// Функция транспонирования матрицы без использования разделяемой памяти
// 
// inputMatrix - указатель на исходную матрицу 
// outputMatrix - указатель на матрицу результат
// width - ширина исходной матрицы (она же высота матрицы-результата)
// height - высота исходной матрицы (она же ширина матрицы-результата)
//
__global__ void transposeMatrixSlow(float* inputMatrix, float* outputMatrix, int width, int height)
{
	int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

	if ((xIndex < width) && (yIndex < height))
	{
		//Линейный индекс элемента строки исходной матрицы  
		int inputIdx = xIndex + width * yIndex;

		//Линейный индекс элемента столбца матрицы-результата
		int outputIdx = yIndex + height * xIndex;

		outputMatrix[outputIdx] = inputMatrix[inputIdx];
	}
}

#define BLOCK_DIM 16

// Функция транспонирования матрицы c использования разделяемой памяти
// 
// inputMatrix - указатель на исходную матрицу 
// outputMatrix - указатель на матрицу результат
// width - ширина исходной матрицы (она же высота матрицы-результата)
// height - высота исходной матрицы (она же ширина матрицы-результата)
//
__global__ void transposeMatrixFast(float* inputMatrix, float* outputMatrix, int width, int height)
{
	__shared__ float temp[BLOCK_DIM][BLOCK_DIM];

	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if ((xIndex < width) && (yIndex < height))
	{
		// Линейный индекс элемента строки исходной матрицы  
		int idx = yIndex * width + xIndex;

		//Копируем элементы исходной матрицы
		temp[threadIdx.y][threadIdx.x] = inputMatrix[idx];
	}

	//Синхронизируем все нити в блоке
	__syncthreads();

	xIndex = blockIdx.y * blockDim.y + threadIdx.x;
	yIndex = blockIdx.x * blockDim.x + threadIdx.y;

	if ((xIndex < height) && (yIndex < width))
	{
		// Линейный индекс элемента строки исходной матрицы  
		int idx = yIndex * height + xIndex;

		//Копируем элементы исходной матрицы
		outputMatrix[idx] = temp[threadIdx.x][threadIdx.y];
	}
}

// Функция транспонирования матрицы, выполняемая на CPU
__host__ void transposeMatrixCPU(float* inputMatrix, float* outputMatrix, int width, int height)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			outputMatrix[x * height + y] = inputMatrix[y * width + x];
		}
	}
}

__host__ void printMatrixToFile(char* fileName, float* matrix, int width, int height)
{
	FILE* file = fopen(fileName, "wt");
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			fprintf(file, "%.0f\t", matrix[y * width + x]);
		}
		fprintf(file, "\n");
	}
	fclose(file);
}


#define GPU_SLOW 1
#define GPU_FAST 2
#define CPU 3

#define ITERATIONS 20    //Количество нагрузочных циклов

__host__ int main()
{
	int width = 2048;    //Ширина матрицы
	int height = 1056;    //Высота матрицы

	int matrixSize = width * height;
	int byteSize = matrixSize * sizeof(float);

	//Выделяем память под матрицы на хосте
	float* inputMatrix = new float[matrixSize];
	float* outputMatrix = new float[matrixSize];

	//Заполняем исходную матрицу данными
	for (int i = 0; i < matrixSize; i++)
	{
		inputMatrix[i] = i;
	}

	//Выбираем способ расчета транспонированной матрицы
	printf("Select compute mode: 1 - Slow GPU, 2 - Fast GPU, 3 - CPU\n");
	int mode;
	scanf("%i", &mode);

	//Записываем исходную матрицу в файл
	printMatrixToFile("before.txt", inputMatrix, width, height);

	if (mode == CPU)    //Если используеться только CPU
	{
		int start = GetTickCount();
		for (int i = 0; i < ITERATIONS; i++)
		{
			transposeMatrixCPU(inputMatrix, outputMatrix, width, height);
		}
		//Выводим время выполнения функции на CPU (в миллиекундах)
		printf("CPU compute time: %i\n", GetTickCount() - start);
	}
	else  //В случае расчета на GPU
	{
		float* devInputMatrix;
		float* devOutputMatrix;

		//Выделяем глобальную память для храния данных на девайсе
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devInputMatrix, byteSize));
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devOutputMatrix, byteSize));

		//Копируем исходную матрицу с хоста на девайс
		CUDA_CHECK_ERROR(cudaMemcpy(devInputMatrix, inputMatrix, byteSize, cudaMemcpyHostToDevice));

		//Конфигурация запуска ядра
		dim3 gridSize = dim3(width / BLOCK_DIM, height / BLOCK_DIM, 1);
		dim3 blockSize = dim3(BLOCK_DIM, BLOCK_DIM, 1);

		cudaEvent_t start;
		cudaEvent_t stop;

		//Создаем event'ы для синхронизации и замера времени работы GPU
		CUDA_CHECK_ERROR(cudaEventCreate(&start));
		CUDA_CHECK_ERROR(cudaEventCreate(&stop));

		//Отмечаем старт расчетов на GPU
		cudaEventRecord(start, 0);

		if (mode == GPU_SLOW)    //Используеться функция без разделяемой памяти
		{
			for (int i = 0; i < ITERATIONS; i++)
			{

				transposeMatrixSlow << <gridSize, blockSize >> >(devInputMatrix, devOutputMatrix, width, height);
			}
		}
		else if (mode == GPU_FAST)  //Используеться функция с разделяемой памятью
		{
			for (int i = 0; i < ITERATIONS; i++)
			{

				transposeMatrixFast << <gridSize, blockSize >> >(devInputMatrix, devOutputMatrix, width, height);
			}
		}

		//Отмечаем окончание расчета
		cudaEventRecord(stop, 0);

		float time = 0;
		//Синхронизируемя с моментом окончания расчетов
		cudaEventSynchronize(stop);
		//Рассчитываем время работы GPU
		cudaEventElapsedTime(&time, start, stop);

		//Выводим время расчета в консоль
		printf("GPU compute time: %.0f\n", time);

		//Копируем результат с девайса на хост
		CUDA_CHECK_ERROR(cudaMemcpy(outputMatrix, devOutputMatrix, byteSize, cudaMemcpyDeviceToHost));

		//
		//Чистим ресурсы на видеокарте
		//

		CUDA_CHECK_ERROR(cudaFree(devInputMatrix));
		CUDA_CHECK_ERROR(cudaFree(devOutputMatrix));

		CUDA_CHECK_ERROR(cudaEventDestroy(start));
		CUDA_CHECK_ERROR(cudaEventDestroy(stop));
	}

	//Записываем матрицу-результат в файл
	printMatrixToFile("after.txt", outputMatrix, height, width);

	//Чистим память на хосте
	delete[] inputMatrix;
	delete[] outputMatrix;

	return 0;
}
