
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

constexpr auto BLOCK_DIM = 32;

template <typename T>
__global__ void mm_kernel(T const* a, T const* b, T* c, size_t m, size_t n, size_t p)
{
	__shared__ T a_tile[BLOCK_DIM][BLOCK_DIM];
	__shared__ T b_tile[BLOCK_DIM][BLOCK_DIM];

	T acc_sum{ 0 };

	for (size_t tile_idx{ 0 }; tile_idx < ceilf(static_cast<float>(n) / BLOCK_DIM); ++tile_idx)
	{
		size_t i{ blockIdx.y * blockDim.y + threadIdx.y };
		size_t j{ tile_idx * blockDim.x + threadIdx.x };

		if ((i < m) && (j < n))
			a_tile[threadIdx.y][threadIdx.x] = a[i * n + j];
		else
			a_tile[threadIdx.y][threadIdx.x] = 0;

		i = tile_idx * blockDim.y + threadIdx.y;
		j = blockIdx.x * blockDim.x + threadIdx.x;

		if ((i < n) && (j < p))
			b_tile[threadIdx.y][threadIdx.x] = b[i * p + j];
		else
			b_tile[threadIdx.y][threadIdx.x] = 0;

		__syncthreads();
		for (size_t k{ 0 }; k < BLOCK_DIM; ++k)
			acc_sum += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
		__syncthreads();
	}

	size_t i{ blockIdx.y * blockDim.y + threadIdx.y };
	size_t j{ blockIdx.x * blockDim.x + threadIdx.x };

	if ((i < m) && (j < p))
		c[i * p + j] = acc_sum;
}

template <typename T>
void mm_cuda(T const* a, T const* b, T* c, size_t m, size_t n, size_t p)
{
	dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
	dim3 blocks_per_grid(1, 1);
	blocks_per_grid.x = static_cast<unsigned int>(std::ceil(static_cast<double>(p / threads_per_block.x)));
	blocks_per_grid.y = static_cast<unsigned int>(std::ceil(static_cast<double>(m / threads_per_block.y)));

	mm_kernel<<<blocks_per_grid, threads_per_block>>>(a, b, c, m, n, p);
}

template <typename T>
class Matrix
{
private:
	std::vector<T> data;
	size_t rows, columns;
	static float last_multiplication_time;

	void clear()
	{
		data.clear();
		rows = 0;
		columns = 0;
	}

public:
	Matrix()
		:
		rows(0),
		columns(0),
		data()
	{}

	Matrix(const size_t& rows, const size_t& columns)
		:
		rows(rows),
		columns(columns),
		data(rows * columns, 0)
	{}

	Matrix(const std::string& filepath)
	{
		read_file(filepath);
	}

	Matrix(const Matrix& other)
		:
		rows(other.rows),
		columns(other.columns),
		data(other.data)
	{}

	Matrix(Matrix&& other) noexcept
		:
		rows(other.rows),
		columns(other.columns),
		data(other.data)
	{
		other.data.clear();
		other.rows = 0;
		other.columns = 0;
	}

	~Matrix()
	{
		clear();
	}


	Matrix& operator= (Matrix other)
	{
		std::swap(data, other.data);
		std::swap(rows, other.rows);
		std::swap(columns, other.columns);
		return *this;
	}

	T operator() (const size_t& row, const size_t& column) const
	{
		if (row >= rows)
			throw std::out_of_range("Wrong row index: " + std::to_string(rows));

		if (column >= columns)
			throw std::out_of_range("Wrong column index: " + std::to_string(columns));

		return data[row * columns + column];
	}

	T& operator() (const size_t& row, const size_t& column)
	{
		if (row >= rows)
			throw std::out_of_range("Wrong row index: " + std::to_string(rows));

		if (column >= columns)
			throw std::out_of_range("Wrong column index: " + std::to_string(columns));

		return data[row * columns + column];
	}

	friend Matrix operator* (Matrix& a, Matrix& b)
	{
		if (a.columns != b.rows)
			throw std::logic_error("Multiplication is impossible: mismatch in matrix A columns and matrix B rows");

		Matrix c(a.columns, b.rows);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0);

		T const* a_data{ a.data.data() };
		T const* b_data{ b.data.data() };
		T* c_data{ c.data.data() };

		T* aa, * bb, * cc;
		cudaMalloc((void**)&aa, sizeof(T) * a.rows * a.columns);
		cudaMalloc((void**)&bb, sizeof(T) * b.rows * b.columns);
		cudaMalloc((void**)&cc, sizeof(T) * a.rows * b.columns);

		cudaMemcpy(aa, a_data, sizeof(T) * a.data.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(bb, b_data, sizeof(T) * a.data.size(), cudaMemcpyHostToDevice);

		mm_cuda(aa, bb, cc, a.rows, a.columns, b.columns);

		cudaDeviceSynchronize();
		cudaError_t err{ cudaGetLastError() };
		if (err != cudaSuccess)
			throw std::runtime_error(cudaGetErrorString(err))

		cudaMemcpy(c_data, cc, sizeof(T) * c.data.size(), cudaMemcpyDeviceToHost);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaFree(aa);
		cudaFree(bb);
		cudaFree(cc);
		
		cudaEventElapsedTime(&last_multiplication_time, start, stop);
		return c;
	}


	void read_file(const std::string& filepath)
	{
		std::ifstream file;
		file.exceptions(std::ifstream::badbit);
		file.open(filepath);

		std::vector<std::vector<T>> matrix;
		for (std::string buffer; getline(file, buffer);)
		{
			std::stringstream iss(buffer);

			T value{ 0 };
			std::vector<T> temp;
			while (iss >> value)
				temp.push_back(value);

			matrix.push_back(temp);
		}
		file.close();

		if (matrix.empty())
			throw std::logic_error("No matrix in file \"" + filepath + '\"');

		const size_t column_size = matrix.begin()->size();
		for (auto iter = matrix.begin() + 1; iter != matrix.end(); iter++)
		{
			if (iter->size() != column_size)
				throw std::logic_error("Matrix A and B order mismatch");
		}

		clear();
		rows = matrix.size();
		columns = column_size;
		data = std::vector<T>(rows * columns);
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < columns; j++)
				(*this)(i, j) = matrix[i][j];
		}
	}

	void write_file(const std::string& filepath) const
	{
		std::ofstream file;
		file.exceptions(std::ofstream::badbit);
		file.open(filepath);

		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < columns; j++)
				file << (*this)(i, j) << ' ';
			file << '\n';
		}

		file.close();
	}

	void write_multiplication_result(const std::string& filepath) const
	{
		std::ofstream file;
		file.exceptions(std::ofstream::badbit);
		file.open(filepath, std::ofstream::app);

		file << '\n' << "Runtime" << ' ' << last_multiplication_time << ' ' << " seconds" << '\n';
		file << "Volume" << ' ' << rows * columns;

		file.close();
	}
};