#include <chrono>
//#include <format>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>
#include <mpi.h>

class Timer {
public:
    Timer() {
        Reset();
    }

    void Reset() {
        start = MPI_Wtime();
    }

    double GetDurationSec() const {
        return MPI_Wtime() - start;
    }

private:
    double start = 0.0;
};

struct Coordinate {
    Coordinate(int coords[]) {
        i = coords[0];
        j = coords[1];
    }
    int i;
    int j;
};

class Matrix {
private:
    std::vector<double> Data;
    size_t Rows{ 0 };
    size_t Cols{ 0 };

public:
    Matrix() = default;

    Matrix(const Matrix& other) = default;

    Matrix(Matrix&& other) = default;

    Matrix(double* buffer, int count, size_t rows, size_t cols) : Data(buffer, buffer + count) {
        if (count <= 0) {
            throw std::invalid_argument("Count is zero");
        }
        Rows = rows;
        Cols = cols;
    }

    Matrix(size_t rows, size_t cols) {
        Rows = rows;
        Cols = cols;
        Data.resize(rows * cols, 0.0f);
    }

    Matrix(size_t rows, size_t cols, double value) {
        Rows = rows;
        Cols = cols;
        Data.resize(rows * cols, value);
    }

    Matrix(const std::vector<double>& vector, size_t rows, size_t cols) {
        if (vector.size() != rows * cols) {
            throw std::out_of_range("Vector size does not match matrix size");
        }
        Data = vector;
        Rows = rows;
        Cols = cols;
    }

    void Resize(size_t rows, size_t cols, double value) {
        Rows = rows;
        Cols = cols;
        Data.resize(rows * cols, value);
    }

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            Data = other.Data;
            Rows = other.Rows;
            Cols = other.Cols;
        }
        return *this;
    }

    double& At(size_t i, size_t j) {
        if (i >= Rows || j >= Cols) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return Data[i * Cols + j];
    }

    const double& At(size_t i, size_t j) const {
        if (i >= Rows || j >= Cols) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return Data[i * Cols + j];
    }

    void Print() const {
        for (size_t i = 0; i < GetRowsCount(); i++) {
            for (size_t j = 0; j < GetColsCount(); j++) {
                std::cout << At(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    size_t GetRowsCount() const {
        return Rows;
    }

    size_t GetColsCount() const {
        return Cols;
    }

    double* Raw() {
        return Data.data();
    }

    const double* Raw() const {
        return Data.data();
    }
};

struct Distribution {
    std::vector<int> Sizes;
    std::vector<int> Offsets;

    int GetSizeAt(int rank) const {
        return Sizes.at(rank);
    }

    int GetOffsetAt(int rank) const {
        return Offsets.at(rank);
    }

    Distribution() = default;

    Distribution(int size, int splitSize) {
        Calculate(size, splitSize);
    }

    Distribution(size_t matrixCols, size_t matrixRows, int rowsSplitCount) {
        Calculate(matrixCols, matrixRows, rowsSplitCount);
    }

    void Calculate(int size, int splitSize) {
        Sizes.resize(size);
        Offsets.resize(size);

        const int base = size / splitSize;
        const int remaider = size % splitSize;

        int offset{ 0 };
        for (int i = 0; i < size; i++) {
            int n = base + (i < remaider ? 1 : 0);
            Sizes[i] = n;
            Offsets[i] = offset;
            offset += n;
        }
    }

    void Calculate(size_t matrixCols, size_t matrixRows, int rowsSplitCount) {
        Sizes.resize(rowsSplitCount);
        Offsets.resize(rowsSplitCount);
        const int base = matrixRows / rowsSplitCount;
        const int reminder = matrixRows % rowsSplitCount;

        int offset{ 0 };

        for (int i = 0; i < rowsSplitCount; i++) {
            const int rowsPerProcess = base + (i < reminder ? 1 : 0);
            const int numsPerProcess = matrixCols * rowsPerProcess;
            Sizes[i] = numsPerProcess;
            Offsets[i] = offset;
            offset += numsPerProcess;
        }
    }
};

template <typename T>
void PrintVector(const std::vector<T>& vector) {
    for (int i = 0; i < vector.size() - 1; i++) {
        std::cout << vector[i] << " ";
    }
    std::cout << vector[vector.size() - 1] << std::endl;
}

std::string CheckAnswer(const Matrix& matrix, size_t expectedValue, size_t expectedCols, size_t expectedRows) {
    if (matrix.GetRowsCount() != expectedRows) {
        return "Rows count mismatch";
    }
    if (matrix.GetColsCount() != expectedCols) {
        return "Cols count mismatch";
    }
    for (int i = 0; i < matrix.GetRowsCount(); i++) {
        for (int j = 0; j < matrix.GetColsCount(); j++) {
            if (matrix.At(i, j) != expectedValue) {
                return "Value does not match, excpected " + std::to_string(expectedValue) + ", got " + std::to_string(matrix.At(i, j));
            }
        }
    }
    return "OK";
}

// std::string PrintResult(double time, std::string status, const Matrix& a, size) {
//
// }

std::string GetCurrentDateTime() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    return std::ctime(&now_time);
}

Matrix Multiply(const Matrix& A, const Matrix& B, int rank, int size) {
    if (rank < 0 || size < 0) {
        throw std::invalid_argument("rank must be non-negative");
    }

    int isValidMatrices = rank == 0 ? A.GetColsCount() == B.GetRowsCount() : 0;
    MPI_Bcast(&isValidMatrices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!isValidMatrices) {
        throw std::invalid_argument("Invalid matrix size");
    }

    int dims[] = { 0, 0 };
    MPI_Dims_create(size, 2, dims);
    //PrintVector(dimensions);
    int periods[] = { 0, 0 };
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    Coordinate rankCoordinate(coords);


    return Matrix(0, 0);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Matrix A;
    Matrix B;
    if (rank == 0) {
        A.Resize(100, 100, 1);
        B.Resize(100, 100, 1);
    }

    Matrix result = Multiply(A, B, rank, size);

    result.Print();

    MPI_Finalize();
}
