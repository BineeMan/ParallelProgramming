#include <chrono>
#include <format>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>
#include <mpi.h>

class Timer {
public:
    void Reset() {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    }

    Timer() {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    }

    double GetDurationSec() const {
        struct timespec end;
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        return end.tv_sec - start.tv_sec + 0.000000001 * (end.tv_nsec - start.tv_nsec);
    }

private:
    static std::chrono::time_point<std::chrono::system_clock> GetCurrentTime() {
        return std::chrono::high_resolution_clock::now();
    }

    struct timespec start;
};

class Matrix {
private:
    std::vector<double> Data;
    size_t Rows;
    size_t Cols;

public:
    Matrix() = default;

    Matrix(const Matrix& other) = default;

    Matrix(Matrix&& other) = default;

    Matrix(double *buffer, int count, size_t rows, size_t cols) : Data(buffer, buffer + count) {
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

    Matrix Multiply(const Matrix& other) const {
        if (GetCols() != other.GetRows()) {
            throw std::invalid_argument("Matrix dimensions mismatch");
        }

        Matrix result(GetRows(), other.GetCols());

        for (size_t i = 0; i < GetRows(); i++) {
            for (size_t j = 0; j < other.GetCols(); j++) {
                double sum{ 0.0f };
                for (size_t k = 0; k < GetCols(); k++) {
                    sum += this->At(i, k) * other.At(k, j);
                }
                result.At(i, j) = sum;
            }
        }
        return result;
    }

    Matrix Add(const Matrix& other) const {
        CheckSize(other);

        Matrix result(GetRows(), GetCols());
        for (size_t i = 0; i < GetRows(); i++) {
            for (size_t j = 0; j < GetCols(); j++) {
                result.At(i, j) = this->At(i, j) + other.At(i, j);
            }
        }

        return result;
    }

    Matrix Subtract(const Matrix& other) const {
        CheckSize(other);

        Matrix result(GetRows(), GetCols());
        for (size_t i = 0; i < GetRows(); i++) {
            for (size_t j = 0; j < GetCols(); j++) {
                result.At(i, j) = this->At(i, j) - other.At(i, j);
            }
        }

        return result;
    }

    Matrix MultiplyScalar(double scalar) const {
        Matrix result(GetRows(), GetCols());
        for (size_t i = 0; i < GetRows(); i++) {
            for (size_t j = 0; j < GetCols(); j++) {
                result.At(i, j) = this->At(i, j) * scalar;
            }
        }
        return result;
    }

    std::vector<double> operator*(const std::vector<double>& other) const {
        if (other.size() != GetCols()) {
            throw std::invalid_argument("Vector dimensions mismatch");
        }

        std::vector<double> result(GetRows(), 0.0f);

        for (int i = 0; i < result.size(); i++) {
            for (size_t j = 0; j < GetCols(); j++) {
                result[i] += this->At(i, j) * other[j];
            }
        }
        return result;
    }

    size_t GetRows() const {
        return Rows;
    }

    size_t GetCols() const {
        return Cols;
    }

    double* Raw() {
        return Data.data();
    }

    const double* Raw() const {
        return Data.data();
    }

    void Print() const {
        for (size_t i = 0; i < GetRows(); i++) {
            for (size_t j = 0; j < GetCols(); j++) {
                std::cout << At(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    bool IsSameSize(const Matrix& other) const {
        return GetCols() == other.GetCols() && GetRows() == other.GetRows();
    }

    void CheckSize(const Matrix& other) const {
        if (!IsSameSize(other)) {
            throw std::out_of_range("Matrix dimensions mismatch");
        }
    }

    bool IsSquare() const {
        return GetRows() == GetCols();
    }
};

template<typename T>
void PrintVector(const std::vector<T>& vector) {
    for (T element: vector) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

void FillArray(const std::vector<double>& vector) {
}

std::vector<double> operator*(const std::vector<double>& A, const std::vector<double>& B) {
    if (A.size() != B.size()) {
        throw std::invalid_argument("Vector dimensions mismatch");
    }
    std::vector<double> result(A.size(), 0.0f);
    for (size_t i = 0; i < A.size(); i++) {
        result[i] = A[i] * B[i];
    }
    return result;
}

std::vector<double> operator-(const std::vector<double>& A, const std::vector<double>& B) {
    if (A.size() != B.size()) {
        throw std::invalid_argument("Vector dimensions mismatch");
    }
    std::vector<double> result(A.size(), 0.0f);
    for (size_t i = 0; i < A.size(); i++) {
        result[i] = A[i] - B[i];
    }
    return result;
}

std::vector<double> operator*(double scalar, const std::vector<double>& vector) {
    std::vector<double> result(vector.size());
    for (size_t i = 0; i < vector.size(); i++) {
        result[i] = scalar * vector[i];
    }
    return result;
}

double Norm2(const std::vector<double>& vector) {
    double sum{ 0.0f };
    for (double element: vector) {
        sum += element * element;
    }
    return std::sqrt(sum);
}

std::vector<double> SolveLinearEquation(const Matrix& A, const std::vector<double>& b, double epsilon) {
    const size_t size = b.size();
    if (!A.IsSquare() || b.size() != A.GetRows()) {
        throw std::invalid_argument("!matrix.IsSquare() || rightPart.size() != matrix.GetRows()");
    }
    double tay{ 0.01 };
    std::vector<double> x(size);

    double normB = Norm2(b);

    while ( Norm2(A * x - b) / normB >= epsilon ) {
        x = x - tay * (A * x - b);
    }
    return x;
}

std::vector<double> SolveLinearEquation_Mpi1(const Matrix& A, const std::vector<double>& b,
    double epsilon, int rank, int size) {
    if (rank <= 0 || size <= 0) {
        throw std::runtime_error("rank <= 0 || size <= 0");
    }
    if (!A.IsSquare() || b.size() != A.GetRows()) {
        throw std::invalid_argument("!matrix.IsSquare() || rightPart.size() != matrix.GetRows()");
    }


    std::vector<double> x(b.size());
    const double normB = Norm2(b);

    std::vector<double> Ax = A * x;

    const int base = A.GetRows() / size;
    const int reminder = A.GetRows() % size;

    std::vector<int> sendCounts(size);
    std::vector<int> offsets(size);
    for (int i = 0; i < size; i++) {
        const int rowsPerProcess = base + (i < reminder ? 1 : 0);

        sendCounts[i] = A.GetCols() * rowsPerProcess;
        offsets[i] = 0;
    }
    //const int rowsPerProcess = base + (rank < reminder ? 1 : 0);
    //const int startRow = rank * base + std::min(reminder, rank);

    Matrix localA(rowsPerProcess, A.GetCols());

   // MPI_Scatterv

    while ( Norm2(Ax - b) / normB >= epsilon ) {
        const double tay{ 0.01 };

        x = x - tay * (Ax - b);
        Ax = A * x;
    }
    return x;
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //int N = 12000;
    int N = 100;
    Matrix matrix(N, N, 1.0);
    if (rank == 0) {
     for (int i = 0; i < N; i++) {
         matrix.At(i, i) = 2.0;
     }
    }

    const double epsilon{ std::pow(10, -5) };
    std::vector<double> b(N, N + 1);
    Timer timer;
    auto res = SolveLinearEquation_Mpi1(matrix, b, epsilon, rank, size);
    //auto res = SolveLinearEquation(matrix, b, epsilon);
    std::cout << "Time: " << timer.GetDurationSec() << std::endl;
    PrintVector(res);
    MPI_Finalize();
}
