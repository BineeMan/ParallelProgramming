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
        struct timespec end{ };
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        return end.tv_sec - start.tv_sec + 0.000000001 * (end.tv_nsec - start.tv_nsec);
    }

private:
    struct timespec start{ };
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
        if (GetCols() > other.size()) {
            throw std::invalid_argument("Vector dimensions mismatch");
        }

        std::vector<double> result(GetRows(), 0.0f);

        for (int i = 0; i < GetRows(); i++) {
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

    double GetMaxRowSum() const {
        double maxSum{ 0.0f };
        for (size_t i = 0; i < GetRows(); i++) {
            double currentSum{ 0.0f };
            for (size_t j = 0; j < GetCols(); j++) {
                currentSum += std::abs(this->At(i, j));
            }
            maxSum = std::max(maxSum, currentSum);
        }
        return maxSum;
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

template<typename T>
void PrintVector(const std::vector<T>& vector) {
    for (T element: vector) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

double operator*(const std::vector<double>& A, const std::vector<double>& B) {
    if (A.size() != B.size()) {
        throw std::invalid_argument("Vector dimensions mismatch");
    }
    double result = 0;
    for (size_t i = 0; i < A.size(); i++) {
        result += A[i] * B[i];
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

std::vector<double>& operator-=(std::vector<double>& A, const std::vector<double>& B) {
    if (A.size() != B.size()) {
        throw std::invalid_argument("Vector dimensions mismatch");
    }
    for (size_t i = 0; i < A.size(); i++) {
        A[i] -= B[i];
    }
    return A;
}

std::string CheckAnswer(const std::vector<double>& answers, double expectedAnswer) {
    for (size_t i = 0; i < answers.size(); i++) {
        if (std::fabs(answers[i] - expectedAnswer) > 1e-9) {
            // std::cout << "Wrong answer: expected " << expectedAnswer << ", got " << answers[i] << " at index " << i <<
            //         std::endl;
            return "Wrong answer: expected " + std::to_string(expectedAnswer) + ", got " + std::to_string(answers[i]) + " at index " + std::to_string(i);
        }
    }
    //std::cout << "OK" << std::endl;
    return "OK";
}

double Norm2(const std::vector<double>& vector) {
    double sum{ 0.0f };
    for (double element: vector) {
        sum += element * element;
    }
    return std::sqrt(sum);
}

std::vector<double> SolveLinearEquation_Mpi1(const Matrix& matrixA, const std::vector<double>& vecB,
                                             double epsilon, int rank, int size) {
    if (rank < 0 || size <= 0) {
        throw std::runtime_error(
            "rank <= 0 or size <= 0. Rank = " +
            std::to_string(rank) + ", size = " + std::to_string(size));
    }

    int isMatrixValid = rank == 0 ? matrixA.IsSquare() && vecB.size() == matrixA.GetRows() : 0;
    MPI_Bcast(&isMatrixValid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!isMatrixValid) {
        throw std::invalid_argument("!matrix.IsSquare() || rightPart.size() != matrix.GetRows()");
    }

    const double normB = Norm2(vecB);
    if (normB == 0) {
        throw std::runtime_error("normB == 0");
    }

    size_t kRowsCountMatrixA = rank == 0 ? matrixA.GetRows() : 1;
    MPI_Bcast(&kRowsCountMatrixA, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    size_t kColsCountMatrixA = rank == 0 ? matrixA.GetCols() : -1;
    MPI_Bcast(&kColsCountMatrixA, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    // Offsets
    const int base = kRowsCountMatrixA / size;
    const int reminder = kRowsCountMatrixA % size;

    std::vector<int> sendCounts(size);
    std::vector<int> offsets(size);

    std::vector<int> axSendCounts(size);
    std::vector<int> axOffsets(size);

    int offset = 0;
    int axOffset = 0;

    for (int i = 0; i < size; i++) {
        const int rowsPerProcess = base + (i < reminder ? 1 : 0);
        const int numsPerProcess = kColsCountMatrixA * rowsPerProcess;
        sendCounts[i] = numsPerProcess;
        offsets[i] = offset;
        offset += numsPerProcess;

        axSendCounts[i] = rowsPerProcess;
        axOffsets[i] = axOffset;
        axOffset += rowsPerProcess;
    }

    const int rowsPerCurrentProcess = base + (rank < reminder ? 1 : 0);
    Matrix localA(rowsPerCurrentProcess, kColsCountMatrixA);

    MPI_Scatterv(matrixA.Raw(), sendCounts.data(),
                 offsets.data(), MPI_DOUBLE,
                 localA.Raw(), rowsPerCurrentProcess * kColsCountMatrixA,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> Ax(vecB.size());
    std::vector<double> localAx(rowsPerCurrentProcess);
    std::vector<double> AxMinusB(Ax.size());

    std::vector<double> x(vecB.size(), 0.0);

    const int startRowForCurrentProcess = axOffsets[rank];

    double tau = rank == 0 ? 1.0 / matrixA.GetMaxRowSum() : 0;
    MPI_Bcast(&tau, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    while (true) {
        // 1) vector Ax = A * x
        localAx = localA * x;

        MPI_Allgatherv(localAx.data(), localAx.size(),
                       MPI_DOUBLE, Ax.data(), axSendCounts.data(),
                       axOffsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // 2) vector Ax - b
        AxMinusB = Ax - vecB;
        double localSum = 0;
        for (int i = 0; i < rowsPerCurrentProcess; i++) {
            const int globalIdx = startRowForCurrentProcess + i;
            localSum += AxMinusB[globalIdx] * AxMinusB[globalIdx];
        }
        double globalSum = 0;
        MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // 3) Невязка
        if (std::sqrt(globalSum) / normB < epsilon) {
            break;
        }

        // Новый x
        x -= tau * AxMinusB;
    }
    return x;
}

std::vector<double> SolveLinearEquation_Mpi2(const Matrix& matrixA, const std::vector<double>& vecB,
                                             double epsilon, int rank, int size) {
    if (rank < 0 || size <= 0) {
        throw std::runtime_error(
            "rank <= 0 or size <= 0. Rank = " + std::to_string(rank) + ", size = " + std::to_string(size));
    }

    int isMatrixValid = rank == 0 ? matrixA.IsSquare() && vecB.size() == matrixA.GetRows() : 0;
    MPI_Bcast(&isMatrixValid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!isMatrixValid) {
        throw std::invalid_argument("!matrix.IsSquare() || rightPart.size() != matrix.GetRows()");
    }

    double normB = rank == 0 ? Norm2(vecB) : -1;
    MPI_Bcast(&normB, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (normB <= 0) {
        throw std::runtime_error("normB == 0");
    }

    size_t kRowsCountMatrixA = rank == 0 ? matrixA.GetRows() : 1;
    MPI_Bcast(&kRowsCountMatrixA, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    size_t kVectorSize = rank == 0 ? vecB.size() : -1;
    MPI_Bcast(&kVectorSize, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    // matrixA->localMatrixA
    size_t matrixCols = rank == 0 ? matrixA.GetCols() : -1;
    MPI_Bcast(&matrixCols, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    size_t matrixRows = rank == 0 ? matrixA.GetRows() : -1;
    MPI_Bcast(&matrixRows, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    const Distribution aDistribution(matrixCols, matrixRows, size);
    const int rowsPerCurrentProcess = aDistribution.GetSizeAt(rank) / matrixRows;

    Matrix localMatrixA(rowsPerCurrentProcess, matrixCols);
    MPI_Scatterv(matrixA.Raw(), aDistribution.Sizes.data(),
                 aDistribution.Offsets.data(), MPI_DOUBLE,
                 localMatrixA.Raw(), rowsPerCurrentProcess * matrixCols,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // b->localB
    const Distribution vectorDistribution(kVectorSize, size);
    std::vector<double> localVecB(vectorDistribution.GetSizeAt(rank));
    MPI_Scatterv(vecB.data(), vectorDistribution.Sizes.data(),
                 vectorDistribution.Offsets.data(), MPI_DOUBLE,
                 localVecB.data(), localVecB.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Preallocate data for calculation
    std::vector<double> vecAx(kVectorSize);
    std::vector<double> localVecAx(rowsPerCurrentProcess);

    std::vector<double> vecX(kVectorSize, 0.0);
    std::vector<double> localVecX(vectorDistribution.GetSizeAt(rank), 0.0);

    Distribution axDistibution(kRowsCountMatrixA, size);

    //const int startRowForCurrentProcess = axDistibution.Offsets.at(rank);
    double tau = rank == 0 ? 1.0 / matrixA.GetMaxRowSum() : 0;
    MPI_Bcast(&tau, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    while (true) {
        // 1) vector Ax = A * x
        localVecAx = localMatrixA * vecX;

        // 2) Норма ||Ax - b||
        double localNorm = 0;
        for (int i = 0; i < rowsPerCurrentProcess; i++) {
            const double r = localVecAx.at(i) - localVecB.at(i);
            localNorm += r * r;
        }
        double norm = 0;
        MPI_Allreduce(&localNorm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // 3) Невязка
        if (std::sqrt(norm) / normB < epsilon) {
            break;
        }

        // 4) Обновление x
        localVecX -= tau * (localVecAx - localVecB);
        MPI_Allgatherv(localVecX.data(), localVecX.size(),
                       MPI_DOUBLE, vecX.data(), axDistibution.Sizes.data(),
                       axDistibution.Offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    return vecX;
}

void PrintResult(std::ostream& out, int rank, double durationSec, const std::vector<double>& res, double expectedAnswer) {
    out << "Rank: " << std::to_string(rank) << ", Time: " << durationSec << ", Status: " << CheckAnswer(res, expectedAnswer) << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 768;

    Matrix matrix;
    if (rank == 0) {
        matrix.Resize(N, N, 1.0);
        for (int i = 0; i < N; i++) {
            matrix.At(i, i) = 2.0;
        }
    }
    const double epsilon{ std::pow(10, -5) };
    std::vector<double> b(N, N + 1);

    Timer timer;
    std::vector<double> res;
    double durationSec = 0;
#if 1
    timer.Reset();
    res = SolveLinearEquation_Mpi1(matrix, b, epsilon, rank, size);
    durationSec = timer.GetDurationSec();
    PrintResult(std::cout, rank, durationSec, res, 1);
#endif

#if 0
    timer.Reset();
    res = SolveLinearEquation_Mpi2(matrix, b, epsilon, rank, size);
    durationSec = timer.GetDurationSec();
    PrintResult(std::cout, rank, durationSec, res, 1);
#endif

    sleep(1000);
    MPI_Finalize();
}
