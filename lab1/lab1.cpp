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
    double start{0.0};
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

    double GetMinRowSum() const {
        double minSum{ 0.0f };
        for (size_t i = 0; i < GetRows(); i++) {
            double currentSum{ 0.0f };
            for (size_t j = 0; j < GetCols(); j++) {
                currentSum += std::abs(this->At(i, j));
            }
            minSum = std::min(minSum, currentSum);
        }
        return minSum;
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
        if (std::fabs(answers[i] - expectedAnswer) > 0.0001) {
            return "Wrong answer: expected " + std::to_string(expectedAnswer) + ", got " + std::to_string(answers[i]) +
                   " at index " + std::to_string(i);
        }
    }
    return "OK";
}

double Norm2(const std::vector<double>& vector) {
    double sum{ 0.0f };
    for (double element : vector) {
        sum += element * element;
    }
    return std::sqrt(sum);
}

double SumSquareVector(const std::vector<double>& vector) {
    double sum{ 0.0f };
    for (double element : vector) {
        sum += element * element;
    }
    return sum;
}

std::vector<double> SolveLinearEquation_Mpi1(const Matrix& localMatrixA, int matrixSize, const std::vector<double>& vecB,
                                             double epsilon, double tau, int rank, int size) {
    if (rank < 0 || size <= 0 || matrixSize <= 0) {
        throw std::runtime_error(
            "rank < 0 || size <= 0 || N <= 0" +
            std::to_string(rank) + ", size = " + std::to_string(size));
    }

    if (vecB.size() != matrixSize) {
        throw std::invalid_argument("vecB.size() != N");
    }

    const double normB = Norm2(vecB);
    if (normB == 0) {
        throw std::runtime_error("normB == 0");
    }

    const int kRowsPerCurrentProcess = matrixSize / size + (rank < (matrixSize % size) ? 1 : 0);
    if (kRowsPerCurrentProcess != localMatrixA.GetRows()) {
        throw std::invalid_argument("localMatrixA.GetRows() != kRowsPerCurrentProcess");
    }
    std::vector<double> Ax(vecB.size());
    std::vector<double> localAx(kRowsPerCurrentProcess);
    std::vector<double> AxMinusB(Ax.size());

    std::vector<double> x(vecB.size(), 0.0);

    Distribution axDistribution(matrixSize, size);
    const int startRowForCurrentProcess = axDistribution.GetOffsetAt(rank);

    while (true) {
        // 1) vector Ax = A * x
        localAx = localMatrixA * x;

        MPI_Allgatherv(localAx.data(), localAx.size(),
                       MPI_DOUBLE, Ax.data(), axDistribution.Sizes.data(),
                       axDistribution.Offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // 2) vector Ax - b
        AxMinusB = Ax - vecB;
        double localSum = 0;
        for (int i = 0; i < kRowsPerCurrentProcess; i++) {
            const int globalIdx = startRowForCurrentProcess + i;
            localSum += AxMinusB[globalIdx] * AxMinusB[globalIdx];
        }
        double globalSum = 0;
        MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // 3) Невязка
        if (std::sqrt(globalSum) / normB < epsilon) {
            break;
        }

        x -= tau * AxMinusB;
    }
    return x;
}

std::vector<double> SolveLinearEquation_Mpi2(const Matrix& localMatrixA, int matrixSize, const std::vector<double>& localVecB,
                                             double epsilon, double tau, int rank, int size) {
    if (rank < 0 || size <= 0 || matrixSize <= 0) {
        throw std::runtime_error(
            "rank < 0 || size <= 0 || N <= 0" +
            std::to_string(rank) + ", size = " + std::to_string(size));
    }

    double normB = 0;
    double sumVecB = SumSquareVector(localVecB);
    MPI_Allreduce(&sumVecB, &normB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    normB = std::sqrt(normB);

    const int kRowsPerCurrentProcess = matrixSize / size + (rank < (matrixSize % size) ? 1 : 0);
    if (kRowsPerCurrentProcess != localVecB.size()) {
        throw std::invalid_argument("localVecB.size() != kRowsPerCurrentProcess");
    }

    std::vector<double> localVecAx(kRowsPerCurrentProcess, 0.0);
    std::vector<double> localVecX(kRowsPerCurrentProcess, 0.0);
    const Distribution vectorDistribution(matrixSize, size);
    const int kMaxChunkSize = matrixSize / size + 1;
    std::vector<double> vecXBuffer(kMaxChunkSize, 0.0);

    while (true) {
        vecXBuffer.assign(vecXBuffer.size(), 0.0);
        std::copy(localVecX.begin(), localVecX.end(), vecXBuffer.begin());
        localVecAx.assign(localVecAx.size(), 0.0);
        int owner = rank;
        for (int k = 0; k < size; k++) {
            const int cols = vectorDistribution.GetSizeAt(owner);
            const int offset = vectorDistribution.GetOffsetAt(owner);
            for (int i = 0; i < kRowsPerCurrentProcess; i++) {
                double sum = 0;
                for (int j = 0; j < cols; j++) {
                    sum += localMatrixA.At(i, j + offset) * vecXBuffer[j];
                }
                localVecAx[i] += sum;
            }
            int nextRank = (rank + 1) % size;
            int prevRank = (rank - 1 + size) % size;
            MPI_Sendrecv_replace(vecXBuffer.data(), vecXBuffer.size(),
                MPI_DOUBLE, nextRank, 0, prevRank,
                0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            owner = (owner - 1 + size) % size;
        }

        // 2) Норма ||Ax - b||
        double localNorm = 0;
        for (int i = 0; i < kRowsPerCurrentProcess; i++) {
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
    }
    return localVecX;
}

std::string GetCurrentDateTime() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    return std::ctime(&now_time);
}

void PrintResult(std::ostream& out, const std::string& testName, int size, int rank, double durationSec,
                 const std::vector<double>& res, double expectedAnswer) {
    out << GetCurrentDateTime() <<  "Name: " << testName << ", Size: " << size << ", Rank: " << std::to_string(rank) << ", Time: " << durationSec << ", Status: " <<
            CheckAnswer(res, expectedAnswer) << std::endl;
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int task = -1;
    const bool useTaskSelection = false;
#if 1
    int N = 0;
    double tau = 0.0;
    if (rank == 0) {
        tau = atof(argv[2]);
        N = atoi(argv[1]);
        if (useTaskSelection) {
            task = atoi(argv[3]);
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tau, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (useTaskSelection) {
        MPI_Bcast(&task, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (task == -1) {
            std::cout << "No task found!" << std::endl;
            return 0;
        }
    }
#endif
#if 0
    const int N = 8000;
    const double tau = 0.0001;
#endif
    const double epsilon{ std::pow(10, -5) };
    const int kRowsPerCurrentProcess = N / size + (rank < (N % size) ? 1 : 0);

    const int startRow = (N / size) * rank + std::min(rank, N % size);

    Matrix localMatrixA(kRowsPerCurrentProcess, N, 1.0);
    for (int i = 0; i < kRowsPerCurrentProcess; ++i) {
        int globalRow = startRow + i;
        localMatrixA.At(i, globalRow) = 2.0;
    }

    std::vector<double> b;

    Timer timer;
    std::vector<double> result;
    double localTime = 0;
    double maxDuration = -1;
#if 1
    if (!useTaskSelection || task == 1) {
        b = std::vector<double>(N, N + 1);
        MPI_Barrier(MPI_COMM_WORLD);
        timer.Reset();
        result = SolveLinearEquation_Mpi1(localMatrixA, N, b, epsilon, tau, rank, size);
        MPI_Barrier(MPI_COMM_WORLD);
        localTime = timer.GetDurationSec();
        MPI_Reduce(&localTime, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            PrintResult(std::cout, "task1_new", size, rank, maxDuration, result, 1);
        }
    }
#endif
#if 1
    if (!useTaskSelection || task == 2) {
        b = std::vector<double>(kRowsPerCurrentProcess, N + 1);
        timer.Reset();
        MPI_Barrier(MPI_COMM_WORLD);
        result = SolveLinearEquation_Mpi2(localMatrixA, N, b, epsilon, tau, rank, size);
        MPI_Barrier(MPI_COMM_WORLD);
        localTime = timer.GetDurationSec();
        MPI_Reduce(&localTime, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        std::string msg = CheckAnswer(result, 1);
        if (msg != "OK") {
            std::cout << "SolveLinearEquation_Mpi2 CheckAnswer failed on rank " << rank << " " << msg << std::endl;
            MPI_Finalize();
            return 0;
        }
        if (rank == 0) {
            PrintResult(std::cout, "task2_new", size, rank, maxDuration, result, 1);
        }
    }
#endif

    MPI_Finalize();
}
