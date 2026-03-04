#include <chrono>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>
#include <omp.h>

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
    for (double element: vector) {
        sum += element * element;
    }
    return std::sqrt(sum);
}

std::string GetCurrentDateTime() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    return std::ctime(&now_time);
}

void PrintResult(std::ostream& out, const std::string& testName, double durationSec,
                 const std::vector<double>& res, double expectedAnswer) {
    out << GetCurrentDateTime() << "Name: " << testName << ", Time: " << durationSec << ", Status: " <<
            CheckAnswer(res, expectedAnswer) << std::endl;
}

std::vector<double> SolveLinearEquation_Naive(const Matrix& matrixA,
                                              const std::vector<double>& vecB,
                                              double epsilon, double tau) {
    if (!matrixA.IsSquare() || vecB.size() != matrixA.GetRows()) {
        throw std::invalid_argument("!matrix.IsSquare() || rightPart.size() != matrix.GetRows()");
    }

    const double kNormB = Norm2(vecB);
    if (kNormB == 0) {
        throw std::runtime_error("normB == 0");
    }
    const size_t kVecSize = vecB.size();
    std::vector<double> vecAx(kVecSize, 0.0);
    std::vector<double> vecX(kVecSize, 0.0);
    std::vector<double> vecAxMinusB(kVecSize, 0.0);
    const int kMaxIterations = 1000;
    int iterationCount = 0;
    while (true) {
        // A * x
        for (size_t i = 0; i < matrixA.GetRows(); i++) {
            vecAx[i] = 0;
            for (size_t j = 0; j < matrixA.GetCols(); j++) {
                vecAx[i] += matrixA.At(i, j) * vecX.at(j);
            }
        }

        //Ax - b
        double sum = 0.0;
        for (size_t i = 0; i < vecAx.size(); i++) {
            vecAxMinusB[i] = vecAx[i] - vecB[i];
            sum += vecAxMinusB[i] * vecAxMinusB[i];
        }

        if (sqrt(sum) / kNormB < epsilon) {
            break;
        }
       //std::cout << sqrt(sum) << std::endl;

        for (size_t i = 0; i < vecX.size(); i++) {
            vecX[i] = vecX[i] - tau * vecAxMinusB[i];
        }

        if (iterationCount > kMaxIterations) {
            throw std::runtime_error("iterationCount > kMaxIterations");
        }
        iterationCount++;
    }
    return vecX;
}

std::vector<double> SolveLinearEquation_ParallelA(const Matrix& matrixA,
                                              const std::vector<double>& vecB,
                                              double epsilon, double tau) {
    if (!matrixA.IsSquare() || vecB.size() != matrixA.GetRows()) {
        throw std::invalid_argument("!matrix.IsSquare() || rightPart.size() != matrix.GetRows()");
    }

    const double kNormB = Norm2(vecB);
    if (kNormB == 0) {
        throw std::runtime_error("normB == 0");
    }
    const size_t kVecSize = vecB.size();
    std::vector<double> vecAx(kVecSize, 0.0);
    std::vector<double> vecX(kVecSize, 0.0);
    std::vector<double> vecAxMinusB(kVecSize, 0.0);
    const int kMaxIterations = 1000;
    int iterationCount = 0;
    while (true) {
        // A * x
        #pragma omp parallel for
        for (size_t i = 0; i < matrixA.GetRows(); i++) {
            vecAx[i] = 0;
            for (size_t j = 0; j < matrixA.GetCols(); j++) {
                vecAx[i] += matrixA.At(i, j) * vecX.at(j);
            }
        }

        //Ax - b
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < vecAx.size(); i++) {
            vecAxMinusB[i] = vecAx[i] - vecB[i];
            sum += vecAxMinusB[i] * vecAxMinusB[i];
        }

        if (sqrt(sum) / kNormB < epsilon) {
            break;
        }
        //std::cout << sqrt(sum) << std::endl;
        #pragma omp parallel for
        for (size_t i = 0; i < vecX.size(); i++) {
            vecX[i] = vecX[i] - tau * vecAxMinusB[i];
        }

        if (iterationCount > kMaxIterations) {
            throw std::runtime_error("iterationCount > kMaxIterations");
        }
        iterationCount++;
    }
    return vecX;
}

std::vector<double> SolveLinearEquation_ParallelB(const Matrix& matrixA,
                                              const std::vector<double>& vecB,
                                              double epsilon, double tau) {
    if (!matrixA.IsSquare() || vecB.size() != matrixA.GetRows()) {
        throw std::invalid_argument("!matrix.IsSquare() || rightPart.size() != matrix.GetRows()");
    }

    const double kNormB = Norm2(vecB);
    if (kNormB == 0) {
        throw std::runtime_error("normB == 0");
    }
    const size_t kVecSize = vecB.size();
    std::vector<double> vecAx(kVecSize, 0.0);
    std::vector<double> vecX(kVecSize, 0.0);
    std::vector<double> vecAxMinusB(kVecSize, 0.0);
    const int kMaxIterations = 1000;
    int iterationCount = 0;
    double sum = 0;
    bool isFoundAnswer = false;
    #pragma omp parallel
    {
        while (true) {
            #pragma omp barrier
            // A * x
            #pragma omp for
            for (size_t i = 0; i < matrixA.GetRows(); i++) {
                vecAx[i] = 0;
                for (size_t j = 0; j < matrixA.GetCols(); j++) {
                    vecAx[i] += matrixA.At(i, j) * vecX.at(j);
                }
            }
            #pragma omp barrier

            //Ax - b
            #pragma omp master
            {
                sum = 0;
            }
            #pragma omp barrier

            #pragma omp for reduction(+:sum)
            for (size_t i = 0; i < vecAx.size(); i++) {
                vecAxMinusB[i] = vecAx[i] - vecB[i];
                sum += vecAxMinusB[i] * vecAxMinusB[i];
            }
            #pragma omp barrier

            #pragma omp master
            {
                if (sqrt(sum) / kNormB < epsilon) {
                    isFoundAnswer = true;
                    #pragma omp flush(isFoundAnswer)
                }
            }
            #pragma omp barrier
            if (isFoundAnswer) {
                break;
            }

            #pragma omp for
            for (size_t i = 0; i < vecX.size(); i++) {
                vecX[i] = vecX[i] - tau * vecAxMinusB[i];
            }

            if (iterationCount > kMaxIterations) {
                throw std::runtime_error("iterationCount > kMaxIterations");
            }
            #pragma omp barrier
            #pragma omp master
            {
                iterationCount++;
            }
        }

    }
    return vecX;
}

int main(int argc, char** argv) {
    omp_set_num_threads(4);
#if 0
    int N = 0;
    double tau = 0.0;
    tau = atof(argv[2]);
    N = atoi(argv[1]);
#endif
#if 1
    const int N = 13000;
    const double tau = 0.0001;
#endif

    const double epsilon = std::pow(10, -5);

    Matrix matrixA(N, N, 1.0);
    std::vector<double> vecB(N, N + 1);
    for (size_t i = 0; i < N; i++) {
        matrixA.At(i, i) = 2.0;
    }
    Timer timer;
    std::vector<double> result;
    double duration = 0;

#if 1
    timer.Reset();
    result = SolveLinearEquation_Naive(matrixA, vecB, epsilon, tau);
    duration = timer.GetDurationSec();
    PrintResult(std::cout, "Naive", duration, result, 1.0);
#endif

#if 1
    timer.Reset();
    result = SolveLinearEquation_ParallelA(matrixA, vecB, epsilon, tau);
    duration = timer.GetDurationSec();
    PrintResult(std::cout, "ParallelA", duration, result, 1.0);
#endif

#if 1
    timer.Reset();
    result = SolveLinearEquation_ParallelB(matrixA, vecB, epsilon, tau);
    duration = timer.GetDurationSec();
    PrintResult(std::cout, "ParallelB", duration, result, 1.0);
#endif

}
