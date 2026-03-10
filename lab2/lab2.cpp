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

    Matrix(size_t rows, size_t cols, double value) {
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

    bool IsSquare() const {
        return GetRows() == GetCols();
    }

};

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

void PrintResult(std::ostream& out, const std::string& testName, double durationSec, int threadNum,
                 const std::vector<double>& res, double expectedAnswer) {
    out << GetCurrentDateTime() << "Name: " << testName << ", Thread num: " << threadNum << ", Time: " << durationSec << ", Status: " <<
            CheckAnswer(res, expectedAnswer) << std::endl  << std::endl;
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
    const int kMaxIterations = 200;
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

std::vector<double> SolveLinearEquation_Schedule(const Matrix& matrixA,
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
    const int kChunckSize = 16;
    int iterationCount = 0;
    #define SCHEDULE_STATIC 1
    //#define SCHEDULE_DYNAMIC 1
    //#define SCHEDULE_GUIDED 1
    //#define SCHEDULE_AUTO 1
    while (true) {
        // A * x

        #ifdef SCHEDULE_STATIC
        #pragma omp parallel for schedule(static, kChunckSize)
        #endif

        #ifdef SCHEDULE_DYNAMIC
        #pragma omp parallel for schedule(dynamic, kChunckSize)
        #endif

        #ifdef SCHEDULE_GUIDED
        #pragma omp parallel for schedule(guided, kChunckSize)
        #endif

        #ifdef SCHEDULE_AUTO
        #pragma omp parallel for schedule(auto)
        #endif

        for (size_t i = 0; i < matrixA.GetRows(); i++) {
            vecAx[i] = 0;
            for (size_t j = 0; j < matrixA.GetCols(); j++) {
                vecAx[i] += matrixA.At(i, j) * vecX.at(j);
            }
        }

        //Ax - b
        double sum = 0.0;
        #ifdef SCHEDULE_STATIC
        #pragma omp parallel for reduction(+:sum) schedule(static, kChunckSize)
        #endif

        #ifdef SCHEDULE_DYNAMIC
        #pragma omp parallel for reduction(+:sum) schedule(dynamic, kChunckSize)
        #endif

        #ifdef SCHEDULE_GUIDED
        #pragma omp parallel for reduction(+:sum) schedule(guided, kChunckSize)
        #endif

        #ifdef SCHEDULE_AUTO
        #pragma omp parallel for reduction(+:sum) schedule(auto)
        #endif

        for (size_t i = 0; i < vecAx.size(); i++) {
            vecAxMinusB[i] = vecAx[i] - vecB[i];
            sum += vecAxMinusB[i] * vecAxMinusB[i];
        }

        if (sqrt(sum) / kNormB < epsilon) {
            break;
        }
#pragma omp master
        {
            std::cout << sqrt(sum) << std::endl;
        }
#pragma omp barrier

        //std::cout << sqrt(sum) << std::endl;
        #ifdef SCHEDULE_STATIC
        #pragma omp parallel for schedule(static, kChunckSize)
        #endif

        #ifdef SCHEDULE_DYNAMIC
        #pragma omp parallel for schedule(dynamic, kChunckSize)
        #endif

        #ifdef SCHEDULE_GUIDED
        #pragma omp parallel for schedule(guided, kChunckSize)
        #endif

        #ifdef SCHEDULE_AUTO
        #pragma omp parallel for schedule(auto)
        #endif

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

int main(int argc, char** argv) {

#if 1
    if (argc != 3) {
        std::cerr << "arc != 3" << std::endl;
        return 1;
    }
    int N = std::stoi(argv[1]);
    double tau = std::stod(argv[2]);
#endif
#if 0
    const int N = 30000;
    const double tau = 0.00001;
#endif
    std::cout << "N = " << N << ", tau = " << tau << std::endl;

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
        PrintResult(std::cout,  "Naive", duration, 1, result, 1.0);
    #endif

#if 1
    const std::vector<int> NumThreads = { 1, 2, 4, 6, 8, 12, 16, 32 };
    for (int numThread : NumThreads) {
        omp_set_num_threads(numThread);
        std::vector<double> durations;
    #if 1
        timer.Reset();
        result = SolveLinearEquation_ParallelA(matrixA, vecB, epsilon, tau);
        duration = timer.GetDurationSec();
        durations.push_back(duration);
        //PrintResult(std::cout, "ParallelA", duration, omp_get_max_threads(), result, 1.0);
    #endif

    #if 1
        timer.Reset();
        result = SolveLinearEquation_ParallelB(matrixA, vecB, epsilon, tau);
        duration = timer.GetDurationSec();
        durations.push_back(duration);
        //PrintResult(std::cout, "ParallelB", duration, omp_get_max_threads(), result, 1.0);
    #endif

        std::cout << numThread << "," << durations.at(0) << "," << durations.at(1) << std::endl;
    }
#endif

#if 0
    omp_set_num_threads(8);
    timer.Reset();
    result = SolveLinearEquation_Schedule(matrixA, vecB, epsilon, tau);
    duration = timer.GetDurationSec();
    PrintResult(std::cout, "Parallel_Schedule", duration, omp_get_max_threads(), result, 1.0);
#endif

}
