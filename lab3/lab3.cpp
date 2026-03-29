#include <chrono>
//#include <format>
#include <iostream>
#include <cmath>
#include <fstream>
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

struct Point {
private:
    int coords[2] = { 0, 0 };

public:
    int i() const {
        return coords[0];
    }

    int j() const {
        return coords[1];
    }

    int x() const {
        return coords[1];
    }

    int y() const {
        return coords[0];
    }

    int* Raw() {
        return coords;
    }

    std::string ToStringIJ() const {
        return "(" + std::to_string(i()) + ", " + std::to_string(j()) + ")";
    }

    std::string ToStringXY() const {
        return "(" + std::to_string(x()) + ", " + std::to_string(y()) + ")";
    }
};

class Matrix {
private:
    std::vector<double> Data;

    size_t Rows{ 0 };

    size_t Cols{ 0 };

    bool IsRowMajor = true;

private:
    size_t GetIndexOf(size_t i, size_t j) const {
        if (IsRowMajor) {
            return i * Cols + j;
        }
        return j * Cols + i;
    }
public:
    enum class EMatrixType {
        ROW_MAJOR,
        COL_MAJOR
    };
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

    Matrix(size_t rows, size_t cols, bool isRowMajor) {
        Rows = rows;
        Cols = cols;
        IsRowMajor = isRowMajor;
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
        return Data[GetIndexOf(i, j)];
    }

    const double& At(size_t i, size_t j) const {
        if (i >= Rows || j >= Cols) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return Data[GetIndexOf(i, j)];
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

    size_t GetDataSize() const {
        return Data.size();
    }

    void PrintToFile(const std::string& filename) const {
        std::ofstream out(filename);

        if (!out.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                out << At(i, j);
                if (j + 1 < Cols) {
                    out << " ";
                }
            }
            out << "\n";
        }

        out.close();
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

    void PrintSize() const {
        for (auto v: Sizes) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }

    Distribution() = default;

    Distribution(int size, int splitSize) {
        Calculate(size, splitSize);
    }

    Distribution(size_t matrixCols, size_t matrixRows, int rowsSplitCount) {
        Calculate(matrixCols, matrixRows, rowsSplitCount);
    }

    void Calculate(int size, int splitSize) {
        Sizes.resize(splitSize);
        Offsets.resize(splitSize);

        const int base = size / splitSize;
        const int remainder = size % splitSize;

        int offset{ 0 };
        for (int i = 0; i < splitSize; i++) {
            //std::cout << i << std::endl;
            int n = base + (i < remainder ? 1 : 0);
            Sizes[i] = n;
            //std::cout << n << ", " << Sizes.at(i) << std::endl;
            Offsets[i] = offset;
            offset += n;
        }
    }

    void Calculate(size_t dimensionA, size_t dimensionB, int count) {
        Sizes.resize(count);
        Offsets.resize(count);
        const int base = dimensionB / count;
        const int reminder = dimensionB % count;

        int offset{ 0 };

        for (int i = 0; i < count; i++) {
            const int rowsPerProcess = base + (i < reminder ? 1 : 0);
            const int numsPerProcess = dimensionA * rowsPerProcess;
            Sizes[i] = numsPerProcess;
            Offsets[i] = offset;
            offset += numsPerProcess;
        }
    }

    void Bcast(int root, MPI_Comm comm) {
        int size;
        if (!Sizes.empty()) {
            size = Sizes.size();
        }

        MPI_Bcast(&size, 1, MPI_INT, root, comm);

        if (Sizes.empty()) {
            Sizes.resize(size);
            Offsets.resize(size);
        }

        MPI_Bcast(Sizes.data(), Sizes.size(), MPI_INT, root, comm);
        MPI_Bcast(Offsets.data(), Offsets.size(), MPI_INT, root, comm);
    }
};

template<typename T>
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
                return "Value does not match, excpected " + std::to_string(expectedValue) + ", got " + std::to_string(
                           matrix.At(i, j));
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

    int rowsCount_A = rank == 0 ? A.GetRowsCount() : 0;
    MPI_Bcast(&rowsCount_A, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int colsCount_A = rank == 0 ? A.GetColsCount() : 0;
    MPI_Bcast(&colsCount_A, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rowsCount_B = rank == 0 ? B.GetRowsCount() : 0;
    MPI_Bcast(&rowsCount_B, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int colsCount_B = rank == 0 ? B.GetColsCount() : 0;
    MPI_Bcast(&colsCount_B, 1, MPI_INT, 0, MPI_COMM_WORLD);

    Point gridSize;
    MPI_Dims_create(size, 2, gridSize.Raw());

    int periods[2] = { 0, 0 };
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD,
                    2,
                    gridSize.Raw(),
                    periods,
                    0,
                    &cart_comm);

    Point rankCoordinate;
    MPI_Cart_coords(cart_comm, rank, 2, rankCoordinate.Raw());

    Distribution distribution_A;
    if (rank == 0) {
        distribution_A.Calculate(
            A.GetColsCount(),
            A.GetRowsCount(),
            gridSize.y());
    }
    distribution_A.Bcast(0, MPI_COMM_WORLD);

    const int kRowsPerCurrentProcess =
            distribution_A.GetSizeAt(rankCoordinate.i()) / colsCount_A;

    Matrix A_local(kRowsPerCurrentProcess, colsCount_A, 0.0f);

    const int col0_color = rankCoordinate.j() == 0 ? 0 : MPI_UNDEFINED;
    MPI_Comm col0_comm;
    MPI_Comm_split(cart_comm, col0_color, rankCoordinate.i(), &col0_comm);

    // int w = 0;
    // while (!w) {
    //     sleep(1);
    // }

    if (col0_color != MPI_UNDEFINED) {
        int col0_rank;
        MPI_Comm_rank(col0_comm, &col0_rank);
        MPI_Scatterv(A.Raw(),
                     distribution_A.Sizes.data(),
                     distribution_A.Offsets.data(),
                     MPI_DOUBLE,
                     A_local.Raw(),
                     distribution_A.GetSizeAt(col0_rank),
                     MPI_DOUBLE, 0, col0_comm);
        //A_local.PrintToFile("out/" + rankCoordinate.ToStringIJ() + ".txt");
    }

    MPI_Comm rowComm;
    MPI_Comm_split(cart_comm, rankCoordinate.i(), rankCoordinate.j(), &rowComm);

    MPI_Bcast(A_local.Raw(), A_local.GetDataSize(), MPI_DOUBLE, 0, rowComm);

    // Matrix B
    MPI_Datatype col_block_send;
    MPI_Type_vector(rowsCount_B, 1, colsCount_B, MPI_DOUBLE, &col_block_send);
    MPI_Type_commit(&col_block_send);

    MPI_Datatype col_block_send_resized;
    MPI_Type_create_resized(col_block_send, 0, sizeof(double), &col_block_send_resized);
    MPI_Type_commit(&col_block_send_resized);

    const int row0_color = rankCoordinate.i() == 0 ? 0 : MPI_UNDEFINED;
    MPI_Comm row0_comm;
    MPI_Comm_split(cart_comm, row0_color, rankCoordinate.j(), &row0_comm);

    Distribution distribution_B;
    if (rank == 0) {
        distribution_B.Calculate(colsCount_B, gridSize.x());
        //std::cout << "x = " << gridSize.x() << std::endl;
    }
    distribution_B.Bcast(0, MPI_COMM_WORLD);

    Matrix B_local(rowsCount_B,
        distribution_B.GetSizeAt(rankCoordinate.j()),
        true);
    //Matrix B_local(rowsCount_B, 10, -1);
    //B_local.PrintToFile("out/" + rankCoordinate.ToString() + ".txt");

    if (row0_color != MPI_UNDEFINED) {
        //std::cout << rankCoordinate.j() << " = " << distribution_B.GetSizeAt(rankCoordinate.j()) << std::endl;
        MPI_Scatterv(B.Raw(),
                     distribution_B.Sizes.data(),
                     distribution_B.Offsets.data(),
                     col_block_send_resized,
                     B_local.Raw(),
                     B_local.GetDataSize(),
                     MPI_DOUBLE,
                     0,
                     row0_comm);

        B_local.PrintToFile("out/" + rankCoordinate.ToStringIJ() + ".txt");
    }

    MPI_Comm col_comm;
    MPI_Comm_split(cart_comm, rankCoordinate.j(), rankCoordinate.i(), &col_comm);

    // MPI_Bcast(B_local.Raw(),
    //           B_local.GetDataSize(),
    //           MPI_DOUBLE,
    //           0,
    //           col_comm);
    //B_local.PrintToFile("out/" + rankCoordinate.ToString() + ".txt");

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
        const int kMatrixSize = 10;
        A.Resize(kMatrixSize, kMatrixSize, 1);
        B.Resize(kMatrixSize, kMatrixSize, 1);

        int acc;
        acc = 0;
        for (int i = 0; i < A.GetRowsCount(); i++) {
            for (int j = 0; j < A.GetColsCount(); j++) {
                A.At(i, j) = acc;
                acc++;
            }
        }

        acc = 0;
        for (int i = 0; i < B.GetRowsCount(); i++) {
            for (int j = 0; j < B.GetColsCount(); j++) {
                B.At(j, i) = acc;
                acc++;
            }
        }
        B.Print();
        //std::cout << A.GetRowsCount() << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    Matrix result = Multiply(A, B, rank, size);

    //result.Print();

    MPI_Finalize();
}
