#include <chrono>
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
    Point(int x, int y) {
        coords[1] = x;
        coords[0] = y;
    }

    Point() = default;

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

    void setX(int x) {
        coords[1] = x;
    }

    void setY(int y) {
        coords[0] = y;
    }

    std::string ToStringIJ() const {
        return "(" + std::to_string(i()) + "x" + std::to_string(j()) + ")";
    }

    std::string ToStringXY() const {
        //return "(" + std::to_string(x()) + ", " + std::to_string(y()) + ")";
        return std::to_string(x()) + "x" + std::to_string(y());
    }
};

class Matrix {
private:
    std::vector<double> Data;

    size_t Rows{ 0 };

    size_t Cols{ 0 };

private:
    size_t GetIndexOf(size_t i, size_t j) const {
        if (IsRowMajor) {
            return i * Cols + j;
        }
        return j * Cols + i;
    }

public:
    bool IsRowMajor = true;

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

    double At(size_t i) const {
        if (i >= Data.size()) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return Data[i];
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

    std::string SizeToString() const {
        return std::to_string(GetColsCount()) + "x" + std::to_string(GetRowsCount());
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

    bool operator==(const Matrix& other) const {
        if (GetRowsCount() != other.GetRowsCount() || GetColsCount() != other.GetColsCount()) {
            return false;
        }
        for (size_t i = 0; i < GetRowsCount(); i++) {
            for (size_t j = 0; j < GetColsCount(); j++) {
                if (At(i, j) != other.At(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }

    Matrix operator*(const Matrix& other) const {
        if (GetColsCount() != other.GetRowsCount()) {
            throw std::invalid_argument("Matrix dimensions mismatch");
        }

        Matrix result(GetRowsCount(), other.GetColsCount());

        for (size_t i = 0; i < GetRowsCount(); i++) {
            for (size_t j = 0; j < other.GetColsCount(); j++) {
                double sum{ 0.0f };
                for (size_t k = 0; k < GetColsCount(); k++) {
                    sum += this->At(i, k) * other.At(k, j);
                }
                result.At(i, j) = sum;
            }
        }
        return result;
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
            int n = base + (i < remainder ? 1 : 0);
            Sizes[i] = n;
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

std::string CheckAnswer(const Matrix& matrixA, const Matrix& matrixB) {
    return matrixA == matrixB ? "OK" : "Wrong answer";
}

std::string GetCurrentDateTime() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    return std::ctime(&now_time);
}

MPI_Datatype CreateColumnBlockDatatype(int rowsCount, int colsCount, int blockLength) {
    MPI_Datatype dataType;
    MPI_Type_vector(rowsCount, blockLength, colsCount, MPI_DOUBLE, &dataType);

    MPI_Datatype dataTypeResized;
    MPI_Type_create_resized(dataType, 0, sizeof(double) * blockLength, &dataTypeResized);
    MPI_Type_commit(&dataTypeResized);
    return dataTypeResized;
}

Matrix Multiply(const Matrix& A, const Matrix& B, int rank, int size, Point& gridSize) {
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

    if (rowsCount_A % gridSize.y() != 0 || colsCount_B % gridSize.x() != 0) {
        throw std::invalid_argument("gridSize.x() % colsCount_A != 0 || gridSize.y() % rowsCount_B != 0");
    }

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
    }

    MPI_Comm rowComm;
    MPI_Comm_split(cart_comm, rankCoordinate.i(), rankCoordinate.j(), &rowComm);

    MPI_Bcast(A_local.Raw(), A_local.GetDataSize(), MPI_DOUBLE, 0, rowComm);

    // Matrix B
    const int colsCount_localB = colsCount_B / gridSize.x();

    MPI_Datatype colDataType;
    MPI_Type_vector(rowsCount_B, colsCount_localB,
                    colsCount_B, MPI_DOUBLE, &colDataType);

    MPI_Datatype columnBlockB;
    MPI_Type_create_resized(colDataType, 0, colsCount_localB * sizeof(double), &columnBlockB);
    MPI_Type_commit(&columnBlockB);

    const int row0_color = rankCoordinate.i() == 0 ? 0 : MPI_UNDEFINED;
    MPI_Comm row0_comm;
    MPI_Comm_split(cart_comm, row0_color, rankCoordinate.j(), &row0_comm);

    Matrix B_local(rowsCount_B,
                   colsCount_localB);

    if (row0_color != MPI_UNDEFINED) {
        MPI_Scatter(B.Raw(),
                    1,
                    columnBlockB,
                    B_local.Raw(),
                    B_local.GetDataSize(),
                    MPI_DOUBLE,
                    0,
                    row0_comm);
    }
    MPI_Type_free(&columnBlockB);

    MPI_Comm colComm;
    MPI_Comm_split(cart_comm, rankCoordinate.j(), rankCoordinate.i(), &colComm);
    MPI_Bcast(B_local.Raw(), B_local.GetDataSize(), MPI_DOUBLE, 0, colComm);

    // Multiplication
    Matrix C_local = A_local * B_local;
    
    Matrix C;
    if (rank == 0) {
        C.Resize(rowsCount_A, colsCount_B, 0);
    }

    std::vector<int> recvCounts(size, 1);
    std::vector<int> offsets(size, 0);
    if (rank == 0) {
        int k = 0;
        for (int i = 0; i < gridSize.y(); i++) {
            for (int j = 0; j < gridSize.x(); j++) {
                offsets.at(k) = i * (C.GetColsCount() * C_local.GetRowsCount()) + j * C_local.GetColsCount();
                k++;
            }
        }
    }

    MPI_Datatype dataType;
    MPI_Type_vector(C_local.GetRowsCount(),
                    C_local.GetColsCount(), colsCount_B, MPI_DOUBLE, &dataType);

    MPI_Datatype columnBlockC;
    MPI_Type_create_resized(dataType, 0, sizeof(double), &columnBlockC);
    MPI_Type_commit(&columnBlockC);

    MPI_Gatherv(C_local.Raw(),
                C_local.GetDataSize(),
                MPI_DOUBLE,
                C.Raw(),
                recvCounts.data(),
                offsets.data(),
                columnBlockC,
                0,
                MPI_COMM_WORLD);

    MPI_Type_free(&columnBlockC);

    return rank == 0 ? C : Matrix(0, 0);
}

std::string delim = ";";

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rowsCount_A = 0, colsCount_A = 0;
    int rowsCount_B = 0, colsCount_B = 0;
    int isAutoSize = 0, x = 0, y = 0;
    int isSuccess = 1;

    if (rank == 0) {
        if (argc < 8) {
            std::cout << "Not enough args" << std::endl;
            isSuccess = 0;
        }
        else {
            rowsCount_A = std::stoi(argv[1]);
            colsCount_A = std::stoi(argv[2]);
            rowsCount_B = std::stoi(argv[3]);
            colsCount_B = std::stoi(argv[4]);
            isAutoSize = std::stoi(argv[5]);
            x = std::stoi(argv[6]);
            y = std::stoi(argv[7]);
            if (colsCount_A != rowsCount_B) {
                std::cout << "Bad matrix size" << std::endl;
                isSuccess = 0;
            }
        }
    }
    MPI_Bcast(&isSuccess, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (isSuccess != 1) {
        MPI_Finalize();
        return 0;
    }

    MPI_Bcast(&rowsCount_A, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colsCount_A, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rowsCount_B, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colsCount_B, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&isAutoSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Grid size
    Point gridSize(x, y);
    if (isAutoSize) {
        MPI_Dims_create(size, 2, gridSize.Raw());
    }

    Matrix A, B;
    if (rank == 0) {
        A.Resize(rowsCount_A, colsCount_A, 1);
        B.Resize(rowsCount_B, colsCount_B, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    Timer timer;
    double duration = 0;
    try {
        timer.Reset();
        Matrix result = Multiply(A, B, rank, size, gridSize);
        duration = timer.GetDurationSec();
    }
    catch (const std::exception& e) {
        std::cerr << "std::exception: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown exception (non-standard)" << std::endl;
    }

    if (rank == 0) {
        std::cout << A.SizeToString() << delim << B.SizeToString() << delim << gridSize.ToStringXY() << delim << duration << std::endl;
    }

    MPI_Finalize();
    return 0;
}
