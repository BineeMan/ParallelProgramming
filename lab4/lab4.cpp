#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cmath>

class Matrix {
private:
    std::vector<double> Data;
    size_t Rows = 0;
    size_t Cols = 0;

    size_t GetIndexOf(size_t i, size_t j) const {
        return i * Cols + j;
    }

public:
    Matrix() = default;

    Matrix(const Matrix& other) = default;

    Matrix(Matrix&& other) = default;

    Matrix(double* buffer, int count, size_t rows, size_t cols)
        : Data(buffer, buffer + count) {
        if (count <= 0) throw std::invalid_argument("Count is zero");
        Rows = rows;
        Cols = cols;
    }

    Matrix(size_t rows, size_t cols) {
        Rows = rows;
        Cols = cols;
        Data.resize(rows * cols, 0.0);
    }

    Matrix(size_t rows, size_t cols, double value) {
        Rows = rows;
        Cols = cols;
        Data.resize(rows * cols, value);
    }

    Matrix(const std::vector<double>& vector, size_t rows, size_t cols) {
        if (vector.size() != rows * cols)
            throw std::out_of_range("Vector size does not match matrix size");
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
        return Data[GetIndexOf(i, j)];
    }

    const double& At(size_t i, size_t j) const {
        return Data[GetIndexOf(i, j)];
    }

    size_t GetRowsCount() const { return Rows; }
    size_t GetColsCount() const { return Cols; }
    double* Raw() { return Data.data(); }
    const double* Raw() const { return Data.data(); }
    size_t GetDataSize() const { return Data.size(); }
};

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

class Distribution {
public:
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

    Distribution(size_t matrixCols, size_t matrixRows, int rowsSplitCount) {
        Calculate(matrixCols, matrixRows, rowsSplitCount);
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
};

void FillRandom(Matrix& matrix, double aliveProbability = 0.2) {
    for (size_t i = 0; i < matrix.GetRowsCount(); ++i) {
        for (size_t j = 0; j < matrix.GetColsCount(); ++j) {
            matrix.At(i, j) = (static_cast<double>(rand()) / RAND_MAX < aliveProbability) ? 1.0 : 0.0;
        }
    }
}

int GetNeighborsCount(const Matrix& grid, size_t i, size_t j) {
    const size_t colsCount = grid.GetColsCount();
    int neighborsCount = 0;
    for (int di = -1; di <= 1; di++) {
        size_t ni = i + di;
        for (int dj = -1; dj <= 1; dj++) {
            if (di == 0 && dj == 0) {
                continue;
            }
            size_t nj = (j + dj + colsCount) % colsCount;
            if (grid.At(ni, nj) == 1.0) {
                neighborsCount++;
            }
        }
    }
    return neighborsCount;
}

void StartGhostExchange(Matrix& currentFieldGhost_local, size_t rowsCount_local,
                        int rank, int size,
                        MPI_Request* requests, int& req_count) {
    const size_t colsCount = currentFieldGhost_local.GetColsCount();

    req_count = 0;
    if (size == 1) {
        for (size_t j = 0; j < colsCount; ++j) {
            currentFieldGhost_local.At(0, j) = currentFieldGhost_local.At(rowsCount_local, j);
            currentFieldGhost_local.At(rowsCount_local + 1, j) = currentFieldGhost_local.At(1, j);
        }
        return;
    }
    const int upRank = (rank - 1 + size) % size;
    const int downRank = (rank + 1) % size;

    MPI_Isend(&currentFieldGhost_local.At(rowsCount_local, 0), colsCount, MPI_DOUBLE, downRank, 0,
              MPI_COMM_WORLD, &requests[req_count++]);
    MPI_Irecv(&currentFieldGhost_local.At(0, 0), colsCount, MPI_DOUBLE, upRank, 0,
              MPI_COMM_WORLD, &requests[req_count++]);

    MPI_Isend(&currentFieldGhost_local.At(1, 0), colsCount, MPI_DOUBLE, upRank, 1,
              MPI_COMM_WORLD, &requests[req_count++]);
    MPI_Irecv(&currentFieldGhost_local.At(rowsCount_local + 1, 0), colsCount, MPI_DOUBLE, downRank, 1,
              MPI_COMM_WORLD, &requests[req_count++]);
}

double GetNewCellState(bool isAlive, int aliveNeighborsCount) {
    if (isAlive) {
        return aliveNeighborsCount == 2 || aliveNeighborsCount == 3 ? 1.0 : 0.0;
    }
    return aliveNeighborsCount == 3 ? 1.0 : 0.0;
}

void ComputeInnerRows(const Matrix& currentField_local, Matrix& newField_local) {
    size_t rowsCountReal_local = newField_local.GetRowsCount();
    for (size_t i = 2; i < rowsCountReal_local; i++) {
        for (size_t j = 0; j < currentField_local.GetColsCount(); j++) {
            const int aliveNeighborsCount = GetNeighborsCount(currentField_local, i, j);
            const bool isCellAlive = currentField_local.At(i, j) == 1;
            newField_local.At(i - 1, j) = GetNewCellState(isCellAlive, aliveNeighborsCount);
        }
    }
}

void ComputeBoundaryRows(const Matrix& currentField, Matrix& newField,
                         size_t rowsCountReal_local, size_t colsCount) {
    const size_t topRealRow = 1;
    const size_t bottomRealRow = rowsCountReal_local;

    const size_t topNewRow = 0;
    const size_t bottomNewRow = rowsCountReal_local - 1;

    for (size_t j = 0; j < colsCount; j++) {
        int liveNeighborsTop = GetNeighborsCount(currentField, topRealRow, j);
        bool isAliveTopCell = currentField.At(topRealRow, j) == 1.0;
        newField.At(topNewRow, j) = GetNewCellState(isAliveTopCell, liveNeighborsTop);

        int liveNeighborsBottom = GetNeighborsCount(currentField, bottomRealRow, j);
        bool isAliveBottomCell = currentField.At(bottomRealRow, j) == 1.0;
        newField.At(bottomNewRow, j) = GetNewCellState(isAliveBottomCell, liveNeighborsBottom);
    }
}

void CopyFromGhostFieldFieldTo(Matrix& ghostFieldOld, const Matrix& fieldNew) {
    if (ghostFieldOld.GetColsCount() != fieldNew.GetColsCount()
        || ghostFieldOld.GetRowsCount() - 2 != fieldNew.GetRowsCount()) {
        throw std::invalid_argument("ghostFieldOld.GetColsCount() != fieldNew.GetColsCount() "
            "|| ghostFieldOld.GetRowsCount() != fieldNew.GetRowsCount()");
    }
    for (size_t i = 0; i < fieldNew.GetRowsCount(); i++) {
        for (size_t j = 0; j < fieldNew.GetColsCount(); j++) {
            ghostFieldOld.At(i + 1, j) = fieldNew.At(i, j);
        }
    }
}

Matrix SimulateGameOfLife_MPI(const Matrix& initialField, int rank, int size, int iterationCount) {
    if (size < 1) {
        throw std::invalid_argument("size must be greater than 0.");
    }

    const size_t initialFieldRowsCount = initialField.GetRowsCount();
    const size_t initialFieldColsCount = initialField.GetColsCount();

    Distribution rowDistribution(initialFieldColsCount, initialFieldRowsCount, size);
    int rowsCount_local = rowDistribution.GetSizeAt(rank) / initialFieldColsCount;

    Matrix currentFieldGhost_local(rowsCount_local + 2, initialFieldColsCount);
    Matrix newField_local(rowsCount_local, initialFieldColsCount);

    MPI_Scatterv(initialField.Raw(),
                 rowDistribution.Sizes.data(),
                 rowDistribution.Offsets.data(),
                 MPI_DOUBLE,
                 currentFieldGhost_local.Raw() + initialFieldColsCount,
                 rowsCount_local * initialFieldColsCount,
                 MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Request requests[4];
    for (int step = 0; step < iterationCount; ++step) {
        int requestCount = 0;
        StartGhostExchange(currentFieldGhost_local,
                           rowsCount_local,
                           rank,
                           size,
                           requests,
                           requestCount);
        ComputeInnerRows(currentFieldGhost_local, newField_local);
        if (size > 1) {
            MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
        }
        ComputeBoundaryRows(currentFieldGhost_local, newField_local, rowsCount_local, initialFieldColsCount);
        CopyFromGhostFieldFieldTo(currentFieldGhost_local, newField_local);
    }

    Matrix result(initialFieldRowsCount, initialFieldColsCount);
    MPI_Gatherv(currentFieldGhost_local.Raw() + initialFieldColsCount,
                rowsCount_local * initialFieldColsCount,
                MPI_DOUBLE,
                result.Raw(),
                rowDistribution.Sizes.data(),
                rowDistribution.Offsets.data(),
                MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    return result;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        std::cerr << "Invalid arguments count" << std::endl;
        return 1;
    }

    int colsCount = 0;
    int rowsCount = 0;
    int iterationsCount = 0;

    if (rank == 0) {
        rowsCount = std::atoi(argv[1]);
        colsCount = std::atoi(argv[2]);
        iterationsCount = std::atoi(argv[3]);
    }
    double alive_prob = 0.2;

    MPI_Bcast(&colsCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rowsCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterationsCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //std::cout << colsCount << " " << rowsCount << " " << iterationsCount << std::endl;

    if (colsCount <= 0 || rowsCount <= 0 || iterationsCount <= 0) {
        std::cerr << "Error: colsCount, rowsCount, iterationsCount must be positive integers.\n";
        return 1;
    }


    Matrix initial(rowsCount, colsCount);
    if (rank == 0) {
        srand(12345);
        FillRandom(initial, alive_prob);
    }

    Timer timer;
    timer.Reset();
    Matrix final_grid = SimulateGameOfLife_MPI(initial, rank, size, iterationsCount);
    double elapsed = timer.GetDurationSec();

    if (rank == 0) {
        std::cout << size << ";" << iterationsCount << ";" << colsCount << "x" << rowsCount << ";" << elapsed << std::endl;
    }

    MPI_Finalize();
    return 0;
}
