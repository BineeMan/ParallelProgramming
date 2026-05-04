#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cmath>

// ========== Класс Matrix (двумерная матрица, row-major) ==========
class Matrix {
private:
    std::vector<double> Data;
    size_t Rows{ 0 };
    size_t Cols{ 0 };

    size_t GetIndexOf(size_t i, size_t j) const {
        return i * Cols + j; // RowMajor
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

void StartGhostExchage(Matrix& currentField_local, size_t local_nx, size_t Ny,
                       int rank, int size,
                       MPI_Request* requests, int& req_count) {
    req_count = 0;
    if (size == 1) {
        for (size_t j = 0; j < Ny; ++j) {
            currentField_local.At(0, j) = currentField_local.At(local_nx, j); // верхняя ghost = последняя живая
            currentField_local.At(local_nx + 1, j) = currentField_local.At(1, j); // нижняя ghost = первая живая
        }
        return;
    }
    int upRank = (rank - 1 + size) % size;
    int downRank = (rank + 1) % size;

    // Отправляем верхнюю живую строку (local_nx) соседу снизу
    MPI_Isend(&currentField_local.At(local_nx, 0), Ny, MPI_DOUBLE, downRank, 0,
              MPI_COMM_WORLD, &requests[req_count++]);
    // Принимаем верхнюю ghost (строка 0) от соседа сверху
    MPI_Irecv(&currentField_local.At(0, 0), Ny, MPI_DOUBLE, upRank, 0,
              MPI_COMM_WORLD, &requests[req_count++]);
    // Отправляем нижнюю живую строку (1) соседу сверху
    MPI_Isend(&currentField_local.At(1, 0), Ny, MPI_DOUBLE, upRank, 1,
              MPI_COMM_WORLD, &requests[req_count++]);
    // Принимаем нижнюю ghost (local_nx+1) от соседа снизу
    MPI_Irecv(&currentField_local.At(local_nx + 1, 0), Ny, MPI_DOUBLE, downRank, 1,
              MPI_COMM_WORLD, &requests[req_count++]);
}

double GetNewCellState(bool isAlive, int live_neighbors) {
    if (isAlive) {
        return live_neighbors == 2 || live_neighbors == 3 ? 1.0 : 0.0;
    }
    return live_neighbors == 3 ? 1.0 : 0.0;
}

// Вычисление внутренних строк (индексы 2..local_nx-1), не зависящих от ghost
void ComputeInnerRows(const Matrix& currentField_local, Matrix& newField_local) {
    size_t rowsCountReal_local = newField_local.GetRowsCount();
    for (size_t i = 2; i < rowsCountReal_local; i++) {
        const size_t new_i = i - 1;
        for (size_t j = 0; j < currentField_local.GetColsCount(); j++) {
            const int liveNeighbors = GetNeighborsCount(currentField_local, i, j);
            const bool isCellAlive = currentField_local.At(i, j) == 1;
            newField_local.At(new_i, j) = GetNewCellState(isCellAlive, liveNeighbors);
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
        double cellStateBottom = currentField.At(bottomRealRow, j) == 1.0;
        newField.At(bottomNewRow, j) = GetNewCellState(cellStateBottom, liveNeighborsBottom);
    }
}

void CopyField(Matrix& local_old, const Matrix& local_new,
               size_t local_nx, size_t Ny) {
    for (size_t i = 0; i < local_nx; ++i) {
        for (size_t j = 0; j < Ny; ++j) {
            local_old.At(i + 1, j) = local_new.At(i, j);
        }
    }
}

Matrix GameOfLifeMPI(const Matrix& initialGrid, int rank, int size, int iterationCount) {
    const size_t gridRowsCount = initialGrid.GetRowsCount();
    const size_t colsCount = initialGrid.GetColsCount();

    Distribution distribution(colsCount, gridRowsCount, size);
    int rowsCount_local = distribution.GetSizeAt(rank) / colsCount;

    Matrix currentField_local(rowsCount_local + 2, colsCount);
    Matrix newField_local(rowsCount_local, colsCount);

    MPI_Scatterv(initialGrid.Raw(),
                 distribution.Sizes.data(),
                 distribution.Offsets.data(),
                 MPI_DOUBLE,
                 currentField_local.Raw() + colsCount,
                 rowsCount_local * colsCount,
                 MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Request requests[4];
    for (int step = 0; step < iterationCount; ++step) {
        int requestCount = 0;
        StartGhostExchage(currentField_local,
                          rowsCount_local,
                          colsCount,
                          rank,
                          size,
                          requests,
                          requestCount);
        ComputeInnerRows(currentField_local, newField_local);
        if (size > 1) {
            MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);
        }
        ComputeBoundaryRows(currentField_local, newField_local, rowsCount_local, colsCount);
        CopyField(currentField_local, newField_local, rowsCount_local, colsCount);
    }

    // Сбор итоговой матрицы на процессе 0
    Matrix result(gridRowsCount, colsCount);
    MPI_Gatherv(currentField_local.Raw() + colsCount,
                rowsCount_local * colsCount,
                MPI_DOUBLE,
                result.Raw(),
                distribution.Sizes.data(),
                distribution.Offsets.data(),
                MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    return result;
}

// ========== Пример использования ==========
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    //std::cout << "rank: " << rank << " size: " << size << std::endl;

    // Параметры сетки и симуляции
    const size_t Nx = 1000;
    const size_t Ny = 1000;
    const int max_steps = 50;

    // Готовим начальную матрицу на процессе 0
    Matrix initial(Nx, Ny);
    if (rank == 0) {
        srand(12345);
        FillRandom(initial, 0.2);
    }

    // Замер времени и запуск
    Timer timer;
    Matrix final_grid = GameOfLifeMPI(initial, rank, size, max_steps);
    double elapsed = timer.GetDurationSec();

    if (rank == 0) {
        std::cout << "Game of Life MPI:\n"
                << "  Grid: " << Nx << " x " << Ny << "\n"
                << "  Generations: " << max_steps << "\n"
                << "  Processes: " << size << "\n"
                << "  Elapsed time: " << elapsed << " sec." << std::endl;
        // При необходимости здесь можно работать с final_grid (анализ, сохранение)
    }

    MPI_Finalize();
    return 0;
}
