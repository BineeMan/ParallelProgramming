#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

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

enum class EMessageTag {
    AskForTasks = 1,
    SendTaskCount = 2,
    SendTasks = 3,
    IsDone = 4
};

struct WorkerState {
private:
    int NumTasks = 0;

public:
    int Size = 0;

    int Rank = 0;

    std::vector<int> TaskList;

    std::mutex TaskMutex;

    std::size_t NextLocalTaskIndex = 0;

    std::size_t DonateBoundary = 0;

    double GlobalResult = 0.0;

    int CompletedTasks = 0;

    WorkerState(int numTasks, int size, int rank)
        : NumTasks(numTasks),
          Size(size),
          Rank(rank),
          TaskList(static_cast<std::size_t>(numTasks), 0) {
    }

    void InitializeTasks(int iteration) {
        std::lock_guard<std::mutex> lock(TaskMutex);

        TaskList.resize(NumTasks);
        for (int i = 0; i < NumTasks; ++i) {
            constexpr int L = 2048;
            TaskList[i] = std::abs(50 - i % 100) * std::abs(Rank - (iteration % Size)) * L;
        }

        NextLocalTaskIndex = 0;
        DonateBoundary = TaskList.size();
        CompletedTasks = 0;
    }

    int GetLocalTaskIndex() {
        std::lock_guard<std::mutex> lock(TaskMutex);

        if (NextLocalTaskIndex >= DonateBoundary) {
            return -1;
        }

        return NextLocalTaskIndex++;
    }

    std::pair<int, std::size_t> GetDonation() {
        std::lock_guard<std::mutex> lock(TaskMutex);

        std::size_t available = 0;
        if (DonateBoundary > NextLocalTaskIndex) {
            available = DonateBoundary - NextLocalTaskIndex;
        }

        int tasksToSend = static_cast<int>(available / (Size * 2));
        std::size_t startIndex = 0;

        if (tasksToSend > 0) {
            startIndex = DonateBoundary - static_cast<std::size_t>(tasksToSend);
            DonateBoundary = startIndex;
        }

        return { tasksToSend, startIndex };
    }

    void ExecuteTask(int weight) {
        for (int j = 0; j < weight; ++j) {
            GlobalResult += std::sin(static_cast<double>(j));
        }
        ++CompletedTasks;
    }
};

void ProcessLocalTasks(WorkerState& state) {
    while (true) {
        int idx = state.GetLocalTaskIndex();
        if (idx < 0) {
            break;
        }

        state.ExecuteTask(state.TaskList.at(idx));
    }
}

void RequestAndProcessRemoteTasks(WorkerState& state) {
    bool receivedAny = false;

    do {
        receivedAny = false;

        for (int peer = 0; peer < state.Size; ++peer) {
            if (peer == state.Rank) {
                continue;
            }

            int requestRank = state.Rank;
            MPI_Send(&requestRank, 1, MPI_INT, peer, static_cast<int>(EMessageTag::AskForTasks), MPI_COMM_WORLD);

            int count = 0;
            MPI_Recv(&count, 1, MPI_INT, peer, static_cast<int>(EMessageTag::SendTaskCount),
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (count <= 0) {
                continue;
            }

            std::vector<int> chunk(static_cast<std::size_t>(count));
            MPI_Recv(chunk.data(), count, MPI_INT, peer, static_cast<int>(EMessageTag::SendTasks),
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            receivedAny = true;

            for (int weight: chunk) {
                state.ExecuteTask(weight);
            }
        }
    } while (receivedAny);
}

void RunTasks(WorkerState& state, int iterationsCount) {
    for (int iter = 0; iter < iterationsCount; iter++) {

        Timer timer;

        state.InitializeTasks(iter);

        ProcessLocalTasks(state);
        RequestAndProcessRemoteTasks(state);

        double iterationTime = timer.GetDurationSec();

        double maxIterationTime = 0.0;
        double minIterationTime = 0.0;

        MPI_Allreduce(&iterationTime, &maxIterationTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&iterationTime, &minIterationTime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        double disbalance = maxIterationTime - minIterationTime;
        double disbalancePercent = (maxIterationTime > 0.0)
                                       ? (disbalance / maxIterationTime) * 100.0
                                       : 0.0;

        std::cout << "Rank = " << state.Rank << std::endl;
        std::cout << "Iteration = " << iter << std::endl;
        std::cout << "Time = " << iterationTime << std::endl;
        std::cout << "Task list = " << state.CompletedTasks << std::endl;
        std::cout << "Global result = " << state.GlobalResult << std::endl;

        if (state.Rank == 0) {
            std::cout << "Iteration № " << iter << std::endl;
            std::cout << "Disbalance = " << disbalance << ", disbalance percent = " << disbalancePercent << std::endl;
        }
        std::cout << "-----------------------------" << std::endl;
    }

    int done = static_cast<int>(EMessageTag::IsDone);
    MPI_Send(&done, 1, MPI_INT, state.Rank, static_cast<int>(EMessageTag::AskForTasks), MPI_COMM_WORLD);
}

void ListenTasks(WorkerState& state) {
    while (true) {
        int requester = 0;
        MPI_Recv(&requester, 1, MPI_INT, MPI_ANY_SOURCE, static_cast<int>(EMessageTag::AskForTasks),
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (requester == static_cast<int>(EMessageTag::IsDone)) {
            break;
        }

        int tasksToSend = 0;
        std::size_t startIndex = 0;

        {
            std::lock_guard<std::mutex> lock(state.TaskMutex);

            std::size_t available = 0;
            if (state.DonateBoundary > state.NextLocalTaskIndex) {
                available = state.DonateBoundary - state.NextLocalTaskIndex;
            }

            tasksToSend = static_cast<int>(available / (state.Size * 2));

            if (tasksToSend > 0) {
                startIndex = state.DonateBoundary - static_cast<std::size_t>(tasksToSend);
                state.DonateBoundary = startIndex;
            }
        }

        MPI_Send(&tasksToSend, 1, MPI_INT, requester, static_cast<int>(EMessageTag::SendTaskCount), MPI_COMM_WORLD);

        if (tasksToSend > 0) {
            MPI_Send(state.TaskList.data() + startIndex, tasksToSend, MPI_INT,
                     requester, static_cast<int>(EMessageTag::SendTasks), MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided != MPI_THREAD_MULTIPLE) {
        std::cout << "Can't set MPI_THREAD_MULTIPLE." << std::endl;
        MPI_Finalize();
        return 1;
    }

    int size = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Timer timer;

    const int iterationsCount = 2;
    const int tasksCount = 2000;

    WorkerState state(tasksCount, size, rank);

    std::thread listenerThread(ListenTasks, std::ref(state));
    std::thread workerThread(RunTasks, std::ref(state), iterationsCount);

    workerThread.join();
    listenerThread.join();

    double durationSec = timer.GetDurationSec();

    if (rank == 0) {
        std::cout << "Time spent: " << durationSec << " seconds." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
