#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <ostream>
#include <thread>
#include <vector>

constexpr int L = 2048;
//constexpr int NUM_OF_TASKS = 2048;
//constexpr int NUM_OF_ITERATIONS = 3;

constexpr int ASK_FOR_TASKS_TAG = 128;
constexpr int SEND_NUM_OF_TASKS_TAG = 256;
constexpr int SEND_TASKS_TAG = 512;
constexpr int IS_DONE = 8;

int size = 0;
int rank = 0;

struct SharedState {
private:
    int NumOfTasks = 0;

public:
    std::vector<int> TaskList;

    std::mutex TaskMutex;

    std::size_t NextLocalTaskIndex = 0; // следующий локальный таск на выполнение

    std::size_t DonateBoundary = 0; // правая граница задач, которые ещё можно отдать

    double global_result = 0.0;

    int completed_tasks = 0;

    SharedState(int numOfTasks) {
        NumOfTasks = numOfTasks;
    }

    void InitializeTasks(int iteration, int numOfTasks) {
        TaskList.clear();
        TaskList.resize(numOfTasks);
        for (int i = 0; i < numOfTasks; i++) {
            TaskList[i] = std::abs(50 - i % 100) * std::abs(rank - (iteration % size)) * L;
        }
    }

};

void DoTask(int weight) {
    for (int j = 0; j < weight; ++j) {
        global_result += std::sin(static_cast<double>(j));
    }
}

void ProcessLocalTasks() {
    while (true) {
        int idx = -1;

        {
            std::lock_guard<std::mutex> lock(task_mutex);
            if (next_local_task < donate_boundary) {
                idx = static_cast<int>(next_local_task++);
            }
        }

        if (idx < 0) {
            break;
        }

        DoTask(taskList[static_cast<std::size_t>(idx)]);
        ++completed_tasks;
    }
}

void RequestAndProcessRemoteTasks() {
    bool received_any = false;

    do {
        received_any = false;

        for (int peer = 0; peer < size; ++peer) {
            if (peer == rank) {
                continue;
            }

            int request_rank = rank;
            MPI_Send(&request_rank, 1, MPI_INT, peer, ASK_FOR_TASKS_TAG, MPI_COMM_WORLD);

            int count = 0;
            MPI_Recv(&count, 1, MPI_INT, peer, SEND_NUM_OF_TASKS_TAG,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (count <= 0) {
                continue;
            }

            std::vector<int> chunk(static_cast<std::size_t>(count));
            MPI_Recv(chunk.data(), count, MPI_INT, peer, SEND_TASKS_TAG,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            received_any = true;

            for (int weight: chunk) {
                DoTask(weight);
                ++completed_tasks;
            }
        }
    } while (received_any);
}

void RunTasks(int iterationsCount, int tasksCount) {
    SharedState workerState(tasksCount);
    for (int iter = 0; iter < iterationsCount; ++iter) {
        double start_iteration = MPI_Wtime();

        {
            std::lock_guard<std::mutex> lock(workerState.TaskMutex);
            workerState.InitializeTasks(iter, tasksCount);
            completed_tasks = 0;
            next_local_task = 0;
            donate_boundary = taskList.size();

            workerState.NextLocalTaskIndex = 0;
        }

        ProcessLocalTasks();
        RequestAndProcessRemoteTasks();

        double end_iteration = MPI_Wtime();
        double iteration_time = end_iteration - start_iteration;

        double max_iteration_time = 0.0;
        double min_iteration_time = 0.0;

        MPI_Allreduce(&iteration_time, &max_iteration_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&iteration_time, &min_iteration_time, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        double disbalance = max_iteration_time - min_iteration_time;
        double disbalance_percent = (max_iteration_time > 0.0)
                                        ? (disbalance / max_iteration_time) * 100.0
                                        : 0.0;

        std::printf("\n");
        std::printf("rank = %d\n", rank);
        std::printf("iteration = %d\n", iter);
        std::printf("time = %f\n", iteration_time);
        std::printf("task_list = %d\n", completed_tasks);
        std::printf("global_result = %.2f\n", global_result);

        if (rank == 0) {
            std::printf("iteration #%d\n", iter);
            std::printf("disbalance = %.2f disbalance_percent = %.2f\n",
                        disbalance, disbalance_percent);
        }

        std::printf("\n");
    }

    int done = IS_DONE;
    MPI_Send(&done, 1, MPI_INT, rank, ASK_FOR_TASKS_TAG, MPI_COMM_WORLD);
}

void ListenTasks() {
    while (true) {
        int requester = 0;
        MPI_Recv(&requester, 1, MPI_INT, MPI_ANY_SOURCE, ASK_FOR_TASKS_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (requester == IS_DONE) {
            break;
        }

        int tasks_to_send = 0;
        std::size_t start_index = 0;

        {
            std::lock_guard<std::mutex> lock(task_mutex);

            std::size_t available = 0;
            if (donate_boundary > next_local_task) {
                available = donate_boundary - next_local_task;
            }

            tasks_to_send = static_cast<int>(available / (size * 2));

            if (tasks_to_send > 0) {
                start_index = donate_boundary - static_cast<std::size_t>(tasks_to_send);
                donate_boundary = start_index;
            }
        }

        MPI_Send(&tasks_to_send, 1, MPI_INT, requester, SEND_NUM_OF_TASKS_TAG, MPI_COMM_WORLD);

        if (tasks_to_send > 0) {
            MPI_Send(taskList.data() + start_index, tasks_to_send, MPI_INT,
                     requester, SEND_TASKS_TAG, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided != MPI_THREAD_MULTIPLE) {
        std::printf("Can't set MPI_THREAD_MULTIPLE.\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start_time = MPI_Wtime();

    std::thread listenerThread(ListenTasks);

    const int iterationsCount = 2;
    const int tasksCount = 2000;

    std::thread workerThread(RunTasks, iterationsCount, tasksCount);

    workerThread.join();
    listenerThread.join();

    double end_time = MPI_Wtime();

    if (rank == 0) {
        std::printf("Time spent: %.2lf seconds.\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
