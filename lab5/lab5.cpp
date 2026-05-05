#include <mpi.h>

#include <atomic>
#include <condition_variable>
#include <cmath>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

struct Task {
    int repeatNum = 0;
};

static constexpr int TAG_REQUEST = 1;
static constexpr int TAG_TASKS   = 2;
static constexpr int TAG_DONE    = 3;
static constexpr int TAG_STOP    = 4;

struct SharedState {
    std::mutex mtx;
    std::condition_variable cv;

    std::deque<Task> localTasks;

    bool shutdown = false;
    bool iterationDone = false;
    bool localNeedWork = false;

    int iter = 0;
    long long localDoneCount = 0;
};

static std::vector<Task> generateTasks(int rank, int size, int iter) {
    std::mt19937 rng(1234567u + rank * 10007u + iter * 7919u);
    std::uniform_int_distribution<int> noise(0, 3000);

    // Волна нагрузки, зависящая от rank и iter.
    int wave = std::abs(rank - (iter % size));
    int taskCount = 12 + wave * 4; // чем ближе к пику волны, тем больше задач

    std::vector<Task> tasks;
    tasks.reserve(taskCount);
    for (int i = 0; i < taskCount; ++i) {
        Task t;
        t.repeatNum = 15000 + wave * 12000 + noise(rng);
        tasks.push_back(t);
    }
    return tasks;
}

static void runTask(const Task& t, double& globalRes) {
    double local = 0.0;
    for (int i = 0; i < t.repeatNum; ++i) {
        local += std::sqrt(static_cast<double>(i));
    }
    globalRes += local;
}

static void sendTaskBatch(int dest, int iter, const std::vector<Task>& batch) {
    int header[2] = {iter, static_cast<int>(batch.size())};
    MPI_Send(header, 2, MPI_INT, dest, TAG_TASKS, MPI_COMM_WORLD);
    if (!batch.empty()) {
        std::vector<int> payload(batch.size());
        for (size_t i = 0; i < batch.size(); ++i) payload[i] = batch[i].repeatNum;
        MPI_Send(payload.data(), static_cast<int>(payload.size()), MPI_INT, dest, TAG_TASKS, MPI_COMM_WORLD);
    }
}

static std::vector<Task> recvTaskBatch(int src) {
    int header[2] = {0, 0};
    MPI_Recv(header, 2, MPI_INT, src, TAG_TASKS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int count = header[1];
    std::vector<Task> batch;
    batch.resize(count);
    if (count > 0) {
        std::vector<int> payload(count);
        MPI_Recv(payload.data(), count, MPI_INT, src, TAG_TASKS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < count; ++i) batch[i].repeatNum = payload[i];
    }
    return batch;
}

static void workerThreadFunc(SharedState* st, double* localRes) {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(st->mtx);
            st->cv.wait(lock, [&] {
                return st->shutdown || !st->localTasks.empty();
            });
            if (st->shutdown) break;
            task = st->localTasks.front();
            st->localTasks.pop_front();
        }

        runTask(task, *localRes);

        {
            std::lock_guard<std::mutex> lock(st->mtx);
            ++st->localDoneCount;
            if (st->localTasks.empty()) {
                st->localNeedWork = true;
            }
        }
    }
}

static void commThreadFunc(SharedState* st, int rank, int size) {
    std::vector<Task> pendingPool; // резерв задач, которые rank 0 может раздавать
    pendingPool.reserve(1024);

    long long expectedDone = 0;
    bool haveExpectedDone = false;

    while (true) {
        int flag = 0;
        MPI_Status status{};
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

        if (flag) {
            if (status.MPI_TAG == TAG_REQUEST) {
                int dummy = 0;
                MPI_Recv(&dummy, 1, MPI_INT, status.MPI_SOURCE, TAG_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (rank == 0) {
                    std::vector<Task> batch;
                    {
                        std::lock_guard<std::mutex> lock(st->mtx);
                        if (!pendingPool.empty()) {
                            size_t give = std::max<size_t>(1, pendingPool.size() / 2);
                            give = std::min(give, pendingPool.size());
                            batch.insert(batch.end(), pendingPool.begin(), pendingPool.begin() + static_cast<long>(give));
                            pendingPool.erase(pendingPool.begin(), pendingPool.begin() + static_cast<long>(give));
                        }
                    }
                    sendTaskBatch(status.MPI_SOURCE, st->iter, batch);
                } else {
                    // Не-координатору проще не раздавать задачи; он только отвечает "нет".
                    std::vector<Task> empty;
                    sendTaskBatch(status.MPI_SOURCE, st->iter, empty);
                }
            } else if (status.MPI_TAG == TAG_TASKS) {
                std::vector<Task> batch = recvTaskBatch(status.MPI_SOURCE);
                {
                    std::lock_guard<std::mutex> lock(st->mtx);
                    for (auto& t : batch) st->localTasks.push_back(t);
                    st->localNeedWork = false;
                }
                st->cv.notify_one();
            } else if (status.MPI_TAG == TAG_DONE) {
                long long one = 0;
                MPI_Recv(&one, 1, MPI_LONG_LONG, status.MPI_SOURCE, TAG_DONE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (rank == 0) {
                    expectedDone += one;
                }
            } else if (status.MPI_TAG == TAG_STOP) {
                int dummy = 0;
                MPI_Recv(&dummy, 1, MPI_INT, status.MPI_SOURCE, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::lock_guard<std::mutex> lock(st->mtx);
                st->shutdown = true;
                st->cv.notify_all();
                break;
            }
        } else {
            bool needWork = false;
            {
                std::lock_guard<std::mutex> lock(st->mtx);
                needWork = st->localNeedWork && !st->shutdown;
            }

            if (needWork) {
                // Запросить работу у rank 0.
                if (rank != 0) {
                    int dummy = 0;
                    MPI_Send(&dummy, 1, MPI_INT, 0, TAG_REQUEST, MPI_COMM_WORLD);

                    // Получить ответ.
                    std::vector<Task> batch = recvTaskBatch(0);
                    if (!batch.empty()) {
                        {
                            std::lock_guard<std::mutex> lock(st->mtx);
                            for (auto& t : batch) st->localTasks.push_back(t);
                            st->localNeedWork = false;
                        }
                        st->cv.notify_one();
                    } else {
                        // Пока работ нет, подождать немного и попробовать снова.
                        std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    }
                } else {
                    // rank 0 сам может “подкармливать” свой worker из pendingPool.
                    std::lock_guard<std::mutex> lock(st->mtx);
                    if (!pendingPool.empty() && st->localTasks.empty()) {
                        size_t give = std::max<size_t>(1, pendingPool.size() / 2);
                        give = std::min(give, pendingPool.size());
                        for (size_t i = 0; i < give; ++i) {
                            st->localTasks.push_back(pendingPool[i]);
                        }
                        pendingPool.erase(pendingPool.begin(), pendingPool.begin() + static_cast<long>(give));
                        st->localNeedWork = false;
                        st->cv.notify_one();
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    }
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        // Если rank 0, после накопления завершенных задач можно окончить итерацию.
        if (rank == 0) {
            std::lock_guard<std::mutex> lock(st->mtx);
            if (!haveExpectedDone) {
                // expectedDone заполняется сообщениями TAG_DONE; здесь просто ждём.
            }
        }

        // Условие остановки для comm-thread выставляет main thread через shutdown.
        {
            std::lock_guard<std::mutex> lock(st->mtx);
            if (st->shutdown) break;
        }
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        if (provided == MPI_THREAD_SINGLE) {
            std::cerr << "MPI does not support MPI_THREAD_MULTIPLE on this system.\n";
        } else {
            std::cerr << "MPI thread support is too weak for this example.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int iterations = 5;

    double globalRes = 0.0;
    long long globalDone = 0;

    for (int iter = 0; iter < iterations; ++iter) {
        SharedState st;
        st.iter = iter;

        // Генерация локальных задач на каждой итерации.
        std::vector<Task> generated = generateTasks(rank, size, iter);

        // Часть задач отдаем в общий пул rank 0, часть оставляем у себя.
        std::vector<Task> localPart;
        std::vector<Task> remotePart;
        localPart.reserve((generated.size() + 1) / 2);
        remotePart.reserve(generated.size() / 2);

        for (size_t i = 0; i < generated.size(); ++i) {
            if (i % 2 == 0) localPart.push_back(generated[i]);
            else remotePart.push_back(generated[i]);
        }

        {
            std::lock_guard<std::mutex> lock(st.mtx);
            for (auto& t : localPart) st.localTasks.push_back(t);
            st.localNeedWork = st.localTasks.empty();
        }

        // Если есть чем поделиться, отправляем это на rank 0.
        if (!remotePart.empty()) {
            if (rank == 0) {
                // rank 0 сам забирает свои "лишние" задачи в локальный пул через comm thread.
                // Для простоты сразу добавим их в локальную очередь rank 0.
                std::lock_guard<std::mutex> lock(st.mtx);
                for (auto& t : remotePart) st.localTasks.push_back(t);
            } else {
                // Отправляем в общий пул rank 0.
                int header[2] = {iter, static_cast<int>(remotePart.size())};
                MPI_Send(header, 2, MPI_INT, 0, TAG_TASKS, MPI_COMM_WORLD);
                std::vector<int> payload(remotePart.size());
                for (size_t i = 0; i < remotePart.size(); ++i) payload[i] = remotePart[i].repeatNum;
                MPI_Send(payload.data(), static_cast<int>(payload.size()), MPI_INT, 0, TAG_TASKS, MPI_COMM_WORLD);
            }
        }

        std::thread worker(workerThreadFunc, &st, &globalRes);
        std::thread comm(commThreadFunc, &st, rank, size);

        // Ожидаем, пока локальная очередь опустеет.
        // Упрощенная логика: когда worker добрался до пустой очереди, comm thread будет пытаться добрать работу.
        // В учебной постановке этого достаточно, чтобы показать взаимодействие потоков и MPI.
        while (true) {
            {
                std::lock_guard<std::mutex> lock(st.mtx);
                if (st.localTasks.empty() && st.localNeedWork) {
                    // Если новых задач не пришло, считаем, что локальная работа закончена.
                    // В реальном варианте тут обычно делают более строгую детекцию завершения.
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // Сигнал завершения для потоков текущей итерации.
        {
            std::lock_guard<std::mutex> lock(st.mtx);
            st.shutdown = true;
            st.iterationDone = true;
        }
        st.cv.notify_all();

        // Сообщим rank 0 число выполненных задач данной итерации.
        long long localDone = 0;
        {
            std::lock_guard<std::mutex> lock(st.mtx);
            localDone = st.localDoneCount;
        }
        MPI_Send(&localDone, 1, MPI_LONG_LONG, 0, TAG_DONE, MPI_COMM_WORLD);

        worker.join();
        comm.join();

        // Глобальная синхронизация между итерациями.
        MPI_Allreduce(&localDone, &globalDone, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "Iteration " << iter << ": total completed tasks = " << globalDone << '\n';
        }

        // На следующей итерации будет новый список задач.
    }

    if (rank == 0) {
        std::cout << "globalRes = " << globalRes << '\n';
    }

    MPI_Finalize();
    return 0;
}
