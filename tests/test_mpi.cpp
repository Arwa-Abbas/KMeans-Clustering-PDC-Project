// Run with several ranks, e.g. `mpirun -np 3 build/test_mpi`.
#include "kmeans/kernel.hpp"
#include "kmeans/mpi_kmeans.hpp"
#include "kmeans/openmp.hpp"
#include "synthetic.hpp"

#include <iostream>
#include <memory>

int main(int argc, char** argv) {
    using namespace kmeans;
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const std::string binary = "build/test_mpi.bin";
    const std::string metadata = "build/test_mpi.metadata.json";
    const Matrix data = test::make_blobs(10007, 6, 7);
    if (rank == 0) test::write_matrix(binary, metadata, data);
    MPI_Barrier(MPI_COMM_WORLD);

    Config config;
    config.clusters = 6;
    config.sample_size = 2000;
    config.max_iterations = 50;
    config.print_progress = false;

    const AssignerFactory serial = [](const float* points, std::uint64_t rows, int dims, int k) {
        return std::unique_ptr<Assigner>(std::make_unique<SerialAssigner>(points, rows, dims, k));
    };
    const AssignerFactory threaded = [](const float* points, std::uint64_t rows, int dims, int k) {
        return std::unique_ptr<Assigner>(std::make_unique<OpenMPAssigner>(points, rows, dims, k, 2));
    };

    bool good = true;
    // All rows, then a 5,003-row prefix as used by the weak-scaling runs.
    for (const std::uint64_t rows : {std::uint64_t{0}, std::uint64_t{5003}}) {
        const Result reference = run_sequential(rows ? test::first_rows(data, rows) : data, config);
        for (const AssignerFactory* factory : {&serial, &threaded}) {
            MpiRun run;
            good = run_mpi(binary, metadata, rows, config, *factory, MPI_COMM_WORLD, run) &&
                   run.rows == (rows ? rows : data.rows) &&
                   test::same_clustering(run.result, reference) && good;
        }
    }

    int local = good ? 1 : 0;
    int all = 0;
    MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) std::cout << (all ? "MPI comparison passed\n" : "MPI comparison failed\n");
    MPI_Finalize();
    return all ? 0 : 1;
}
