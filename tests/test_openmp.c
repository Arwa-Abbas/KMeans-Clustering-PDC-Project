#include "kmeans/sequential.h"
#include <math.h>
#include <stdio.h>

int main(void) {
    float points[8 * 28] = {0};
    for (int row = 0; row < 8; ++row)
        for (int d = 0; d < 28; ++d)
            points[row * 28 + d] = row < 4 ? (float)row : (float)(row + 10);
    KMeansMatrix matrix = {.points = points, .rows = 8, .dimensions = 28};
    KMeansConfig config = {.clusters = 2, .max_iterations = 20,
                           .tolerance = 0.001, .seed = 42, .sample_size = 8};
    KMeansResult sequential, parallel;
    if (kmeans_sequential(&matrix, &config, &sequential) ||
        kmeans_openmp(&matrix, &config, 3, &parallel)) return 1;
    int good = sequential.iterations == parallel.iterations &&
        sequential.changed == parallel.changed &&
        fabs(sequential.sse - parallel.sse) < 1e-9;
    for (int i = 0; i < 2 * 28; ++i)
        if (fabs(sequential.centroids[i] - parallel.centroids[i]) > 1e-9) good = 0;
    kmeans_free_result(&sequential);
    kmeans_free_result(&parallel);
    if (!good) {
        fputs("OpenMP comparison failed\n", stderr);
        return 1;
    }
    puts("OpenMP comparison passed");
    return 0;
}
