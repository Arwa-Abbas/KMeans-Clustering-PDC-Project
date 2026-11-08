#include "kmeans/sequential.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    float points[4 * 28] = {0};
    for (int d = 0; d < 28; ++d) {
        points[0 * 28 + d] = 0;
        points[1 * 28 + d] = 2;
        points[2 * 28 + d] = 10;
        points[3 * 28 + d] = 12;
    }
    KMeansMatrix matrix = {.points = points, .rows = 4, .dimensions = 28};
    KMeansConfig config = {.clusters = 2, .max_iterations = 20,
                           .tolerance = 0.001, .seed = 42, .sample_size = 4};
    KMeansResult a, b;
    if (kmeans_sequential(&matrix, &config, &a) ||
        kmeans_sequential(&matrix, &config, &b)) return 1;
    int good = fabs(a.sse - 112.0) < 1e-8 && a.iterations == b.iterations &&
        fabs(a.sse - b.sse) < 1e-8 &&
        memcmp(a.centroids, b.centroids, 2 * 28 * sizeof(double)) == 0;
    kmeans_free_result(&a);
    kmeans_free_result(&b);
    if (!good) {
        fputs("sequential test failed\n", stderr);
        return 1;
    }
    puts("sequential test passed");
    return 0;
}
