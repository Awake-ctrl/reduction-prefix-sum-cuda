#include <cstdio>
#include <cstdlib>
#include <ctime>

void generateTestData(const char* filename, int n) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        printf("Error creating test file!\n");
        return;
    }

    fwrite(&n, sizeof(int), 1, file);

    float *data = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }

    fwrite(data, sizeof(float), n, file);
    fclose(file);
    free(data);
}

int main() {
    srand(time(NULL));

    int sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    system("mkdir -p test_cases");

    for (int i = 0; i < num_sizes; i++) {
        char filename[100];
        sprintf(filename, "test_cases/test_%d.bin", sizes[i]);
        generateTestData(filename, sizes[i]);
        printf("Generated test case: %s\n", filename);
    }

    return 0;
}