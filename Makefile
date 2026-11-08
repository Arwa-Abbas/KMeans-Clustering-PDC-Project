CC := gcc
CPPFLAGS := -Iinclude
CFLAGS := -std=c11 -O2 -Wall -Wextra -Wpedantic
LDLIBS := -lz

BUILD_DIR := build
OBJECT_DIR := $(BUILD_DIR)/obj
DATASET_TOOL := $(BUILD_DIR)/higgs_dataset
DATASET_SOURCES := src/tools/higgs_dataset_main.c src/dataset/higgs_dataset.c
DATASET_OBJECTS := $(patsubst src/%.c,$(OBJECT_DIR)/%.o,$(DATASET_SOURCES))
PREPROCESS_TOOL := $(BUILD_DIR)/preprocess_higgs
PREPROCESS_SOURCES := src/tools/preprocess_higgs_main.c \
	src/preprocessing/preprocess.c src/preprocessing/online_stats.c \
	src/preprocessing/binary_dataset.c src/dataset/higgs_reader.c
PREPROCESS_OBJECTS := $(patsubst src/%.c,$(OBJECT_DIR)/%.o,$(PREPROCESS_SOURCES))
STATS_TEST := $(BUILD_DIR)/test_online_stats
STATS_TEST_OBJECTS := $(OBJECT_DIR)/tests/test_online_stats.o \
	$(OBJECT_DIR)/preprocessing/online_stats.o
ALL_OBJECTS := $(sort $(DATASET_OBJECTS) $(PREPROCESS_OBJECTS) $(STATS_TEST_OBJECTS))
KMEANS_OBJECTS := $(OBJECT_DIR)/kmeans/matrix.o $(OBJECT_DIR)/kmeans/sequential.o $(OBJECT_DIR)/kmeans/result.o
KMEANS_TOOL := $(BUILD_DIR)/kmeans_sequential
KMEANS_TEST := $(BUILD_DIR)/test_sequential
ALL_OBJECTS += $(KMEANS_OBJECTS) $(OBJECT_DIR)/tools/kmeans_sequential_main.o $(OBJECT_DIR)/tests/test_sequential.o
DEPENDENCIES := $(ALL_OBJECTS:.o=.d)

.PHONY: all test clean

all: $(DATASET_TOOL) $(PREPROCESS_TOOL) $(KMEANS_TOOL)

$(KMEANS_TOOL): $(OBJECT_DIR)/tools/kmeans_sequential_main.o $(KMEANS_OBJECTS) | $(BUILD_DIR)
	$(CC) $^ -o $@ -lm

$(KMEANS_TEST): $(OBJECT_DIR)/tests/test_sequential.o $(OBJECT_DIR)/kmeans/sequential.o | $(BUILD_DIR)
	$(CC) $^ -o $@ -lm

$(DATASET_TOOL): $(DATASET_OBJECTS) | $(BUILD_DIR)
	$(CC) $(DATASET_OBJECTS) -o $@ $(LDLIBS)

$(PREPROCESS_TOOL): $(PREPROCESS_OBJECTS) | $(BUILD_DIR)
	$(CC) $(PREPROCESS_OBJECTS) -o $@ $(LDLIBS) -lm

$(STATS_TEST): $(STATS_TEST_OBJECTS) | $(BUILD_DIR)
	$(CC) $(STATS_TEST_OBJECTS) -o $@ -lm

$(OBJECT_DIR)/%.o: src/%.c
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(OBJECT_DIR)/tests/%.o: tests/%.c
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

test: $(DATASET_TOOL) $(STATS_TEST) $(KMEANS_TEST)
	$(DATASET_TOOL) inspect --sample-rows 1000
	$(STATS_TEST)
	$(KMEANS_TEST)

clean:
	rm -rf $(BUILD_DIR)

-include $(DEPENDENCIES)

