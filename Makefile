CXX := g++
MPICXX := mpicxx
MPIRUN := mpirun
CPPFLAGS := -Iinclude
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic
MPI_FLAGS := -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
OPENMP := -fopenmp

BUILD := build
OBJ := $(BUILD)/obj

# Shared by all four K-Means programs. Built without OpenMP or MPI, so kernel.o (the
# assignment loop) is one machine-code copy used by every variant.
CORE := $(addprefix $(OBJ)/kmeans/,matrix.o initialize.o kernel.o lloyd.o result.o cli.o)
OPENMP_OBJ := $(OBJ)/kmeans/openmp.o
MPI_OBJ := $(OBJ)/kmeans/mpi.o
DATASET_OBJECTS := $(OBJ)/tools/higgs_dataset_main.o $(OBJ)/dataset/higgs_dataset.o \
	$(OBJ)/dataset/higgs_reader.o
PREPROCESS_OBJECTS := $(OBJ)/tools/preprocess_higgs_main.o $(OBJ)/preprocessing/preprocess.o \
	$(OBJ)/preprocessing/online_stats.o $(OBJ)/preprocessing/binary_dataset.o \
	$(OBJ)/dataset/higgs_reader.o

PROGRAMS := $(addprefix $(BUILD)/,higgs_dataset preprocess_higgs kmeans_sequential \
	kmeans_openmp kmeans_mpi kmeans_hybrid)
TESTS := $(addprefix $(BUILD)/,test_online_stats test_sequential test_openmp test_mpi)

.PHONY: all test clean

all: $(PROGRAMS)

$(BUILD)/higgs_dataset: $(DATASET_OBJECTS)
	$(CXX) $^ -o $@ -lz
$(BUILD)/preprocess_higgs: $(PREPROCESS_OBJECTS)
	$(CXX) $^ -o $@ -lz
$(BUILD)/kmeans_sequential: $(OBJ)/tools/kmeans_sequential_main.o $(CORE)
	$(CXX) $^ -o $@
$(BUILD)/kmeans_openmp: $(OBJ)/tools/kmeans_openmp_main.o $(OPENMP_OBJ) $(CORE)
	$(CXX) $(OPENMP) $^ -o $@
$(BUILD)/kmeans_mpi: $(OBJ)/tools/kmeans_mpi_main.o $(MPI_OBJ) $(CORE)
	$(MPICXX) $^ -o $@
$(BUILD)/kmeans_hybrid: $(OBJ)/tools/kmeans_hybrid_main.o $(MPI_OBJ) $(OPENMP_OBJ) $(CORE)
	$(MPICXX) $(OPENMP) $^ -o $@

$(BUILD)/test_online_stats: $(OBJ)/tests/test_online_stats.o $(OBJ)/preprocessing/online_stats.o
	$(CXX) $^ -o $@
$(BUILD)/test_sequential: $(OBJ)/tests/test_sequential.o $(CORE)
	$(CXX) $^ -o $@
$(BUILD)/test_openmp: $(OBJ)/tests/test_openmp.o $(OPENMP_OBJ) $(CORE)
	$(CXX) $(OPENMP) $^ -o $@
$(BUILD)/test_mpi: $(OBJ)/tests/test_mpi.o $(MPI_OBJ) $(OPENMP_OBJ) $(CORE)
	$(MPICXX) $(OPENMP) $^ -o $@

COMPILE = @mkdir -p $(dir $@) && echo "  CXX $@" && $(1) $(CPPFLAGS) $(CXXFLAGS) $(2) -MMD -MP -c $< -o $@

$(OBJ)/%.o: src/%.cpp
	$(call COMPILE,$(CXX),)
$(OBJ)/tests/%.o: tests/%.cpp
	$(call COMPILE,$(CXX),)
$(OPENMP_OBJ): src/kmeans/openmp.cpp
	$(call COMPILE,$(CXX),$(OPENMP))
$(MPI_OBJ): src/kmeans/mpi.cpp
	$(call COMPILE,$(MPICXX),$(MPI_FLAGS))
$(OBJ)/tools/kmeans_sequential_main.o: src/tools/kmeans_main.cpp
	$(call COMPILE,$(CXX),)
$(OBJ)/tools/kmeans_openmp_main.o: src/tools/kmeans_main.cpp
	$(call COMPILE,$(CXX),-DKMEANS_OPENMP_CLI)
$(OBJ)/tools/kmeans_mpi_main.o: src/tools/kmeans_mpi_main.cpp
	$(call COMPILE,$(MPICXX),$(MPI_FLAGS))
$(OBJ)/tools/kmeans_hybrid_main.o: src/tools/kmeans_mpi_main.cpp
	$(call COMPILE,$(MPICXX),$(MPI_FLAGS) -DKMEANS_HYBRID_CLI)
$(OBJ)/tests/test_mpi.o: tests/test_mpi.cpp
	$(call COMPILE,$(MPICXX),$(MPI_FLAGS))

test: $(TESTS)
	$(BUILD)/test_online_stats
	$(BUILD)/test_sequential
	$(BUILD)/test_openmp
	$(MPIRUN) --oversubscribe -np 3 $(BUILD)/test_mpi

clean:
	rm -rf $(BUILD)

-include $(shell find $(OBJ) -name '*.d' 2>/dev/null)
