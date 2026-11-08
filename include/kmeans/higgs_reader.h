#ifndef KMEANS_HIGGS_READER_H
#define KMEANS_HIGGS_READER_H

#include <stddef.h>

#include "kmeans/higgs_dataset.h"

typedef struct HiggsReader HiggsReader;

HiggsReader *higgs_reader_open(const char *path);
int higgs_reader_next(HiggsReader *reader, float features[HIGGS_FEATURE_COLUMNS],
                      int *label);
unsigned long long higgs_reader_row(const HiggsReader *reader);
const char *higgs_reader_error(const HiggsReader *reader);
void higgs_reader_close(HiggsReader *reader);

#endif

