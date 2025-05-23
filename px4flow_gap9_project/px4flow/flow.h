#ifndef FLOW_H
#define FLOW_H

#include "pmsis.h"

#define IMAGE_WIDTH 160
#define IMAGE_HEIGHT 120
#define SEARCH_SIZE 4
#define TILE_SIZE 8
#define NUM_BLOCKS 8
#define NUM_CORES 8
#define HIST_SIZE (2*(2*SEARCH_SIZE +1)+1)

typedef struct unsigned_coordinates
{
    uint16_t x;
    uint16_t y;
} unsigned_coordinates_t;

typedef struct signed_coordinates
{
    int8_t x;
    int8_t y;
} signed_coordinates_t;

typedef struct processing_arguments
{
    uint8_t* frame1;
    uint8_t* frame2;
    struct unsigned_coordinates* coordinates;
    uint8_t total_num_points;
    struct signed_coordinates* results;
} processing_arguments_t;

/**
 * @brief Function to perform optical flow processing using the parallelized PX4Flow algorithm.
 *
 * This function executes the optical flow algorithm. It may utilize multiple cores for parallel
 * processing, depending on the hardware capabilities (designed for 8-core processing on GAP8 and GAP9).
 *
 * @param arg        Pointer to the processing arguments (processing_arguments_t as void pointer).
 * @param num_cores  Number of processor cores to use for the flow computation.
 */
void flow(void* arg, uint8_t num_cores);

#endif /* FLOW_H */