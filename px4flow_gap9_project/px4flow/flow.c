#include "flow.h"

#include "pmsis.h"

#define BOTTOM_FLOW_FEATURE_THRESHOLD 40
#define BOTTOM_FLOW_VALUE_THRESHOLD 5000


/* ARM to pulp alias */
static inline uint32_t __USAD8(v4u in1, v4u in2)
{
    /* Convert input to pulp format */
    v4u max = __builtin_pulp_maxu4(in1,in2);
    v4u min = __builtin_pulp_minu4(in1,in2);
    v4u diff = __builtin_pulp_sub4(max,min);
    return __builtin_pulp_dotuspsc4(diff,1);
}

/* ARM to pulp alias */
static inline uint32_t __USADA8(v4u in1, v4u in2, uint32_t accumulate)
{
    /* Convert input to pulp format */
    v4u max = __builtin_pulp_maxu4(in1,in2);
    v4u min = __builtin_pulp_minu4(in1,in2);
    v4u diff = __builtin_pulp_sub4(max,min);
    return __builtin_pulp_sdotuspsc4(diff,1,accumulate);
}


/* ARM to pulp alias */
static inline v4u __UHADD8(v4u in1, v4u in2)
{
    /* ToDo: Find overflow safe avg instructions */
    v4u and = __builtin_pulp_and4(in1,in2);
    v4u exor = __builtin_pulp_exor4(in1,in2);
    v4u unity = {1,1,1,1};
    v4u exor_half = __builtin_pulp_srl4(exor, unity);
    return __builtin_pulp_add4(and, exor_half);
}


/**
 * @brief Compute SAD of two 8x8 pixel windows.
 *
 * @param image1  an array holding pixel data
 * @param image2  an array holding pixel data
 * @param off1X   x coordinate of upper left corner of pattern in image1
 * @param off1Y   y coordinate of upper left corner of pattern in image1
 * @param off2X   x coordinate of upper left corner of pattern in image2
 * @param off2Y   y coordinate of upper left corner of pattern in image2
 *
 * @return        SAD of two 8x8 pixel windows
 */
static inline uint32_t compute_sad_8x8(uint8_t *image1, uint8_t *image2, uint16_t off1X, uint16_t off1Y, uint16_t off2X, uint16_t off2Y, uint16_t row_size)
{
    /* calculate position in image buffer */
    uint16_t off1 = off1Y * row_size + off1X; // image1
    uint16_t off2 = off2Y * row_size + off2X; // image2

    uint32_t acc;
    acc = __USAD8 (*(v4u*) &image1[off1 + 0 + 0 * row_size], *(v4u*) &image2[off2 + 0 + 0 * row_size]);
    acc = __USADA8(*(v4u*) &image1[off1 + 4 + 0 * row_size], *(v4u*) &image2[off2 + 4 + 0 * row_size], acc);

    acc = __USADA8(*(v4u*) &image1[off1 + 0 + 1 * row_size], *(v4u*) &image2[off2 + 0 + 1 * row_size], acc);
    acc = __USADA8(*(v4u*) &image1[off1 + 4 + 1 * row_size], *(v4u*) &image2[off2 + 4 + 1 * row_size], acc);

    acc = __USADA8(*(v4u*) &image1[off1 + 0 + 2 * row_size], *(v4u*) &image2[off2 + 0 + 2 * row_size], acc);
    acc = __USADA8(*(v4u*) &image1[off1 + 4 + 2 * row_size], *(v4u*) &image2[off2 + 4 + 2 * row_size], acc);

    acc = __USADA8(*(v4u*) &image1[off1 + 0 + 3 * row_size], *(v4u*) &image2[off2 + 0 + 3 * row_size], acc);
    acc = __USADA8(*(v4u*) &image1[off1 + 4 + 3 * row_size], *(v4u*) &image2[off2 + 4 + 3 * row_size], acc);

    acc = __USADA8(*(v4u*) &image1[off1 + 0 + 4 * row_size], *(v4u*) &image2[off2 + 0 + 4 * row_size], acc);
    acc = __USADA8(*(v4u*) &image1[off1 + 4 + 4 * row_size], *(v4u*) &image2[off2 + 4 + 4 * row_size], acc);

    acc = __USADA8(*(v4u*) &image1[off1 + 0 + 5 * row_size], *(v4u*) &image2[off2 + 0 + 5 * row_size], acc);
    acc = __USADA8(*(v4u*) &image1[off1 + 4 + 5 * row_size], *(v4u*) &image2[off2 + 4 + 5 * row_size], acc);

    acc = __USADA8(*(v4u*) &image1[off1 + 0 + 6 * row_size], *(v4u*) &image2[off2 + 0 + 6 * row_size], acc);
    acc = __USADA8(*(v4u*) &image1[off1 + 4 + 6 * row_size], *(v4u*) &image2[off2 + 4 + 6 * row_size], acc);

    acc = __USADA8(*(v4u*) &image1[off1 + 0 + 7 * row_size], *(v4u*) &image2[off2 + 0 + 7 * row_size], acc);
    acc = __USADA8(*(v4u*) &image1[off1 + 4 + 7 * row_size], *(v4u*) &image2[off2 + 4 + 7 * row_size], acc);

    return acc;
}

/**
 * @brief Compute the average pixel gradient of all horizontal and vertical steps
 *
 * TODO compute_diff is not appropriate for low-light mode images
 *
 * @param image  the array holding pixel data
 * @param offX   x coordinate of upper left corner of 8x8 pattern in image
 * @param offY   y coordinate of upper left corner of 8x8 pattern in image
 *
 * @return       value indicating how suitable a pattern is for optical flow calculation
 */
static inline uint32_t compute_diff(uint8_t *image, uint16_t offX, uint16_t offY, uint16_t row_size)
{
    /* calculate position in image buffer */
    uint16_t off = (offY + 2) * row_size + (offX + 2);
    /* we calculate only the center 4x4 pattern */
    uint32_t acc;

    /* calculate vertical gradient */
    acc = __USAD8 (*(v4u*) &image[off + 0 + 0 * row_size], *(v4u*) &image[off + 0 + 1 * row_size]);
    acc = __USADA8(*(v4u*) &image[off + 0 + 1 * row_size], *(v4u*) &image[off + 0 + 2 * row_size], acc);
    acc = __USADA8(*(v4u*) &image[off + 0 + 2 * row_size], *(v4u*) &image[off + 0 + 3 * row_size], acc);

    /* we need to get columns */
    v4u col1 = {image[off + 0 + 0 * row_size], image[off + 0 + 1 * row_size], image[off + 0 + 2 * row_size], image[off + 0 + 3 * row_size]};
    v4u col2 = {image[off + 1 + 0 * row_size], image[off + 1 + 1 * row_size], image[off + 1 + 2 * row_size], image[off + 1 + 3 * row_size]};
    v4u col3 = {image[off + 2 + 0 * row_size], image[off + 2 + 1 * row_size], image[off + 2 + 2 * row_size], image[off + 2 + 3 * row_size]};
    v4u col4 = {image[off + 3 + 0 * row_size], image[off + 3 + 1 * row_size], image[off + 3 + 2 * row_size], image[off + 3 + 3 * row_size]};

    /* calculate horizontal gradient */
    acc = __USADA8(col1, col2, acc);
    acc = __USADA8(col2, col3, acc);
    acc = __USADA8(col3, col4, acc);

    return acc;

}

/**
 * @brief Compute SAD distances of subpixel shift of two 8x8 pixel patterns.
 *
 * @param image1  an array holding pixel data
 * @param image2  an array holding pixel data
 * @param off1X   x coordinate of upper left corner of pattern in image1
 * @param off1Y   y coordinate of upper left corner of pattern in image1
 * @param off2X   x coordinate of upper left corner of pattern in image2
 * @param off2Y   y coordinate of upper left corner of pattern in image2
 * @param acc     array to store SAD distances for shift in every direction
 */
static inline void compute_subpixel(uint8_t *image1, uint8_t *image2, uint16_t off1X, uint16_t off1Y, uint16_t off2X, uint16_t off2Y, uint32_t *acc, uint16_t row_size)
{
    /* hardcoded for TILE_SIZE of 8 (same as for the original PX4FLOW) */
    /* calculate position in image buffer */
    uint16_t off1 = off1Y * row_size + off1X; // image1
    uint16_t off2 = off2Y * row_size + off2X; // image2

    for (uint16_t i = 0; i < 8; i++)
    {
        acc[i] = 0;
    }

    v4u right_upper [3];
    right_upper[0] = __UHADD8(*(v4u*) &image2[off2-row_size-1+0],*(v4u*) &image2[off2-row_size-0+0]);
    right_upper[1] = __UHADD8(*(v4u*) &image2[off2-row_size-1+4],*(v4u*) &image2[off2-row_size-0+4]);
    right_upper[2] = __UHADD8(*(v4u*) &image2[off2-row_size-1+8],*(v4u*) &image2[off2-row_size-0+8]);

    /*
     *  + - + - + - +
     *  |   |   |   |
     *  + - 5 6 7 - +
     *  |   4 X 0   |
     *  + - 3 2 1 - +
     *  |   |   |   |
     *  + - + - + - +
    */

    /* calculate the i-th interpolation line */
    for(uint16_t i = 0; i < 9; i++)
    {
        /* the upper right values become the lower ones */
        v4u right_lower [3];
        v4u diag[3];
        v4u below[3];

        right_lower[0] = right_upper[0];
        right_lower[1] = right_upper[1];
        right_lower[2] = right_upper[2];

        right_upper[0] = __UHADD8(*(v4u*) &image2[off2+i*row_size-1+0],*(v4u*) &image2[off2+i*row_size-0+0]);
        right_upper[1] = __UHADD8(*(v4u*) &image2[off2+i*row_size-1+4],*(v4u*) &image2[off2+i*row_size-0+4]);
        right_upper[2] = __UHADD8(*(v4u*) &image2[off2+i*row_size-1+8],*(v4u*) &image2[off2+i*row_size-0+8]);

        diag[0] = __UHADD8(right_upper[0],right_lower[0]);
        diag[1] = __UHADD8(right_upper[1],right_lower[1]);
        diag[2] = __UHADD8(right_upper[2],right_lower[2]);

        uint8_t* right_a = (uint8_t*) right_upper;
        uint8_t* diag_a = (uint8_t*) diag;

        if(i < 8)
        {
            /* the i-th interpolation line can be used to calculate the i-th line of offsets 0,4,5,6,7 */
            acc[0] = __USADA8 ((*((v4u*) &image1[off1 + 0 + i * row_size])), *(v4u*) &right_a[1], acc[0]);
            acc[4] = __USADA8 ((*((v4u*) &image1[off1 + 0 + i * row_size])), *(v4u*) &right_a[0], acc[4]);
            acc[0] = __USADA8 ((*((v4u*) &image1[off1 + 0 + i * row_size + 4])), *(v4u*) &right_a[1+4], acc[0]);
            acc[4] = __USADA8 ((*((v4u*) &image1[off1 + 0 + i * row_size + 4])), *(v4u*) &right_a[0+4], acc[4]);

            acc[5] = __USADA8 ((*((v4u*) &image1[off1 + 0 + i * row_size])), *(v4u*) &diag_a[0], acc[5]);
            acc[7] = __USADA8 ((*((v4u*) &image1[off1 + 0 + i * row_size])), *(v4u*) &diag_a[1], acc[7]);
            acc[5] = __USADA8 ((*((v4u*) &image1[off1 + 0 + i * row_size + 4])), *(v4u*) &diag_a[0+4], acc[5]);
            acc[7] = __USADA8 ((*((v4u*) &image1[off1 + 0 + i * row_size + 4])), *(v4u*) &diag_a[1+4], acc[7]);
        }
        if(i > 0)
        {
            /* the i-th interpolation line can be used to calculate the (i-1)-th line of offsets 1,2,3 */
            acc[1] = __USADA8 ((*((v4u*) &image1[off1 + 0 + (i-1) * row_size])), *(v4u*) &diag_a[1], acc[1]);
            acc[3] = __USADA8 ((*((v4u*) &image1[off1 + 0 + (i-1) * row_size])), *(v4u*) &diag_a[0], acc[3]);
            acc[1] = __USADA8 ((*((v4u*) &image1[off1 + 0 + (i-1) * row_size + 4])), *(v4u*) &diag_a[1+4], acc[1]);
            acc[3] = __USADA8 ((*((v4u*) &image1[off1 + 0 + (i-1) * row_size + 4])), *(v4u*) &diag_a[0+4], acc[3]);
        }

        below[0] = __UHADD8(*(v4u*) &image2[off2+(i-1)*row_size-1+0],*(v4u*) &image2[off2+i*row_size-1+0]);
        below[1] = __UHADD8(*(v4u*) &image2[off2+(i-1)*row_size-1+4],*(v4u*) &image2[off2+i*row_size-1+4]);
        below[2] = __UHADD8(*(v4u*) &image2[off2+(i-1)*row_size-1+8],*(v4u*) &image2[off2+i*row_size-1+8]);
        uint8_t* below_a = (uint8_t*) below;

        if(i < 8)
        {
            /* the i-th interpolation line can be used to calculate the i-th line of offsets 0,4,5,6,7 */
            acc[6] = __USADA8 ((*((v4u*) &image1[off1 + 0 + i * row_size])), *(v4u*) &below_a[1], acc[6]);
            acc[6] = __USADA8 ((*((v4u*) &image1[off1 + 0 + i * row_size + 4])), *(v4u*) &below_a[1+4], acc[6]);
        }
        if(i > 0)
        {
            /* the i-th interpolation line can be used to calculate the (i-1)-th line of offsets 1,2,3 */
            acc[2] = __USADA8 ((*((v4u*) &image1[off1 + 0 + (i-1) * row_size])), *(v4u*) &below_a[1], acc[2]);
            acc[2] = __USADA8 ((*((v4u*) &image1[off1 + 0 + (i-1) * row_size + 4])), *(v4u*) &below_a[1+4], acc[2]);
        }
    }
}

void flow(void* arg, uint8_t num_cores)
{
    struct processing_arguments* args = (struct processing_arguments*) arg;
    uint32_t core_id = pi_core_id(), cluster_id = pi_cluster_id();

    const int16_t winmin = -SEARCH_SIZE;
    const int16_t winmax = SEARCH_SIZE;

    uint8_t* image1 = args->frame1;
    uint8_t* image2 = args->frame2;

    struct signed_coordinates* results = args->results;


    for(uint8_t point_i = core_id; point_i < args->total_num_points; point_i += num_cores)
    {
        uint16_t j = args->coordinates[point_i].y;
        uint16_t i = args->coordinates[point_i].x;

        /* test pixel if it is suitable for flow tracking */
        uint32_t diff = compute_diff(image1, i, j, (uint16_t) IMAGE_WIDTH);
        if (diff < BOTTOM_FLOW_FEATURE_THRESHOLD)
        {
            /* Inappropriate point */
            results[point_i].x = -128;
            results[point_i].y = -128;
            continue;
        }

        uint32_t dist = 0xFFFFFFFF; // set initial distance to "infinity"
        int8_t sumx = 0;
        int8_t sumy = 0;
        int8_t ii, jj;

        uint8_t *base1 = image1 + j * (uint16_t) IMAGE_WIDTH + i;

        for (jj = winmin; jj <= winmax; jj++)
        {
            uint8_t *base2 = image2 + (j+jj) * (uint16_t) IMAGE_WIDTH + i;

            for (ii = winmin; ii <= winmax; ii++)
            {
                uint32_t temp_dist = compute_sad_8x8(image1, image2, i, j, i + ii, j + jj, (uint16_t) IMAGE_WIDTH);
                if (temp_dist < dist)
                {
                    sumx = ii;
                    sumy = jj;
                    dist = temp_dist;
                }
            }
        }
        if (dist < BOTTOM_FLOW_VALUE_THRESHOLD)
        {
            uint32_t acc[8] = {0};           // subpixels
            compute_subpixel(image1, image2, i, j, i + sumx, j + sumy, acc, (uint16_t) IMAGE_WIDTH);

            uint32_t mindist = dist; // best SAD until now
            uint8_t mindir = 8; // direction 8 for no direction
            for(uint8_t k = 0; k < 8; k++)
            {
                if (acc[k] < mindist)
                {
                    // SAD becomes better in direction k
                    mindist = acc[k];
                    mindir = k;
                }
            }
            int8_t x_direction_half_pixel = 2*sumx;
            int8_t y_direction_half_pixel = 2*sumy;
            if(mindir == 0 || mindir == 1 || mindir == 7) x_direction_half_pixel += 1;
            if(mindir == 3 || mindir == 4 || mindir == 5) x_direction_half_pixel -= 1;

            if(mindir == 1 || mindir == 2 || mindir == 3) y_direction_half_pixel += 1;
            if(mindir == 5 || mindir == 6 || mindir == 7) y_direction_half_pixel -= 1;

            results[point_i].x = x_direction_half_pixel;
            results[point_i].y = y_direction_half_pixel;
        }
        else
        {
            /* Inappropriate point */
            results[point_i].x = -128;
            results[point_i].y = -128;
        }

    }

}