#ifndef __IMG_FEATURE_DEFINITIONS_H__
#define __IMG_FEATURE_DEFINITIONS_H__

#include "pmsis.h"

typedef struct image_data
{
    uint16_t width;
    uint16_t height;
    uint8_t* pixels;
} image_data_t;

typedef struct point2D_u16
{
    uint16_t x;
    uint16_t y;
} point2D_u16_t;

typedef struct feature_match
{
    uint16_t feat_idx0;
    uint16_t feat_idx1;
    uint8_t match_score;
} feature_match_t;


#endif /* __IMG_FEATURE_DEFINITIONS_H__ */
