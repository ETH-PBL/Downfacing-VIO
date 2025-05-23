#include "motion_histogram.h"

void estimate_motion_histogram(rigid_body_motion_t* motion, signed_coordinates_t* displacements)
{
    int8_t winmin = -SEARCH_SIZE;
    int8_t winmax = SEARCH_SIZE;
    uint8_t histogram_size = 2*(winmax-winmin+1)+1;

    uint16_t histx[HIST_SIZE] = {0};
    uint16_t histy[HIST_SIZE] = {0};
    for(uint8_t i = 0 ; i < (NUM_BLOCKS*NUM_BLOCKS); ++i)
    {
        if(displacements[i].x != -128)
        {
            uint8_t hist_index_x = displacements[i].x + (winmax-winmin+1);
            uint8_t hist_index_y = displacements[i].y + (winmax-winmin+1);
            histx[hist_index_x]++;
            histy[hist_index_y]++;
        }
    }


    uint16_t maxvaluex = 0;
    uint16_t maxvaluey = 0;
    uint16_t maxpositionx = 0;
    uint16_t maxpositiony = 0;
    for(uint8_t j = 0; j < histogram_size; ++j)
    {
        if(histx[j] > maxvaluex)
        {
            maxvaluex = histx[j];
            maxpositionx = j;
        }
        if(histy[j] > maxvaluey)
        {
            maxvaluey = histy[j];
            maxpositiony = j;
        }
    }

    uint16_t hist_x_min = maxpositionx;
    uint16_t hist_x_max = maxpositionx+2;
    uint16_t hist_y_min = maxpositiony;
    uint16_t hist_y_max = maxpositiony+2;

    if(maxpositionx < 2)
    {
        hist_x_min = 0;
    }
    else
    {
        hist_x_min -= 2;
    }
    if(maxpositiony < 2)
    {
        hist_y_min = 0;
    }
    else
    {
        hist_y_min -= 2;
    }
    if(hist_x_max >= histogram_size)
    {
        hist_x_max = histogram_size - 1;
    }
    if(hist_y_max >= histogram_size)
    {
        hist_y_max = histogram_size - 1;
    }

    uint16_t hist_x_value = 0;
    uint16_t hist_x_weight = 0;

    uint16_t hist_y_value = 0;
    uint16_t hist_y_weight = 0;

    for (uint8_t h = hist_x_min; h < hist_x_max+1; h++)
    {
        hist_x_value += h*histx[h];
        hist_x_weight += histx[h];
    }

    for (uint8_t h = hist_y_min; h<hist_y_max+1; h++)
    {
        hist_y_value += h*histy[h];
        hist_y_weight += histy[h];
    }

    motion->flow_x = (((float) hist_x_value)/((float) hist_x_weight) - (winmax-winmin+1)) / 2.0f;
    motion->flow_y = (((float) hist_y_value)/((float) hist_y_weight) - (winmax-winmin+1)) / 2.0f;
    motion->rot_z = 0.0f;
}
