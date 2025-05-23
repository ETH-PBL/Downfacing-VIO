#include "bf_matcher.h"

#define Abs(a)          (((int)(a)<0)?(-(a)):(a))

static inline uint8_t calculate_hamming_distance(orb_descriptor_t desc0, orb_descriptor_t desc1)
{
    uint8_t hamming_score = 0;
    hamming_score += __builtin_popcount(desc0[0] ^ desc1[0]);
    hamming_score += __builtin_popcount(desc0[1] ^ desc1[1]);
    hamming_score += __builtin_popcount(desc0[2] ^ desc1[2]);
    hamming_score += __builtin_popcount(desc0[3] ^ desc1[3]);
    hamming_score += __builtin_popcount(desc0[4] ^ desc1[4]);
    hamming_score += __builtin_popcount(desc0[5] ^ desc1[5]);
    hamming_score += __builtin_popcount(desc0[6] ^ desc1[6]);
    hamming_score += __builtin_popcount(desc0[7] ^ desc1[7]);
    return hamming_score;
}

uint16_t bf_match_max_flow(feature_match_t* matches,
                           orb_features_t* features0,
                           orb_features_t* features1,
                           uint16_t max_flow,
                           uint8_t hamming_threshold)
{
    uint16_t match_counter = 0;
    for(uint16_t idx1 = 0; idx1 < features1->kpt_counter; ++idx1)
    {
        uint8_t current_distance = hamming_threshold;
        /* Use length value to indicate no match */
        uint16_t matching_idx0 = features0->kpt_counter;
        for(uint16_t idx0 = 0; idx0 < features0->kpt_counter; ++idx0)
        {
            uint8_t new_distance = calculate_hamming_distance(features0->descs[idx0],features1->descs[idx1]);
            if(new_distance < current_distance)
            {
                point2D_u16_t kpt0 = features0->kpts[idx0];
                point2D_u16_t kpt1 = features1->kpts[idx1];
                if(Abs((int16_t)kpt0.x-kpt1.x) <= max_flow && Abs((int16_t)kpt0.y-kpt1.y) <= max_flow)
                {
                    current_distance = new_distance;
                    matching_idx0 = idx0;
                }
            }
        }
        if(matching_idx0 < features0->kpt_counter)
        {
            feature_match_t match = {matching_idx0, idx1, current_distance};
            matches[match_counter] = match;
            ++match_counter;
        }
    }
    return match_counter;
}

uint16_t two_way_filter(feature_match_t* matches,
                        uint16_t match_counter,
                        uint16_t start_offset,
                        uint16_t slice_size)
{
    uint16_t two_way_match_counter = 0;
    for (uint16_t i = start_offset; i < start_offset+slice_size; i++)
    {
        feature_match_t match = matches[i];
        uint8_t keep = 1;
        for (uint16_t j = 0; j < match_counter; j++)
        {
            if(i == j)
            {
                continue;
            }
            feature_match_t match_iter = matches[j];
            /* Comparing for equal or bigger score can lead to different results between single and multicore execution */
            if(match.feat_idx0 == match_iter.feat_idx0 && match.match_score >= match_iter.match_score)
            {
                keep = 0;
            }
        }
        if (keep == 1)
        {
            matches[start_offset+two_way_match_counter] = match;
            two_way_match_counter++;
        }
    }
    return two_way_match_counter;
}

uint16_t bf_match_two_way_max_flow(feature_match_t* matches,
                                   orb_features_t* features0,
                                   orb_features_t* features1,
                                   uint16_t max_flow,
                                   uint8_t hamming_threshold)
{
    uint16_t match_counter = bf_match_max_flow(matches, features0, features1, 
                                               max_flow, hamming_threshold);
    printf("Match counter %d\n",match_counter);
    uint16_t two_way_match_counter = two_way_filter(matches, match_counter, 0 , match_counter);
    printf("Two Way Match counter %d\n",two_way_match_counter);
    return two_way_match_counter;
}

typedef struct bf_matcher_arguments
{
    feature_match_t* matches;
    orb_features_t* features0;
    orb_features_t* features1;
    uint16_t* match_counter;
    uint16_t slice_size;
    uint16_t max_flow;
    uint8_t hamming_threshold;
} bf_matcher_arguments_t;

uint16_t reorganize_matches_memory(feature_match_t* matches, uint16_t* match_counters, uint16_t slice_size)
{
    uint8_t nb_cores = pi_cl_cluster_nb_cores();
    uint16_t match_counter = match_counters[0];
    for(uint8_t index = 1; index < nb_cores; ++index)
    {
        uint16_t nb_match_pool = match_counters[index];
        uint16_t offset = slice_size*index;
        for (uint8_t match_nr = 0; match_nr < nb_match_pool; ++match_nr)
        {
            matches[match_counter] = matches[offset+match_nr];
            ++match_counter;
        }
    }
    return match_counter;
}

void bf_match_max_flow_subset(void* args)
{
    bf_matcher_arguments_t* bfm_args = (bf_matcher_arguments_t*) args;
    uint16_t core_id = pi_core_id();
	uint16_t nb_cores = pi_cl_cluster_nb_cores();
    uint16_t nb_features1 = bfm_args->features1->kpt_counter;
    uint16_t slice_size = bfm_args->slice_size;
    uint32_t offset = slice_size*core_id;
    if (core_id == (nb_cores-1))
    {
        slice_size = nb_features1-offset;
    }
    orb_features_t temp_features1;
    temp_features1.kpts = bfm_args->features1->kpts + offset;
    temp_features1.descs = bfm_args->features1->descs + offset;
    temp_features1.kpt_counter = slice_size;
    bfm_args->match_counter[core_id] = bf_match_max_flow(bfm_args->matches+offset,
                                                         bfm_args->features0,
                                                         &temp_features1,
                                                         bfm_args->max_flow,
                                                         bfm_args->hamming_threshold);
}

typedef struct two_way_filter_arguments
{
    feature_match_t* matches;
    uint16_t* match_counter;
    uint16_t total_one_way_matches;
    uint16_t slice_size;
} two_way_filter_arguments_t;

void two_way_filter_subset(void* args)
{
    two_way_filter_arguments_t* twf_args = (two_way_filter_arguments_t*) args;
    uint16_t core_id = pi_core_id();
	uint16_t nb_cores = pi_cl_cluster_nb_cores();
    uint16_t slice_size = twf_args->slice_size;
    uint32_t offset = slice_size*core_id;
    if (core_id == (nb_cores-1))
    {
        slice_size = twf_args->total_one_way_matches-offset;
    }
    twf_args->match_counter[core_id] = two_way_filter(twf_args->matches,
                                                      twf_args->total_one_way_matches,
                                                      offset,
                                                      slice_size);
}

uint16_t bf_match_two_way_max_flow_multicore(feature_match_t* matches,
                                             orb_features_t* features0,
                                             orb_features_t* features1,
                                             uint16_t max_flow,
                                             uint8_t hamming_threshold)
{
    uint8_t nb_cores = pi_cl_cluster_nb_cores();
    uint16_t slice_size;
    if(features1->kpt_counter % nb_cores == 0)
    {
        slice_size = features1->kpt_counter/nb_cores;
    }
    else
    {
        slice_size = features1->kpt_counter/nb_cores+1;
    }
    uint16_t utility_counters[8];
    bf_matcher_arguments_t bfm_args = {matches, features0, features1, utility_counters, slice_size, max_flow, hamming_threshold};
    printf("4.1. One Way Matching Parallelized\n");
    pi_perf_reset();
    pi_perf_start();
    pi_cl_team_fork(pi_cl_cluster_nb_cores(), bf_match_max_flow_subset, &bfm_args);
    pi_perf_stop(); 
    printf("Number of cycles %d\n",pi_perf_read(PI_PERF_CYCLES));
    uint16_t match_counter = reorganize_matches_memory(matches, utility_counters, slice_size);
    printf("Match counter %d\n",match_counter);

    if(match_counter % nb_cores == 0)
    {
        slice_size = match_counter/nb_cores;
    }
    else
    {
        slice_size = match_counter/nb_cores+1;
    }
    two_way_filter_arguments_t twf_args = {matches, utility_counters, match_counter, slice_size};
    printf("4.2. Two Way Filter Parallelized\n");
    pi_perf_reset();
    pi_perf_start();
    pi_cl_team_fork(pi_cl_cluster_nb_cores(), two_way_filter_subset, &twf_args);
    pi_perf_stop(); 
    printf("Number of cycles %d\n",pi_perf_read(PI_PERF_CYCLES));
    uint16_t two_way_match_counter = reorganize_matches_memory(matches, utility_counters, slice_size);
    printf("Two Way Match counter %d\n",two_way_match_counter);
    return two_way_match_counter;
}
