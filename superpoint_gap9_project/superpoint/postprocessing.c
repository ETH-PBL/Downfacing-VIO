#include "postprocessing.h"
#include "math.h"

#define Abs(a)          (((int)(a)<0)?(-(a)):(a))

static uint8_t* features_0;
static uint8_t* features_1;

uint16_t extract_kpts(point2D_u16_t* kpts,
                      uint8_t* features,
                      uint16_t img_width,
                      uint16_t img_height,
                      uint8_t reduction)
{
    /* Default superpoint value */
    uint16_t keypoint_counter = 0;
    uint16_t image_border = 4;
    for(uint16_t height_i = 0; height_i < img_height/reduction; height_i++)
    {
        for(uint16_t width_i = 0; width_i < img_width/reduction; width_i++)
        {
            uint8_t highest_score = 0;
            uint16_t feature_i_highest_score = 0;
            /* Ignore dustbin */
            for(uint8_t feature_i = 0; feature_i < (reduction*reduction); feature_i++)
            {
                /* Since the features are quantized and normalized, we take a rudimentary approach to
                   confindence thresholding and NMS and select the highest scoring pixel per bin
                   or none if the highest score is below a threshold. */
                point2D_u16_t coord = {width_i*reduction + feature_i%reduction, height_i*reduction + feature_i/reduction};
                if(coord.x < image_border || coord.y < image_border ||
                   coord.x >= img_width - image_border || coord.x >= img_height - image_border)
                {
                    continue;
                }
                uint16_t feature_offset = feature_i + height_i*(img_width/reduction)*65 + width_i*65;
                uint8_t score = features[feature_offset];
                if (score > highest_score)
                {
                    highest_score = score;
                    feature_i_highest_score = feature_i;
                    kpts[keypoint_counter] = coord;
                }

            }
            if(highest_score > 115 && feature_i_highest_score != 64)
            {
                keypoint_counter++;
            }
        }
    }
    return keypoint_counter;
}

void inplace_matcher_init()
{
    features_0 = pi_l2_malloc((256));
    features_1 = pi_l2_malloc((256));
}


void interpolate_desc(  uint8_t* descs,
                        uint8_t* interpolated_desc,
                        point2D_u16_t kpt,
                        uint16_t reduced_width,
                        uint16_t reduced_height,
                        uint8_t reduction)
{
    uint16_t idx_x = (kpt.x - 4) /reduction;
    uint16_t idx_y = (kpt.y - 4) /reduction;
    uint16_t double_center_offset_x = kpt.x*2 - idx_x * reduction *2 - 7;
    uint16_t double_center_offset_y = kpt.y*2 - idx_y * reduction *2 - 7;
    uint16_t weight_tl = (16 - double_center_offset_x) * (16 - double_center_offset_y);
    uint16_t weight_tr = double_center_offset_x * (16 - double_center_offset_y);
    uint16_t weight_bl = (16 - double_center_offset_x) * double_center_offset_y;
    uint16_t weight_br = double_center_offset_x * double_center_offset_y;
    uint16_t sum = weight_tl + weight_tr + weight_bl + weight_br;
    uint32_t coord_offset_tl = (idx_x + idx_y*reduced_width)*256;
    uint32_t coord_offset_tr = ((idx_x+1) + idx_y*reduced_width)*256;
    uint32_t coord_offset_bl = (idx_x + (idx_y+1)*reduced_width)*256;
    uint32_t coord_offset_br = ((idx_x+1) + (idx_y+1)*reduced_width)*256;
    for(uint16_t feat_index = 0; feat_index < 256; feat_index++)
    {
        uint16_t interpolation_sum = 0;
        interpolation_sum += weight_tl*descs[coord_offset_tl+feat_index];
        interpolation_sum += weight_tr*descs[coord_offset_tr+feat_index];
        interpolation_sum += weight_bl*descs[coord_offset_bl+feat_index];
        interpolation_sum += weight_br*descs[coord_offset_br+feat_index];
        interpolated_desc[feat_index] = (uint8_t)(interpolation_sum/256);
    }
}

uint8_t calculate_cosine_similarity(uint8_t* descs0,
                                    uint8_t* desc1,
                                    uint32_t feat_sum1,
                                    point2D_u16_t kpt0,
                                    uint16_t reduced_width,
                                    uint16_t reduced_height,
                                    uint8_t reduction
                                    )
{
    uint32_t coord_offset = ((kpt0.x/reduction) + (kpt0.y/reduction)*reduced_width)*256;
    uint32_t feat_sum0 = 0;
    uint32_t dot_prod = 0;
    interpolate_desc(descs0, features_0, kpt0, reduced_width, reduced_height, reduction);
    for(uint16_t feat_index = 0; feat_index < 256; feat_index++)
    {
        uint8_t current_feature = features_0[feat_index];
        feat_sum0 += current_feature*current_feature;
        dot_prod += current_feature * desc1[feat_index];
    }
    feat_sum0 = (uint32_t) sqrtf(feat_sum0);
    uint32_t result = ((dot_prod / feat_sum0) * 255) / feat_sum1;
    // printf("Res %d dotprod %d sum0 %d sum1 %d\n", result, dot_prod, feat_sum0, feat_sum1);
    return result;
}

uint16_t inplace_match_one_way_max_flow(uint8_t* descs0,
                                        uint8_t* descs1,
                                        point2D_u16_t* keypoints0,
                                        point2D_u16_t* keypoints1,
                                        feature_match_t* matches,
                                        uint16_t kpt_count0,
                                        uint16_t kpt_count1,
                                        uint16_t max_flow,
                                        uint16_t img_width,
                                        uint16_t img_height,
                                        uint8_t reduction,
                                        uint8_t cosine_sim_threshold)
{
    uint16_t match_counter = 0;
    uint16_t reduced_width = img_width / reduction;
    uint16_t reduced_height = img_height / reduction;
    for(uint16_t idx1 = 0; idx1 < kpt_count1; ++idx1)
    {
        uint32_t coord_offset = ((keypoints1[idx1].x/reduction) + (keypoints1[idx1].y/reduction)*reduced_width)*256;
        uint32_t feat_sum1 = 0;
        point2D_u16_t kpt1 = keypoints1[idx1];
        interpolate_desc(descs1,features_1,kpt1,reduced_width,reduced_height,reduction);
        for(uint16_t feat_index = 0; feat_index < 256; feat_index++)
        {
            feat_sum1 += features_1[feat_index]*features_1[feat_index];
        }
        feat_sum1 = (uint32_t) sqrtf(feat_sum1);
        uint8_t current_similarity = cosine_sim_threshold;
        /* Use length value to indicate no match */
        uint16_t matching_idx0 = kpt_count0;
        for(uint16_t idx0 = 0; idx0 < kpt_count0; ++idx0)
        {
            point2D_u16_t kpt0 = keypoints0[idx0];
            uint8_t new_similarity = calculate_cosine_similarity(descs0,features_1,feat_sum1,kpt0,reduced_width,reduced_height,reduction);
            if(new_similarity > current_similarity)
            {
                if(Abs((int16_t)kpt0.x-kpt1.x) <= max_flow && Abs((int16_t)kpt0.y-kpt1.y) <= max_flow)
                {
                    current_similarity = new_similarity;
                    matching_idx0 = idx0;
                }
            }
        }
        if(matching_idx0 < kpt_count0)
        {
            feature_match_t match = {matching_idx0, idx1, current_similarity};
            matches[match_counter] = match;
            ++match_counter;
        }
    }
    printf("Match counter %d\n\n",match_counter);
    return match_counter;
}

uint16_t inplace_match_two_way_max_flow(uint8_t* descs0,
                                        uint8_t* descs1,
                                        point2D_u16_t* keypoints0,
                                        point2D_u16_t* keypoints1,
                                        feature_match_t* matches,
                                        uint16_t kpt_count0,
                                        uint16_t kpt_count1,
                                        uint16_t max_flow,
                                        uint16_t img_width,
                                        uint16_t img_height,
                                        uint8_t reduction,
                                        uint8_t cosine_sim_threshold)
{
    uint16_t match_counter = inplace_match_one_way_max_flow(descs0, descs1, keypoints0, keypoints1,
                                                            matches, kpt_count0, kpt_count1, max_flow,
                                                            img_width, img_height, reduction, cosine_sim_threshold);
    uint16_t two_way_match_counter = 0;
    for (uint16_t i = 0; i < match_counter; i++)
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
            if(match.feat_idx0 == match_iter.feat_idx0 && match.match_score >= match_iter.match_score)
            {
                keep = 0;
            }
        }
        if (keep == 1)
        {
            matches[two_way_match_counter] = match;
            two_way_match_counter++;
        }
    }
    printf("Two Way Match counter %d\n\n",two_way_match_counter);
    return two_way_match_counter;
}
