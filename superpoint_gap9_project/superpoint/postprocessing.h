#ifndef __POSTPROCESSING_H__
#define __POSTPROCESSING_H__

#include "pmsis.h"
#include "img_feature_definitions/img_feature_definitions.h"

/**
 * @brief Extracts keypoints from a SuperPoint keypoint heatmap.
 *
 * This function processes a SuperPoint keypoint heatmap (including dustbins) to extract 2D keypoints.
 *
 * @param kpts        Pointer to an array where the extracted keypoints will be stored.
 * @param features    Pointer to an array containing the SuperPoint keypoint heatmap (including dustbins).
 * @param img_width   Width of the input image.
 * @param img_height  Height of the input image.
 * @param reduction Reduction factor of the neural network (typically 8 for SuperPoint).
 * @return The number of keypoints extracted and stored in the output array.
 */
uint16_t extract_kpts(point2D_u16_t* kpts,
                      uint8_t* features,
                      uint16_t img_width,
                      uint16_t img_height,
                      uint8_t reduction);

/**
 * @brief Initializes the in-place matcher for feature matching.
 *
 * This function sets up any necessary data structures required for
 * performing in-place feature matching operations. It should be called
 * before using any matcher-related functionality.
 */
void inplace_matcher_init();

/**
 * @brief Performs in-place two-way matching of feature descriptors using a max-flow approach.
 *
 * This function interpolates feature descriptors from SuperPoint descriptor grids (descs0 and descs1) for the given
 * keypoint locations (keypoints0 and keypoints1). These interpolated descriptors are then matched in place. The
 * matching is performed using a two-way consistency check and a maximum flow constraint to limit the number of
 * matches. The results are stored in the provided matches array.
 *
 * @param descs0 Pointer to the first SuperPoint descriptor grid.
 * @param descs1 Pointer to the second SuperPoint descriptor grid.
 * @param keypoints0 Pointer to the first set of keypoints (size: kpt_count0).
 * @param keypoints1 Pointer to the second set of keypoints (size: kpt_count1).
 * @param matches Pointer to the output array for storing matched feature pairs.
 * @param kpt_count0 Number of keypoints in the first set.
 * @param kpt_count1 Number of keypoints in the second set.
 * @param max_flow Maximum pixel displacement of matches to be found (flow constraint).
 * @param img_width Width of the image.
 * @param img_height Height of the image.
 * @param reduction Reduction factor of the neural network (typically 8 for SuperPoint).
 * @param cosine_sim_threshold Minimum cosine similarity threshold for accepting a match.
 *
 * @return The number of valid matches found and stored in the matches array.
 */
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
                                        uint8_t cosine_sim_threshold);

#endif /* __POSTPROCESSING_H__ */
