#ifndef __RIGID_BODY_MOTION_ESTIMATION__H__
#define __RIGID_BODY_MOTION_ESTIMATION__H__

#include "pmsis.h"
#include "img_feature_definitions/img_feature_definitions.h"

typedef struct
{
    float flow_x;
    float flow_y;
    float rot_z;
} rigid_body_motion_t;

/**
 * @brief Estimates the rigid body motion between two sets of keypoints.
 *
 * This function computes the rigid body transformation (i.e., planar rotation and translation)
 * that best aligns the matched keypoints from two frames. It uses the provided
 * matches to determine correspondences and outputs the estimated motion parameters.
 *
 * @param keypoints0      Pointer to the array of keypoints from the first frame.
 * @param keypoints1      Pointer to the array of keypoints from the second frame.
 * @param matches         Pointer to the array of feature matches between keypoints0 and keypoints1.
 * @param motion          Pointer to the structure where the estimated rigid body motion will be stored.
 * @param inlier_mask     Pointer to an array where the inlier mask will be stored (1 for inlier, 0 for outlier).
 * @param match_counter   Number of matches (size of the matches array).
 * @param img_width       Width of the input images.
 * @param img_height      Height of the input images.
 */
void rigid_body_motion_estimation(point2D_u16_t* keypoints0,
                                  point2D_u16_t* keypoints1,
                                  feature_match_t* matches,
                                  rigid_body_motion_t* motion,
                                  uint8_t* inlier_mask,
                                  uint16_t match_counter,
                                  uint16_t img_width,
                                  uint16_t img_height);

/**
 * @brief Estimates the rigid body motion between two sets of keypoints using robust methods.
 *
 * This function computes the rigid body transformation (i.e., planar rotation and translation)
 * that best aligns the matched keypoints from two frames. It uses the provided
 * matches to determine correspondences and outputs the estimated motion parameters.
 * It uses robust estimation techniques to handle outliers in the feature matches.
 *
 * @param keypoints0      Pointer to the array of 2D keypoints from the first image (reference frame).
 * @param keypoints1      Pointer to the array of 2D keypoints from the second image (current frame).
 * @param matches         Pointer to the array of feature matches between keypoints0 and keypoints1.
 * @param motion          Pointer to the output structure where the estimated rigid body motion will be stored.
 * @param match_counter   Number of valid feature matches.
 * @param img_width       Width of the input images.
 * @param img_height      Height of the input images.
 * @param max_flow        Maximum allowed flow (displacement) between matched keypoints.
 */
void robust_rigid_body_motion_estimation(point2D_u16_t* keypoints0,
                                         point2D_u16_t* keypoints1,
                                         feature_match_t* matches,
                                         rigid_body_motion_t* motion,
                                         uint16_t match_counter,
                                         uint16_t img_width,
                                         uint16_t img_height,
                                         uint16_t max_flow);


#endif /* __RIGID_BODY_MOTION_ESTIMATION__H__ */
