#ifndef ___ORB_GAP_H__
#define ___ORB_GAP_H__

#include "pmsis.h"
#include "img_feature_definitions/img_feature_definitions.h"
#include "orb.h"

/**
 * @brief Initializes the ORB (Oriented FAST and Rotated BRIEF) storage for GAP9 platform.
 *
 * This function sets up the necessary storage structures and resources required for
 * ORB feature extraction on the GAP9 device. It should be called before performing
 * any ORB-related operations.
 *
 * @param cluster_device Pointer to the GAP9 cluster device structure.
 * @param image_width    Width of the input image in pixels.
 * @param image_height   Height of the input image in pixels.
 */
void initialize_orb_storage_gap(pi_device_t* cluster_device, int16_t image_width, int16_t image_height);

/**
 * @brief Detects ORB (Oriented FAST and Rotated BRIEF) features and computes their descriptors on a single core.
 *
 * This function processes the given image to identify keypoints using the ORB algorithm and computes their corresponding descriptors.
 *
 * @param img Pointer to the input image data structure.
 * @param features Pointer to the structure where detected ORB features and their descriptors will be stored.
 */
void orb_detect_and_compute_single_core(image_data_t* img, orb_features_t* features);

/**
 * @brief Detects ORB (Oriented FAST and Rotated BRIEF) features and computes their descriptors using multiple cores.
 *
 * This function processes the input image to detect keypoints and compute their corresponding ORB descriptors,
 * leveraging multi-core processing for improved performance.
 *
 * @param img Pointer to the input image data structure.
 * @param features Pointer to the structure where detected features and their descriptors will be stored.
 */
void orb_detect_and_compute_multi_core(image_data_t* img, orb_features_t* features);


#endif /* ___ORB_GAP_H__ */
