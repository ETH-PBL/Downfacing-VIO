#ifndef __FEATURE_TRACKER_ORB_H__
#define __FEATURE_TRACKER_ORB_H__

#include "pmsis.h"
#include "img_feature_definitions/img_feature_definitions.h"

#define PATCH_RADIUS 15 /* The ORB descriptor is 31x31 pixels. */
#define FAST_THRESHOLD 20
#define HARRIS_BLOCK_SIZE 7

typedef uint32_t orb_descriptor_t[8];

typedef struct orb_features
{
    point2D_u16_t* kpts;
    orb_descriptor_t* descs;
    uint16_t kpt_counter;
    uint16_t kpt_capacity;
} orb_features_t;

/**
 * @brief Initializes the ORB (Oriented FAST and Rotated BRIEF) feature detector.
 *
 * This function sets up the necessary offset tables for the FAST and Harris corner detectors
 * used in the ORB algorithm, based on the provided image width.
 *
 * @param fast_offsets_storage Pointer to the storage array for FAST detector offsets.
 * @param harris_offsets_storage Pointer to the storage array for Harris detector offsets.
 * @param image_width The width of the input image, used to calculate pixel offsets.
 */
void initialize_orb(int16_t* fast_offsets_storage, int16_t* harris_offsets_storage, uint16_t image_width);

/**
 * @brief Runs the ORB (Oriented FAST and Rotated BRIEF) feature detector on a specified region of the image.
 *
 * This function detects ORB features within the rows specified by start_row and end_row in the given image.
 * The detected features are stored in the provided features structure.
 *
 * @param img             Pointer to the image data structure to process.
 * @param features        Pointer to the structure where detected ORB features will be stored.
 * @param start_row       The starting row (inclusive) of the image region to process.
 * @param end_row         The ending row (exclusive) of the image region to process.
 * @param fast_threshold  Threshold value for the FAST corner detector used in ORB.
 */
void run_orb_detector(image_data_t* img, orb_features_t* features, uint16_t start_row, uint16_t end_row, uint8_t fast_threshold);

/**
 * @brief Applies a Gaussian blur filter to a specified range of rows in the input image.
 *
 * This function processes the input image by applying a Gaussian blur to the rows
 * from start_row (inclusive) to end_row (exclusive), and writes the result to the output image.
 *
 * @param img_in Pointer to the input image data structure.
 * @param img_out Pointer to the output image data structure where the blurred image will be stored.
 * @param start_row The starting row index (inclusive) for applying the Gaussian blur.
 * @param end_row The ending row index (exclusive) for applying the Gaussian blur.
 */
void apply_gaussian_blurr(image_data_t* img_in, image_data_t* img_out, uint16_t start_row, uint16_t end_row);

/**
 * @brief Calculates the ORB (Oriented FAST and Rotated BRIEF) descriptor for detected features in a blurred image.
 *
 * This function processes the provided blurred image and computes the ORB descriptors
 * for the given set of feature points. The resulting descriptors are stored in the
 * provided features structure.
 *
 * @param blurred_img Pointer to the input image data that has been pre-processed (blurred).
 * @param features Pointer to the structure containing detected feature points and where the computed descriptors will be stored.
 */
void calculate_orb_decriptor(image_data_t* blurred_img, orb_features_t* features);

#endif /* __FEATURE_TRACKER_ORB_H__ */
