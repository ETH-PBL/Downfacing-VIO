#ifndef __MOTION_HISTOGRAM_H__
#define __MOTION_HISTOGRAM_H__

#include "pmsis.h"
#include "flow.h"
#include "rigid_body_motion/rigid_body_motion.h"

/**
 * @brief Estimates the planar motion from pixel displacements using the histogram method of PX4Flow.
 *
 * This function uses the pixel displacements stored in the displacements structure to create a motion histogram
 * following the method presented in the PX4Flow paper. The resulting translation is stored in the motion structure.
 * (https://people.inf.ethz.ch/~pomarc/pubs/HoneggerICRA13.pdf)
 *
 * @param motion Pointer to a rigid_body_motion_t structure where the computed motion estimate will be stored.
 * @param displacement Pointer to a signed_coordinates_t structure containing flow estimates multiple pixel locations.
 */
void estimate_motion_histogram(rigid_body_motion_t* motion, signed_coordinates_t* displacements);

#endif /* __MOTION_HISTOGRAM_H__ */
