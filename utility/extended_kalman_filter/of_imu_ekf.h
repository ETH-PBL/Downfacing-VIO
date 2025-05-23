#ifndef __OF_IMU_EKF__H__
#define __OF_IMU_EKF__H__

#include "pmsis.h"
#include "rigid_body_motion/rigid_body_motion.h"

// State vector structure
typedef struct
{
    float p[3];    // Position (x, y, z)
    float v[3];    // Velocity (x, y, z)
    float a[3];    // Acceleration (x, y, z)
    float yaw;     // Orientation yaw
    float scalar;  // Scalar m/px
} ekf_state_t;

typedef struct imu_measurement
{
	float x_lin_acc;
	float y_lin_acc;
	float z_lin_acc;
	float x_rot_vel;
	float y_rot_vel;
	float z_rot_vel;
	int16_t tof_idx;
} imu_measurement_t;

/**
 * @brief Performs one iteration of the Extended Kalman Filter (EKF) using the provided IMU measurement and
 * motion estimation.
 *
 * @param state Pointer to the EKF state structure to be updated.
 * @param imu_measurement Pointer to the structure containing the latest IMU measurement data.
 * @param motion_estimation Pointer to the structure containing the latest rigid body motion estimate.
 */
void ekf_iteration(ekf_state_t* state,
                   imu_measurement_t* imu_measurement,
				   rigid_body_motion_t* motion_estimation);

#endif /* __OF_IMU_EKF__H__ */
