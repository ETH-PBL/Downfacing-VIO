#include "of_imu_ekf.h"

#include "math.h"

#define Abs(a)          (((int)(a)<0)?(-(a)):(a))
// Constants
#define DT 0.01 // EKF time step
#define K_FLOW 0.3 // Kalman Gain for linear Flow
#define K_ACCEL 0.9 // Kalman Gain for Acceleration
#define K_SCALAR 0.01 // Kalman Gain for Scale Factor
#define IMU_WEIGHT 0.7 // rotation weight balancing IMU vs rotation of rigid body motion
#define X_GYR_BIAS 0.02913
#define Y_GYR_BIAS -0.014914
#define Z_GYR_BIAS -0.002797

// IMU Data (accelerometer and gyroscope)
typedef struct 
{
    float accel[3]; // Linear acceleration in 3D
    float gyro[3];  // Rotational velocity in 3D
} imu_data_t;

// Rotational offset between camera and IMU obtained through calibration
static float R_cam_imu[3][3] = {{0.011066192829442545, -0.9999378844759764, -0.001329122255283725},
                                {0.9997516704710715, 0.011089824322592534, -0.0193290762012916},
                                {0.019342615297908816, -0.0011148929105217024, 0.9998122925065656}};

// Quaternion multiplication
void quaternion_multiply(float q1[4], float q2[4], float result[4]) 
{
    result[0] = q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1];
    result[1] = q1[3] * q2[1] + q1[1] * q2[3] + q1[2] * q2[0] - q1[0] * q2[2];
    result[2] = q1[3] * q2[2] + q1[2] * q2[3] + q1[0] * q2[1] - q1[1] * q2[0];
    result[3] = q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2];
    float norm = sqrtf(result[0] * result[0] + result[1] * result[1] + result[2] * result[2] + result[3] * result[3]);
    result[0] /= norm;
    result[1] /= norm;
    result[2] /= norm;
    result[3] /= norm;
}

void transform_imu_to_cam_frame(imu_measurement_t* sensor_meas, imu_data_t* transforemd_data)
{
    // Matrix multiplication of R_cam_imu with accel and gyro measurements
    transforemd_data->accel[0] = R_cam_imu[0][0] * sensor_meas->x_lin_acc + R_cam_imu[0][1] * sensor_meas->y_lin_acc + R_cam_imu[0][2] * sensor_meas->z_lin_acc;
    transforemd_data->accel[1] = R_cam_imu[1][0] * sensor_meas->x_lin_acc + R_cam_imu[1][1] * sensor_meas->y_lin_acc + R_cam_imu[1][2] * sensor_meas->z_lin_acc;
    transforemd_data->accel[2] = R_cam_imu[2][0] * sensor_meas->x_lin_acc + R_cam_imu[2][1] * sensor_meas->y_lin_acc + R_cam_imu[2][2] * sensor_meas->z_lin_acc;
    transforemd_data->gyro[0] = R_cam_imu[0][0] * sensor_meas->x_rot_vel + R_cam_imu[0][1] * sensor_meas->y_rot_vel + R_cam_imu[0][2] * sensor_meas->z_rot_vel;
    transforemd_data->gyro[1] = R_cam_imu[1][0] * sensor_meas->x_rot_vel + R_cam_imu[1][1] * sensor_meas->y_rot_vel + R_cam_imu[1][2] * sensor_meas->z_rot_vel;
    transforemd_data->gyro[2] = R_cam_imu[2][0] * sensor_meas->x_rot_vel + R_cam_imu[2][1] * sensor_meas->y_rot_vel + R_cam_imu[2][2] * sensor_meas->z_rot_vel;
}

void euler_to_quaternion(float euler[3], float quaternion[4])
{
    float cr = cosf(euler[0] * 0.5);
    float sr = sinf(euler[0] * 0.5);
    float cp = cosf(euler[1] * 0.5);
    float sp = sinf(euler[1] * 0.5);
    float cy = cosf(euler[2] * 0.5);
    float sy = sinf(euler[2] * 0.5);
    quaternion[0] = sr * cp * cy - cr * sp * sy; // x
    quaternion[1] = cr * sp * cy + sr * cp * sy; // y
    quaternion[2] = cr * cp * sy - sr * sp * cy; // z
    quaternion[3] = cr * cp * cy + sr * sp * sy; // w
}

// Quaternion multiplication for transforming euler vectors
void transform_absolute(float q[4], float v[3], float result[3]) {
    // Quaternion conjugate
    float q_conjugate[4] = {q[0], -q[1], -q[2], -q[3]};
    
    // Perform the quaternion * vector * quaternion_conjugate operation
    float qv[4] = {0, v[0], v[1], v[2]};

    // First quaternion multiplication: q * qv
    float temp[4] = {
        q[0] * qv[0] - q[1] * qv[1] - q[2] * qv[2] - q[3] * qv[3],
        q[0] * qv[1] + q[1] * qv[0] + q[2] * qv[3] - q[3] * qv[2],
        q[0] * qv[2] - q[1] * qv[3] + q[2] * qv[0] + q[3] * qv[1],
        q[0] * qv[3] + q[1] * qv[2] - q[2] * qv[1] + q[3] * qv[0]
    };

    // Second quaternion multiplication: temp * q_conjugate
    float transformed[4] = {
        temp[0] * q_conjugate[0] - temp[1] * q_conjugate[1] - temp[2] * q_conjugate[2] - temp[3] * q_conjugate[3],
        temp[0] * q_conjugate[1] + temp[1] * q_conjugate[0] + temp[2] * q_conjugate[3] - temp[3] * q_conjugate[2],
        temp[0] * q_conjugate[2] - temp[1] * q_conjugate[3] + temp[2] * q_conjugate[0] + temp[3] * q_conjugate[1],
        temp[0] * q_conjugate[3] + temp[1] * q_conjugate[2] - temp[2] * q_conjugate[1] + temp[3] * q_conjugate[0]
    };

    // Store the result in the result array
    result[0] = transformed[1];
    result[1] = transformed[2];
    result[2] = transformed[3];
}

// Prediction step based on IMU data
void ekf_predict(ekf_state_t *state, imu_data_t *imu) 
{
    // Update velocity and position based on acceleration
    float dp[3];
    for (int i = 0; i < 3; i++) {
        // Assume constant acceleration
        // Estimate increment in position
        dp[i] = state->v[i] * DT + state->a[i] * DT * DT / 2;    // Integrate velocity to get position
        state->v[i] += imu->accel[i] * DT;  // Integrate acceleration to get velocity
    }
    // Rotate position incerement according to orientation of device
    state->p[0] += cosf(state->yaw)*dp[0] - sinf(state->yaw)*dp[1];
    state->p[1] += sinf(state->yaw)*dp[0] + cosf(state->yaw)*dp[1];
    state->p[2] = 0;

    // Simplified rotation model, assume constant orientation for prediction step
}

// Update step based on optical flow measurement
void ekf_update(ekf_state_t *state, rigid_body_motion_t *flow, imu_data_t *imu) 
{
    // Measurement residual of the velocity (negative flow)
    float y_vx = -(flow->flow_x * state->scalar) / DT - state->v[0];
    float y_vy = -(flow->flow_y * state->scalar) / DT - state->v[1];

    // Update orientation. Fully trust observations according to the IMU_WEIGHT.
    float imu_weight = IMU_WEIGHT;
    float d_yaw = -flow->rot_z * (1-imu_weight) + imu->gyro[2] * DT * imu_weight;
    state->yaw += d_yaw; 

    // Simplified update without covariance matrix (but scalar value K_ACCEL)
    float y_a[3] = {0};
    for (int i = 0; i < 3; i++) {
        y_a[i] = imu->accel[i] - state->a[i];
        state->a[i] += (K_ACCEL * y_a[i]);
    }

    // Simplified update without covariance matrix (but scalar value K_FLOW)
    state->v[0] += (K_FLOW * y_vx);
    state->v[1] += (K_FLOW * y_vy);
}

void ekf_iteration(ekf_state_t* state, imu_measurement_t* imu_measurement, rigid_body_motion_t* motion_estimation)
{
    imu_data_t imu_data;
    transform_imu_to_cam_frame(imu_measurement, &imu_data);
    printf("Accel \tx %f \ty %f \tz %f \n Gyro \tx %f \ty %f \tz %f \n", 
           imu_data.accel[0], imu_data.accel[1], imu_data.accel[2],
           imu_data.gyro[0], imu_data.gyro[1], imu_data.gyro[2]);
    printf("Flow x %f, flow y %f\n", motion_estimation->flow_x, motion_estimation->flow_y);

    ekf_predict(state, &imu_data);
    ekf_update(state, motion_estimation, &imu_data);

    // Output the updated state
    printf("p = [%f, %f, %f]\n", state->p[0], state->p[1], state->p[2]);
    printf("v = [%f, %f, %f]\n", state->v[0], state->v[1], state->v[2]);
    printf("a = [%f, %f, %f]\n", state->a[0], state->a[1], state->a[2]);
    printf("y = [%f]\n", state->yaw);
    printf("s = %f\n", state->scalar);
}
