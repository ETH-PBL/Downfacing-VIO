/* PMSIS includes */
#include "pmsis.h"
#include "bsp/bsp.h"
#include "gaplib/ImgIO.h"
#include "feature_handling/orb.h"
#include "feature_handling/orb_gap.h"
#include "feature_handling/bf_matcher.h"
#include "rigid_body_motion/rigid_body_motion.h"
#include "extended_kalman_filter/of_imu_ekf.h"
#include "imu_measurements.h"
#include "start_stop_indices.h"

#define SINGLE_CORE 1

#define IMG_WIDTH 160
#define IMG_HEIGHT 120
#define IMG_SIZE (IMG_WIDTH*IMG_HEIGHT)

#define MAX_KEYPOINTS 512
#define MAX_FLOW 32

PI_L2 uint8_t* img_buff[2];

uint8_t* image_buffer;
orb_features_t* kpts_descs0;
orb_features_t* kpts_descs1;
feature_match_t* matches;

typedef struct cluster_arguments
{
    pi_device_t* cluster_device;
    uint8_t* img_l2_buffer0;
    imu_measurement_t* current_imu_meas;
    ekf_state_t* state;
    uint8_t feature_extraction_only;
} cluster_arguments_t;

/**
 * @brief Initializes the required data structures.
 *
 * This function sets up and initializes all necessary data structures
 * needed for ORB and feature matching.
 *
 * @param arg Pointer to the GAP9 cluster device structure (as void pointer).
 */
void init_data_structures(void *arg)
{
    pi_device_t* cluster_device = (pi_device_t*) arg;
    /* Memory allocation */
    image_buffer = (uint8_t*) pi_cl_l1_malloc(cluster_device, (uint32_t) (IMG_SIZE));
    kpts_descs0 = pi_cl_l1_malloc(cluster_device, sizeof(orb_features_t));
    kpts_descs0->kpts = pi_cl_l1_malloc(cluster_device, (MAX_KEYPOINTS*sizeof(point2D_u16_t)));
    kpts_descs0->descs = pi_cl_l1_malloc(cluster_device, (MAX_KEYPOINTS*sizeof(orb_descriptor_t)));
    kpts_descs0->kpt_counter = 0;
    kpts_descs0->kpt_capacity = MAX_KEYPOINTS;
    kpts_descs1 = pi_cl_l1_malloc(cluster_device, sizeof(orb_features_t));
    kpts_descs1->kpts = pi_cl_l1_malloc(cluster_device, (MAX_KEYPOINTS*sizeof(point2D_u16_t)));
    kpts_descs1->descs = pi_cl_l1_malloc(cluster_device, (MAX_KEYPOINTS*sizeof(orb_descriptor_t)));
    kpts_descs1->kpt_counter = 0;
    kpts_descs1->kpt_capacity = MAX_KEYPOINTS;
    matches = pi_cl_l1_malloc(cluster_device, (sizeof(feature_match_t)*MAX_KEYPOINTS));
    initialize_orb_storage_gap(cluster_device, IMG_WIDTH, IMG_HEIGHT);
}

/**
 * @brief Delegate function to be executed on cluster core 0.
 *
 * Cluster main entry, executed by core 0.
 *
 * @param arg Pointer to the cluster arguments (cluster_arguments_t as void pointer).
 */
void cluster_delegate(void *arg)
{
    cluster_arguments_t* arguments = (cluster_arguments_t*) arg;
    orb_features_t* prev_features;
    orb_features_t* curr_features;

    if(kpts_descs0->kpt_counter == 0)
    {
        prev_features = kpts_descs1;
        curr_features = kpts_descs0;
    }
    else
    {
        prev_features = kpts_descs0;
        curr_features = kpts_descs1;
    }

    /* Init ORB */
    pi_cl_dma_cmd_t cmd;
    /* DMA Copy Frame 0 */
    pi_cl_dma_cmd((uint32_t) arguments->img_l2_buffer0, (uint32_t) image_buffer, IMG_SIZE, PI_CL_DMA_DIR_EXT2LOC, &cmd);
    /* Wait for DMA transfer to finish. */
    pi_cl_dma_wait(&cmd);
    /* Run ORB on Frame 0 */
    image_data_t img = {IMG_WIDTH, IMG_HEIGHT, image_buffer};
    if(SINGLE_CORE)
    {
        orb_detect_and_compute_single_core(&img, curr_features);
    }
    else
    {
        orb_detect_and_compute_multi_core(&img, curr_features);
    }
    if(arguments->feature_extraction_only)
    {
        prev_features->kpt_counter = 0;
        return;
    }

    uint16_t match_counter;
    if(SINGLE_CORE)
    {
        match_counter = bf_match_two_way_max_flow(matches,prev_features,curr_features,10,20);
    }
    else
    {
        match_counter = bf_match_two_way_max_flow_multicore(matches,prev_features,curr_features,10,20);
    }

    rigid_body_motion_t motion;
    robust_rigid_body_motion_estimation(prev_features->kpts,curr_features->kpts,matches,&motion,match_counter,IMG_WIDTH,IMG_HEIGHT,MAX_FLOW);

    ekf_iteration(arguments->state, arguments->current_imu_meas, &motion);
    prev_features->kpt_counter = 0;
}

/**
 * @brief Fetches an image from the host system and stores it in the provided buffer.
 *
 * This function retrieves the image specified by the given image name from the host
 * and writes its data into the memory pointed to by Input_1.
 *
 * @param ImageName A pointer to a null-terminated string containing the name of the image to fetch.
 * @param Input_1 A pointer to the buffer where the fetched image data will be stored.
 */
void fetch_image_from_host(char* ImageName, uint8_t* Input_1)
{
    ReadImageFromFile(ImageName, IMG_WIDTH, IMG_HEIGHT, 1, Input_1, IMG_SIZE*1*sizeof(char), IMGIO_OUTPUT_CHAR, 0);
}

/* Program Entry. */
int main(void)
{
    printf("\n\n\t *** ORB Descriptor and Detector ***\n\n");
    printf("Entering main controller\n");

    pi_freq_set(PI_FREQ_DOMAIN_FC, 370*1000*1000);
    pi_freq_set(PI_FREQ_DOMAIN_CL, 370*1000*1000);

    img_buff[0] = pi_l2_malloc(IMG_SIZE);

    uint32_t core_id = pi_core_id(), cluster_id = pi_cluster_id();
    pi_device_t* cluster_dev;
    if(pi_open(PI_CORE_CLUSTER, &cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-1);
    }

    ekf_state_t ekf_state = {0};
    ekf_state.scalar = 0.0017f;  // Initial guess for scalar 11px ~= 2.5 cm (~2.3mm/px)
    ekf_state.v[1] = -0.3f; // Initial guess of velocity in y direction

    /* Prepare cluster task and send it to cluster. */
    struct pi_cluster_task cl_task;
    pi_cluster_send_task_to_cl(cluster_dev, pi_cluster_task(&cl_task, init_data_structures, cluster_dev));

    printf("Base Path: %s\n",base_path);
    char img_path[64];
    snprintf(img_path, sizeof(img_path), base_path, start_index-1);
    fetch_image_from_host(img_path, img_buff[0]);
    cluster_arguments_t arguments = {cluster_dev, img_buff[0], NULL, NULL, 1};
    pi_cluster_send_task_to_cl(cluster_dev, pi_cluster_task(&cl_task, cluster_delegate, &arguments));

    for(uint16_t current_idx = start_index; current_idx < stop_index; current_idx+= 1)
    {
        printf("\n***** Index = %d *****\n", current_idx);
        snprintf(img_path, sizeof(img_path), base_path, current_idx);
        fetch_image_from_host(img_path, img_buff[0]);

        imu_measurement_t* current_imu_meas = imu_measurements+current_idx;
        cluster_arguments_t arguments = {cluster_dev, img_buff[0], current_imu_meas, &ekf_state, 0};
        pi_cluster_send_task_to_cl(cluster_dev, pi_cluster_task(&cl_task, cluster_delegate, &arguments));
    }

    pi_cluster_close(cluster_dev);
    printf("Bye !\n");
    return 0;
}
