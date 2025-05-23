/* Autotiler includes. */
#include "superpoint.h"
#include "superpointKernels.h"
#include "gaplib/fs_switch.h"
#include "gaplib/ImgIO.h"

#include "img_feature_definitions/img_feature_definitions.h"
#include "superpoint/postprocessing.h"
#include "extended_kalman_filter/of_imu_ekf.h"

#include "imu_measurements.h"
#include "start_stop_indices.h"

#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif

#define IMG_WIDTH 160
#define IMG_HEIGHT 120
#define IMG_SIZE (IMG_WIDTH*IMG_HEIGHT)

#define MAX_FLOW 32

AT_HYPERFLASH_EXT_ADDR_TYPE superpoint_L3_Flash = 0;

/* Outputs */
/* (Pixel locations + dust bin) x H/8 x W/8 */
L2_MEM unsigned char Output_Kpts[19500];
/* Desc x H/8 x W/8 */
L2_MEM unsigned char Output_Descs0[76800];
L2_MEM unsigned char Output_Descs1[76800];

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

static void cluster0()
{
    printf("Starting Inference: Write to Desc 0 Buffer\n");
    superpointCNN(Output_Kpts,Output_Descs0);
    printf("Runner completed\n");

}

static void cluster1()
{
    printf("Starting Inference: Write to Desc 1 Buffer\n");
    superpointCNN(Output_Kpts,Output_Descs1);
    printf("Runner completed\n");

}

int main(void)
{
    printf("\n\n\t *** NNTOOL superpoint ***\n\n");
    printf("Entering main controller\n");

    /* Configure And open cluster. */
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
    cl_conf.cc_stack_size = STACK_SIZE;

    cl_conf.id = 0; /* Set cluster ID. */
                    // Enable the special icache for the master core
    cl_conf.icache_conf = PI_CLUSTER_MASTER_CORE_ICACHE_ENABLE |
                    // Enable the prefetch for all the cores, it's a 9bits mask (from bit 2 to bit 10), each bit correspond to 1 core
                    PI_CLUSTER_ICACHE_PREFETCH_ENABLE |
                    // Enable the icache for all the cores
                    PI_CLUSTER_ICACHE_ENABLE;

    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-4);
    }

    /* Frequency Settings: defined in the Makefile */
    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    int cur_pe_freq = pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE*1000*1000);
    if (cur_fc_freq == -1 || cur_cl_freq == -1 || cur_pe_freq == -1)
    {
        printf("Error changing frequency !\nTest failed...\n");
        pmsis_exit(-4);
    }
	printf("FC Frequency as %d Hz, CL Frequency = %d Hz, PERIIPH Frequency = %d Hz\n",
            pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), pi_freq_get(PI_FREQ_DOMAIN_PERIPH));


    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    printf("Constructor\n");
    int ConstructorErr = superpointCNN_Construct();
    if (ConstructorErr)
    {
        printf("Graph constructor exited with error: %d\n(check the generated file superpointKernels.c to see which memory have failed to be allocated)\n", ConstructorErr);
        pmsis_exit(-6);
    }

    ekf_state_t ekf_state = {0};
    ekf_state.scalar = 0.0017f;  // Initial guess for scalar 11px ~= 2.5 cm (~2.3mm/px)
    ekf_state.v[1] = -0.3f; // Initial guess of velocity in y direction

    /* Max number of keypoints is 300 by design */
    point2D_u16_t* kpts0 = pi_l2_malloc((300*4*2));
    point2D_u16_t* kpts1 = pi_l2_malloc((300*4*2));
    feature_match_t* matches = pi_l2_malloc((300*sizeof(feature_match_t)));
    inplace_matcher_init();

    printf("Base Path: %s\n",base_path);
    char img_path[64];
    snprintf(img_path, sizeof(img_path), base_path, start_index-1);
    fetch_image_from_host(img_path, Input_1);

    struct pi_cluster_task task0;
    pi_cluster_task(&task0, (void (*)(void *))cluster0, NULL);
    pi_cluster_task_stacks(&task0, NULL, SLAVE_STACK_SIZE);
    printf("Call cluster\n");
    pi_cluster_send_task_to_cl(&cluster_dev, &task0);

    uint16_t prev_keypoint_counter = extract_kpts(kpts0, Output_Kpts, 160, 120, 8);
    uint16_t curr_keypoint_counter = 0;
    printf("Keypoint counter %d \n", prev_keypoint_counter);

    uint8_t even = 1;

    for(uint16_t current_idx = start_index; current_idx < stop_index; current_idx+= 1)
    {
        uint8_t* prev_desc;
        uint8_t* curr_desc;
        point2D_u16_t* prev_kpts;
        point2D_u16_t* curr_kpts;
        snprintf(img_path, sizeof(img_path), base_path, current_idx);
        printf("\n***** Index = %d *****\n", current_idx);

        fetch_image_from_host(img_path, Input_1);

        struct pi_cluster_task task1;

        if(even)
        {
            prev_kpts = kpts0;
            curr_kpts = kpts1;
            prev_desc = Output_Descs0;
            curr_desc = Output_Descs1;
            pi_cluster_task(&task1, (void (*)(void *))cluster1, NULL);
            printf("Even\n");
        }
        else
        {
            prev_kpts = kpts1;
            curr_kpts = kpts0;
            prev_desc = Output_Descs1;
            curr_desc = Output_Descs0;
            pi_cluster_task(&task1, (void (*)(void *))cluster0, NULL);
            printf("Uneven\n");
        }
        even = !even;
        pi_cluster_task_stacks(&task1, NULL, SLAVE_STACK_SIZE);
        printf("Call cluster\n");
        pi_cluster_send_task_to_cl(&cluster_dev, &task1);

        curr_keypoint_counter = extract_kpts(curr_kpts, Output_Kpts, 160, 120, 8);
        printf("Keypoint counter %d \n", curr_keypoint_counter);
        /* Cosine similarity for original SuperPoint is scores = 2-similarities and keep = scores < 0.7 */
        /* Our similiarities are between 0 (least similar) and 255 (most similar), so < 0.7 is equivalent to > 166*/
        uint16_t match_counter = inplace_match_two_way_max_flow(prev_desc, curr_desc, prev_kpts, curr_kpts, matches, prev_keypoint_counter, curr_keypoint_counter, 10, 160, 120, 8, 166);
        rigid_body_motion_t motion;
        robust_rigid_body_motion_estimation(prev_kpts,curr_kpts,matches,&motion,match_counter,IMG_WIDTH,IMG_HEIGHT,MAX_FLOW);
        printf("Flow (x,y,yaw) = [%f %f %f]\n", motion.flow_x, motion.flow_y, motion.rot_z);

        imu_measurement_t* current_imu_meas = imu_measurements+current_idx;
        ekf_iteration(&ekf_state, current_imu_meas, &motion);

        prev_keypoint_counter = curr_keypoint_counter;
    }

    superpointCNN_Destruct();

    printf("Ended\n");
    pmsis_exit(0);
    return 0;
}
