/* PMSIS includes */
#include "pmsis.h"
#include "bsp/bsp.h"
#include "gaplib/ImgIO.h"

#include "img_feature_definitions/img_feature_definitions.h"
#include "px4flow/flow.h"
#include "px4flow/motion_histogram.h"
#include "rigid_body_motion/rigid_body_motion.h"
#include "extended_kalman_filter/of_imu_ekf.h"

#include "imu_measurements.h"
#include "start_stop_indices.h"

#define IMG_WIDTH 160
#define IMG_HEIGHT 120
#define IMG_SIZE (IMG_WIDTH*IMG_HEIGHT)

#define MAX_FLOW 32

#define SINGLE_CORE 1
#define ORIGINAL_PIPELINE 0

PI_L2 uint8_t* img_buff[2];

uint8_t* image_buffer;
uint8_t* image_buffer_prev;
point2D_u16_t* keypoints0;
point2D_u16_t* keypoints1;
feature_match_t* matches;
unsigned_coordinates_t* points_of_interest;
uint8_t num_points;

typedef struct cluster_arguments
{
    pi_device_t* cluster_device;
    uint32_t img_l2_buffer0;
    uint32_t img_l2_buffer1;
    imu_measurement_t* current_imu_meas;
    ekf_state_t* state;
} cluster_arguments_t;

/* Task executed by cluster cores. */
void cluster_flow(void *arg)
{
    flow(arg, 8);
}

void single_core_flow(void *arg)
{
    flow(arg, 1);
}

/**
 * @brief Initializes the required data structures.
 *
 * This function sets up and initializes all necessary data structures
 * needed for parallelized PX4Flow and feature matching.
 *
 * @param arg Pointer to the GAP9 cluster device structure (as void pointer).
 */
void init_data_structures(void *arg)
{
    pi_device_t* cluster_device = (pi_device_t*) arg;
    /* Memory allocation */
    image_buffer = (uint8_t*) pi_cl_l1_malloc(cluster_device, (uint32_t) (IMG_SIZE));
    image_buffer_prev = (uint8_t*) pi_cl_l1_malloc(cluster_device, (uint32_t) (IMG_SIZE));
    keypoints0 = pi_cl_l1_malloc(cluster_device, ((NUM_BLOCKS*NUM_BLOCKS)*4*2));
    keypoints1 = pi_cl_l1_malloc(cluster_device, ((NUM_BLOCKS*NUM_BLOCKS)*4*2));
    matches = pi_cl_l1_malloc(cluster_device, (sizeof(feature_match_t)*(NUM_BLOCKS*NUM_BLOCKS)));
    points_of_interest = pi_cl_l1_malloc(cluster_device, (sizeof(unsigned_coordinates_t)*(NUM_BLOCKS*NUM_BLOCKS)));

    /* Init Points of Interest */
    uint8_t smallest_pixel = SEARCH_SIZE + 1;
    uint8_t biggest_pixel_width = IMAGE_WIDTH - (SEARCH_SIZE + 1) - TILE_SIZE;
    uint8_t biggest_pixel_height = IMAGE_HEIGHT - (SEARCH_SIZE + 1) - TILE_SIZE;
    uint8_t pixel_step_size_width = ((biggest_pixel_width-smallest_pixel)/ ((uint8_t) NUM_BLOCKS))+1;
    uint8_t pixel_step_size_height = ((biggest_pixel_height-smallest_pixel)/ ((uint8_t) NUM_BLOCKS))+1;

    num_points = 0;
    for(uint16_t y = smallest_pixel; y < biggest_pixel_height; y += pixel_step_size_height)
    {
        for(uint16_t x = smallest_pixel; x < biggest_pixel_width; x += pixel_step_size_width)
        {
            struct unsigned_coordinates coord = {x,y};
            points_of_interest[num_points] =  coord;
            num_points++;
        }
    }
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

    pi_cl_dma_cmd_t cmd;
    /* DMA Copy Frame 0 */
    pi_cl_dma_cmd((uint32_t) arguments->img_l2_buffer0, (uint32_t) image_buffer_prev, IMG_SIZE, PI_CL_DMA_DIR_EXT2LOC, &cmd);
    /* Wait for DMA transfer to finish. */
    pi_cl_dma_wait(&cmd);
    
    /* DMA Copy Frame 1 */
    pi_cl_dma_cmd((uint32_t) arguments->img_l2_buffer1, (uint32_t) image_buffer, IMG_SIZE, PI_CL_DMA_DIR_EXT2LOC, &cmd);
    /* Wait for DMA transfer to finish. */
    pi_cl_dma_wait(&cmd);

    signed_coordinates_t results [NUM_BLOCKS*NUM_BLOCKS];
    struct processing_arguments args = {image_buffer_prev, image_buffer, points_of_interest, num_points,results};

    if(SINGLE_CORE)
    {
        /* Single core */
        single_core_flow((void*)&args);
    }
    else
    {
        /* Multi core */
        pi_cl_team_fork(pi_cl_cluster_nb_cores(), cluster_flow, (void*)&args);
    }

    rigid_body_motion_t motion;
    if(ORIGINAL_PIPELINE)
    {
        estimate_motion_histogram(&motion,results);
    }
    else
    {
        /* Subsequently work with flows multiplied by 2 */
        for(int i = 0; i < num_points; i++)
        {
            point2D_u16_t kpt0 = {points_of_interest[i].x*2, points_of_interest[i].y*2};
            point2D_u16_t kpt1 = {points_of_interest[i].x*2 + results[i].x, points_of_interest[i].y*2 + results[i].y};
            feature_match_t match = {i, i, 1};
            keypoints0[i] = kpt0;
            keypoints1[i] = kpt1;
            matches[i] = match;
        }
        robust_rigid_body_motion_estimation(keypoints0,keypoints1,matches,&motion,num_points,2*IMG_WIDTH,2*IMG_HEIGHT,2*MAX_FLOW);
        motion.flow_x = motion.flow_x/2;
        motion.flow_y = motion.flow_y/2;
    }

    /* TODO: If original pipeline is selected, adapt EKF settings */
    ekf_iteration(arguments->state, arguments->current_imu_meas, &motion);
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
    printf("\n\n\t *** PX4FLOW Flow ***\n\n");
    printf("Entering main controller\n");

    pi_freq_set(PI_FREQ_DOMAIN_FC, 370*1000*1000);
    pi_freq_set(PI_FREQ_DOMAIN_CL, 370*1000*1000);

    img_buff[0] = pi_l2_malloc(IMG_SIZE);
    img_buff[1] = pi_l2_malloc(IMG_SIZE);

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

    uint8_t even = 1;
    for(uint16_t current_idx = start_index; current_idx < stop_index; current_idx+= 1)
    {
        printf("\n***** Index = %d *****\n", current_idx);
        uint8_t* prev_frame;
        uint8_t* curr_frame;
        if(even)
        {
            prev_frame = img_buff[0];
            curr_frame = img_buff[1];
        }
        else
        {
            prev_frame = img_buff[1];
            curr_frame = img_buff[0];
        }
        even = !even;

        snprintf(img_path, sizeof(img_path), base_path, current_idx);
        fetch_image_from_host(img_path, curr_frame);

        imu_measurement_t* current_imu_meas = imu_measurements+current_idx;
        cluster_arguments_t arguments = {cluster_dev, (uint32_t) prev_frame, (uint32_t) curr_frame, current_imu_meas, &ekf_state};
        pi_cluster_send_task_to_cl(cluster_dev, pi_cluster_task(&cl_task, cluster_delegate, &arguments));
    }

    pi_cluster_close(cluster_dev);
    printf("Bye !\n");
    return 0;
}
