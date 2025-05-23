#include "orb_gap.h"

typedef struct orb_arguments
{
    image_data_t* img;
    orb_features_t* features;
    uint16_t* utility_counters;
} orb_arguments_t;

uint8_t* blurred_image_buffer;

void initialize_orb_storage_gap(pi_device_t* cluster_device, int16_t image_width, int16_t image_height)
{
    printf("0. Initializing ORB\n");
    int16_t* fast_offsets = pi_cl_l1_malloc(cluster_device, 16*2);
    int16_t* harris_offsets = pi_cl_l1_malloc(cluster_device, (HARRIS_BLOCK_SIZE*HARRIS_BLOCK_SIZE*2));
    blurred_image_buffer = (uint8_t*) pi_cl_l1_malloc(cluster_device, (uint32_t) (image_width*image_height));
    initialize_orb(fast_offsets, harris_offsets,image_width);
}

void orb_detect_and_compute_single_core(image_data_t* img, orb_features_t* features)
{
    //printf("1. Detecting Single Core\n");
    pi_perf_reset();
    pi_perf_start();
    run_orb_detector(img,features,PATCH_RADIUS+1,(img->height-PATCH_RADIUS-1),FAST_THRESHOLD);
    pi_perf_stop(); 
    //printf("Number of cycles %d\n",pi_perf_read(PI_PERF_CYCLES));

    //printf("2. Blurr Single Core\n");
    pi_perf_reset();
    pi_perf_start();
    image_data_t blurred_image = {img->width, img->height, blurred_image_buffer};
    apply_gaussian_blurr(img, &blurred_image,0,img->height);
    pi_perf_stop(); 
    //printf("Number of cycles %d\n",pi_perf_read(PI_PERF_CYCLES));

    //printf("3. Describing Single Core\n");
    pi_perf_reset();
    pi_perf_start();
    calculate_orb_decriptor(&blurred_image,features);
    pi_perf_stop(); 
    //printf("Number of cycles %d\n",pi_perf_read(PI_PERF_CYCLES));
}

void orb_detect_subset(void* args)
{
    orb_arguments_t* orb_args = (orb_arguments_t*) args;
    uint16_t core_id = pi_core_id();
	uint16_t nb_cores = pi_cl_cluster_nb_cores();
    uint16_t aux_patch_radius = 16; /* Multiple of 8 */
    uint16_t total_rows = orb_args->img->height - 2*aux_patch_radius;
    uint16_t start_row = core_id * (total_rows / nb_cores)+aux_patch_radius;
	uint16_t end_row = (core_id + 1) * (total_rows / nb_cores)+aux_patch_radius;
    uint16_t keypoint_offset = orb_args->features->kpt_capacity/nb_cores*core_id;
    orb_features_t temp_feature;
    temp_feature.kpts = orb_args->features->kpts+keypoint_offset;
    temp_feature.kpt_capacity = orb_args->features->kpt_capacity/nb_cores;
    run_orb_detector(orb_args->img,&temp_feature,start_row,end_row,FAST_THRESHOLD);
    orb_args->utility_counters[core_id] = temp_feature.kpt_counter;
}

void orb_blurr_subset(void* args)
{
    orb_arguments_t* orb_args = (orb_arguments_t*) args;
    uint16_t core_id = pi_core_id();
	uint16_t nb_cores = pi_cl_cluster_nb_cores();
    uint16_t total_rows = orb_args->img->height;
    uint16_t start_row = core_id * (total_rows / nb_cores);
	uint16_t end_row = (core_id + 1) * (total_rows / nb_cores);
    image_data_t blurred_image = {orb_args->img->width, orb_args->img->height, blurred_image_buffer};
    apply_gaussian_blurr(orb_args->img, &blurred_image,start_row,end_row);
}

void orb_compute_subset(void* args)
{
    orb_arguments_t* orb_args = (orb_arguments_t*) args;
    uint16_t core_id = pi_core_id();
	uint16_t nb_cores = pi_cl_cluster_nb_cores();
    uint16_t nb_features = orb_args->utility_counters[0];
    uint16_t slice_size;
    if(nb_features % nb_cores == 0)
    {
        slice_size = nb_features/nb_cores;
    }
    else
    {
        slice_size = nb_features/nb_cores+1;
    }
    uint32_t offset = slice_size*core_id;
    if (core_id == (nb_cores-1))
    {
        slice_size = nb_features-(nb_cores-1)*slice_size;
    }
    image_data_t blurred_image = {orb_args->img->width, orb_args->img->height, blurred_image_buffer};
    orb_features_t temp_feature;
    temp_feature.kpts = orb_args->features->kpts + offset;
    temp_feature.descs = orb_args->features->descs + offset;
    temp_feature.kpt_counter = slice_size;
    calculate_orb_decriptor(&blurred_image,&temp_feature);
}

void reorganize_keypoint_memory(orb_arguments_t* orb_arg)
{
    uint8_t nb_cores = pi_cl_cluster_nb_cores();
    uint16_t feature_counter = orb_arg->utility_counters[0];
    for(uint8_t index = 1; index < nb_cores; ++index)
    {
        uint16_t nb_feature_pool = orb_arg->utility_counters[index];
        uint16_t keypoint_offset = orb_arg->features->kpt_capacity/nb_cores*index;
        for (uint8_t feature_nr = 0; feature_nr < nb_feature_pool; ++feature_nr)
        {
            orb_arg->features->kpts[feature_counter] = orb_arg->features->kpts[keypoint_offset+feature_nr];
            ++feature_counter;
        }
    }
    orb_arg->utility_counters[0]=feature_counter;
    orb_arg->features->kpt_counter = feature_counter;
}

void orb_detect_and_compute_multi_core(image_data_t* img, orb_features_t* features)
{
    uint16_t utility_counters[8];
    orb_arguments_t orb_args = {img, features, utility_counters};
    printf("1. Detect Parallelized\n");
    pi_perf_reset();
	pi_perf_start();
    pi_cl_team_fork(pi_cl_cluster_nb_cores(), orb_detect_subset, &orb_args);
    pi_perf_stop(); 
    printf("Number of cycles %d\n",pi_perf_read(PI_PERF_CYCLES));
    reorganize_keypoint_memory(&orb_args);
    
    printf("2. Blurr Parallelized\n");
    pi_perf_reset();
	pi_perf_start();
    pi_cl_team_fork(pi_cl_cluster_nb_cores(), orb_blurr_subset, &orb_args);
    pi_perf_stop(); 
    printf("Number of cycles %d\n",pi_perf_read(PI_PERF_CYCLES));
    
    printf("3. Describe Parallelized\n");
    pi_perf_reset();
	pi_perf_start();
    pi_cl_team_fork(pi_cl_cluster_nb_cores(), orb_compute_subset, &orb_args);
    pi_perf_stop(); 
    printf("Number of cycles %d\n",pi_perf_read(PI_PERF_CYCLES));
    printf("\nNumber of keypoints %d\n",features->kpt_counter);
}
