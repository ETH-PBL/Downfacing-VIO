#include "rigid_body_motion.h"
#include "math.h"

#define Abs(a)          (((int)(a)<0)?(-(a)):(a))
#define INITIAL_OUTLIER_TOLERANCE 5
#define ITERATIVE_OUTLIER_TOLERANCE 1.5f

void svd2x2(float* A, float* U, float* S, float* V) {
    float a = A[0];
    float b = A[1];
    float c = A[2];
    float d = A[3];

    // Compute the symmetric matrix B = A^T * A
    float B[4];
    B[0] = a * a + c * c;
    B[1] = a * b + c * d;
    B[2] = B[1]; // since B is symmetric
    B[3] = b * b + d * d;

    // Compute the trace and determinant of B
    float trace = B[0] + B[3];
    float det = B[0] * B[3] - B[1] * B[2];

    // Compute the eigenvalues of B (which are the squared singular values)
    float eigenvalue1 = (trace + sqrtf((float)(trace * trace - 4 * det))) / 2.0f;
    float eigenvalue2 = (trace - sqrtf((float)(trace * trace - 4 * det))) / 2.0f;

    // The singular values are the square roots of the eigenvalues
    S[0] = sqrtf(eigenvalue1);
    S[1] = sqrtf(eigenvalue2);

    // Compute V (eigenvectors of B)
    if (B[1] != 0) {
        float v1[2] = {(float)B[1], eigenvalue1 - B[0]};
        float v2[2] = {(float)B[1], eigenvalue2 - B[0]};
        float norm1 = sqrtf(v1[0] * v1[0] + v1[1] * v1[1]);
        float norm2 = sqrtf(v2[0] * v2[0] + v2[1] * v2[1]);
        V[0] = v1[0] / norm1;
        V[2] = v1[1] / norm1;
        V[1] = v2[0] / norm2;
        V[3] = v2[1] / norm2;
    } else {
        V[0] = 1.0f;
        V[2] = 0.0f;
        V[1] = 0.0f;
        V[3] = 1.0f;
    }

    // Compute U = A * V * Σ^(-1)
    U[0] = (a * V[0] + b * V[2]) / S[0];
    U[1] = (a * V[1] + b * V[3]) / S[1];
    U[2] = (c * V[0] + d * V[2]) / S[0];
    U[3] = (c * V[1] + d * V[3]) / S[1];
}

void rigid_body_motion_estimation(point2D_u16_t* keypoints0,
                                  point2D_u16_t* keypoints1,
                                  feature_match_t* matches,
                                  rigid_body_motion_t* motion,
                                  uint8_t* inlier_mask,
                                  uint16_t match_counter,
                                  uint16_t img_width,
                                  uint16_t img_height)
{
    /* Calculate the centroids */
    float cx0 = 0;
    float cy0 = 0;
    float cx1 = 0;
    float cy1 = 0;
    uint16_t inlier_count = 0;
    for(uint16_t index = 0; index < match_counter; index++)
    {
        feature_match_t current_match = matches[index];
        point2D_u16_t p0 = keypoints0[current_match.feat_idx0];
        point2D_u16_t p1 = keypoints1[current_match.feat_idx1];
        if (inlier_mask[index])
        {
            cx0 += p0.x;
            cy0 += p0.y;
            cx1 += p1.x;
            cy1 += p1.y;
            inlier_count++;
        }
    }
    cx0 = cx0/inlier_count;
    cy0 = cy0/inlier_count;
    cx1 = cx1/inlier_count;
    cy1 = cy1/inlier_count;

    /* Catch the case of no inliers  */
    if(inlier_count == 0)
    {
        motion->flow_x = 0.0f;
        motion->flow_y = 0.0f;
        motion->rot_z = 0.0f;
        return;
    }

    /* Compose H matrix */
    float H[4] = {0,0,0,0};
    for(uint16_t index = 0; index < match_counter; index++)
    {
        feature_match_t current_match = matches[index];
        point2D_u16_t p0 = keypoints0[current_match.feat_idx0];
        point2D_u16_t p1 = keypoints1[current_match.feat_idx1];
        if (inlier_mask[index])
        {
            H[0] += ((p0.x - cx0)*(p1.x - cx1));
            H[1] += ((p0.x - cx0)*(p1.y - cy1));
            H[2] += ((p0.y - cy0)*(p1.x - cx1));
            H[3] += ((p0.y - cy0)*(p1.y - cy1));
        }
    }

    float U[4];
    float S[2];
    float V[4];
    /* Run SVD */
    svd2x2(H,U,S,V);

    float rot[4];
    if(S[1] > 0)
    {
        rot[0] = V[0]*U[0]+V[1]*U[1];
        rot[1] = V[0]*U[2]+V[1]*U[3];
        rot[2] = V[2]*U[0]+V[3]*U[1];
        rot[3] = V[2]*U[2]+V[3]*U[3];
        if(rot[0]*rot[3]-rot[1]*rot[2] < 0)
        {
            rot[0] = V[0]*U[0]-V[1]*U[1];
            rot[1] = V[0]*U[2]-V[1]*U[3];
            rot[2] = V[2]*U[0]-V[3]*U[1];
            rot[3] = V[2]*U[2]-V[3]*U[3];
        }
    }
    else
    {
        printf("No full rank in SVD, solving only for linear motion.\n");
        rot[0] = 1.0f;
        rot[1] = 0.0f;
        rot[2] = 0.0f;
        rot[3] = 1.0f;
    }

    uint16_t width_offset = img_width/2;
    uint16_t height_offset = img_height/2;
    float translation[2];
    translation[0] = cx1 - width_offset  - (rot[0]*(cx0 - width_offset) + rot[1]*(cy0 - height_offset));
    translation[1] = cy1 - height_offset - (rot[2]*(cx0 - width_offset) + rot[3]*(cy0 - height_offset));

    motion->flow_x = translation[0];
    motion->flow_y = translation[1];
    motion->rot_z = asinf(rot[2]);

    printf("Inlier count %d\n",inlier_count);
    printf("t0 %f, t1 %f, yaw %f \n",translation[0],translation[1], motion->rot_z);
    return;
}

void robust_rigid_body_motion_estimation(point2D_u16_t* keypoints0,
                                         point2D_u16_t* keypoints1,
                                         feature_match_t* matches,
                                         rigid_body_motion_t* motion,
                                         uint16_t match_counter,
                                         uint16_t img_width,
                                         uint16_t img_height,
                                         uint16_t max_flow)
{
    /* Create histogram */
    uint16_t x_buckets[max_flow*2+1];
    uint16_t y_buckets[max_flow*2+1];
    for(uint16_t index = 0; index < max_flow*2+1; index++)
    {
        x_buckets[index] = 0;
        y_buckets[index] = 0;
    }

    for(uint16_t index = 0; index < match_counter; index++)
    {
        feature_match_t current_match = matches[index];
        point2D_u16_t p0 = keypoints0[current_match.feat_idx0];
        point2D_u16_t p1 = keypoints1[current_match.feat_idx1];
        int16_t dx = p1.x - p0.x;
        int16_t dy = p1.y - p0.y;
        if (Abs(dx) <= max_flow && Abs(dy) <= max_flow)
        {
            x_buckets[dx+max_flow] += 1;
            y_buckets[dy+max_flow] += 1;
        }
    }

    uint16_t x_index = 0;
    uint16_t y_index = 0;
    int16_t max_x = 0;
    int16_t max_y = 0;
    /* Find largest entry in histogram */
    for(uint16_t index = 0; index < max_flow*2+1; index++)
    {
        if(x_buckets[index] > max_x)
        {
            max_x = x_buckets[index];
            x_index = index;
        }
        if(y_buckets[index] > max_y)
        {
            max_y = y_buckets[index];
            y_index = index;
        }
    }

    int16_t dominant_x = x_index - max_flow;
    int16_t dominant_y = y_index - max_flow;

    uint8_t inlier_mask[match_counter];
    int16_t tolerance = INITIAL_OUTLIER_TOLERANCE;
    for(uint16_t index = 0; index < match_counter; index++)
    {
        feature_match_t current_match = matches[index];
        point2D_u16_t p0 = keypoints0[current_match.feat_idx0];
        point2D_u16_t p1 = keypoints1[current_match.feat_idx1];
        int16_t dx = p1.x - p0.x;
        int16_t dy = p1.y - p0.y;
        if (dx <= dominant_x + tolerance && dx >= dominant_x - tolerance &&
            dy <= dominant_y + tolerance && dy >= dominant_y - tolerance)
        {
            inlier_mask[index] = 1;
        }
        else
        {
            inlier_mask[index] = 0;
        }
    }
    /* Run first iteration of motion estimation using the
       histogram based outlier rejection. */
    rigid_body_motion_estimation(keypoints0,
                                 keypoints1,
                                 matches,
                                 motion,
                                 inlier_mask,
                                 match_counter,
                                 img_width,
                                 img_height);

    float rot[4];
    rot[0] =  cosf(motion->rot_z);
    rot[1] = -sinf(motion->rot_z);
    rot[2] =  sinf(motion->rot_z);
    rot[3] =  cosf(motion->rot_z);

    /* Determine which features satisfy the first motion estimate with an
       euclidead distance error of less than ITERATIVE_OUTLIER_TOLERANCE. */
    uint16_t inlier_count = 0;
    for(uint16_t index = 0; index < match_counter; index++)
    {
        feature_match_t current_match = matches[index];
        point2D_u16_t p0 = keypoints0[current_match.feat_idx0];
        point2D_u16_t p1 = keypoints1[current_match.feat_idx1];
        float proj_x = rot[0] * p0.x + rot[1] * p0.y + motion->flow_x;
        float proj_y = rot[2] * p0.x + rot[3] * p0.y + motion->flow_y;
        //printf("pro x %f, real x %d, proj y %f, real y %d \n", proj_x, p1.x, proj_y, p1.y);
        float euclidean_sq = (proj_x - p1.x) * (proj_x - p1.x) + (proj_y - p1.y) * (proj_y - p1.y);
        if (euclidean_sq < (ITERATIVE_OUTLIER_TOLERANCE*ITERATIVE_OUTLIER_TOLERANCE))
        {
            inlier_mask[index] = 1;
            inlier_count++;
        }
        else
        {
            inlier_mask[index] = 0;
        }
    }

    /* Run second iteration of motion estimation using the
       euclidean distance based outlier rejection. */
    if(inlier_count <= 3)
    {
        printf("SKIPPING MOTION ESTIMATION REFINEMENT, NOT ENOUGH INLIERS\n");
        return;
    }
    rigid_body_motion_estimation(keypoints0,
                                 keypoints1,
                                 matches,
                                 motion,
                                 inlier_mask,
                                 match_counter,
                                 img_width,
                                 img_height);

    return;
}
