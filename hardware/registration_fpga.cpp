#undef __ARM_NEON__
#undef __ARM_NEON
#include "registration_fpga.h"
#define __ARM_NEON__
#define __ARM_NEON

// #define DEBUG_TEST
const int COARSE_LOOKUP_SIZE = 32;

void sort_cloud_by_horizon_angle(My_PointXYZI* laserCloudInArray, int cloudSize)
{
    My_PointXYZI laserCloudInArray_temp[Horizon_SCAN][1000];
    int laserCloudInArray_temp_count[Horizon_SCAN];
    My_PointXYZI point;
    My_PointXYZI_HW point_hw;
    

    for(int i = 0; i < Horizon_SCAN; i++)
    {
        laserCloudInArray_temp_count[i] = 0;
    }

    for (int i = 0; i < cloudSize; i++)
    {
        point.x = laserCloudInArray[i].x;
        point.y = laserCloudInArray[i].y;
        point.z = laserCloudInArray[i].z;
        point.intensity = laserCloudInArray[i].intensity;

        point_hw.x = point.x;
        point_hw.y = point.y;
        point_hw.z = point.z;

        // float horizonAngle = atan2(point.x, point.y) * 180 / M_PI;	// y轴正方向为0度，顺时针增大。
        // // 减去了90度，那就是x轴正方向为0度。 也就是x轴正方向对应rangeimage正中间。在rangeimage中，正角度往左，负角度往右。
        // // KITTI: Velodyne: x = forward, y = left, z = up
        // int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
        // if (columnIdn >= Horizon_SCAN)
        //     columnIdn -= Horizon_SCAN;

        int columnIdn = find_columnid(point_hw);
       
        if (columnIdn >= 0 && columnIdn < Horizon_SCAN)
        {
            laserCloudInArray_temp[columnIdn][laserCloudInArray_temp_count[columnIdn]] = point;
            laserCloudInArray_temp_count[columnIdn]++;
        }
    }
    int max_points_num = 0;
    for(int i = 0; i < Horizon_SCAN; i++)
    {
       if( laserCloudInArray_temp_count[i] > max_points_num )
        max_points_num = laserCloudInArray_temp_count[i];
    }
    DEBUG_LOG( "max points in a vertial line is " << max_points_num );

    int new_array_count = 0;
    for(int i = 0; i < Horizon_SCAN; i++)
    {
        for(int j = 0; j < laserCloudInArray_temp_count[i]; j++)
        {
            laserCloudInArray[new_array_count] = laserCloudInArray_temp[i][j];
            new_array_count++;
        }
    }
    if(new_array_count != cloudSize)
        DEBUG_ERROR( "error! new_array_count != cloudSize with " << new_array_count << " != " << cloudSize );
}

int find_scanid(My_PointXYZI_HW input_point)
{
#pragma HLS inline
#pragma HLS array_partition variable=scanid_lut complete dim=0
	type_temp_hw temp_x = input_point.x;
	type_temp_hw temp_y = input_point.y;
	type_temp_hw temp_z = input_point.z;
    type_temp_hw temp_quad = (temp_x * temp_x + temp_y * temp_y);

	type_temp_hw temp_zxy;
    if(temp_quad > 0)
        temp_zxy = (temp_z * temp_z) / temp_quad;
    else temp_zxy = 1023;
	
    if(temp_z < 0)
		temp_zxy = -temp_zxy;
	int point_scanID2;
	loop_scanid_lookup:
	for(int lut_i = 0; lut_i < SCANID_LUT_SIZE; lut_i ++)
	{
#pragma HLS UNROLL
		if(temp_zxy < scanid_lut[lut_i])
		{
			if(lut_i == 0)
				point_scanID2 = 0;
			else
				point_scanID2 = lut_i - 1;
			break;
		}
		else if(lut_i == (SCANID_LUT_SIZE-1))
		{
			point_scanID2 = SCANID_LUT_SIZE - 1;
		}
		else{}
	}

	return point_scanID2;
}

static int find_columnid_in_quadrant(type_temp_hw input_lut_value, int quadrant_min, int quadrant_max, int coarse_start_index)
{
#pragma HLS inline
// #pragma HLS array_partition variable=columnid_lut cyclic factor=COARSE_LOOKUP_SIZE dim=1
#pragma HLS array_partition variable=coasrse_columnid_lut complete dim=0
	
	type_radian_hw local_columnid_lut[COARSE_LOOKUP_SIZE];
#pragma HLS array_partition variable=local_columnid_lut complete dim=0

#ifdef DEBUG_TEST	
	int columnid = -1;
	for(int j = quadrant_min; j < quadrant_max; j++)
	{
		if(input_lut_value < columnid_lut[j])
		{
			if(j == 0)
				columnid = 0;
			else
				columnid =  j - 1;
			break;
		}
		else if(j == (quadrant_max-1))
			columnid =  (quadrant_max-1);
		else{}
	}
#endif

#ifdef DEBUG_TEST
	int temp_columnid2_origin = -1;
			// 先以 32 为间隔，粗查找一次
		for(int j = quadrant_min; j < quadrant_max; j+=COARSE_LOOKUP_SIZE)
		{
			#pragma HLS loop_tripcount min=16 max=16
			#pragma HLS UNROLL
			if(input_lut_value < columnid_lut[j])
			{
				if(j == 0)
					temp_columnid2_origin = 0;
				else if(j < (quadrant_min + COARSE_LOOKUP_SIZE))
					temp_columnid2_origin = quadrant_min;
				else
					temp_columnid2_origin =  j - COARSE_LOOKUP_SIZE;
				break;
			}
			else if(j >= (quadrant_max-COARSE_LOOKUP_SIZE))
				temp_columnid2_origin =  (quadrant_max-COARSE_LOOKUP_SIZE);
			else{}
		}
#endif

	int temp_columnid = -1;
	int temp_columnid2 = -1;

	// 先以 32 为间隔，粗查找一次
	ap_uint<1> sign_coarse[COARSE_COLUMN_LOOP_SIZE];
	loop_coarse_compare:
	for(int j = 0; j < COARSE_COLUMN_LOOP_SIZE; j++)
	{
	#pragma HLS UNROLL
		if(input_lut_value < coasrse_columnid_lut[coarse_start_index+j])
			sign_coarse[j] = 1;
		else
			sign_coarse[j] = 0;
	}

	loop_coarse_select:
	for(int j = 0; j < COARSE_COLUMN_LOOP_SIZE-1; j++)
	{
	#pragma HLS UNROLL
		if(j == 0 && sign_coarse[j] == 1)
			temp_columnid2 = 0;
		else if(sign_coarse[j] == 0 && sign_coarse[j+1] == 1)
			temp_columnid2 = j;
		else if(j == (COARSE_COLUMN_LOOP_SIZE-2) && sign_coarse[j] == 0)
			temp_columnid2 = COARSE_COLUMN_LOOP_SIZE-1;
	}
	temp_columnid2 = quadrant_min + temp_columnid2 * COARSE_LOOKUP_SIZE;
	int first_index = temp_columnid2;

	// 将查找表数据读取到本地
	for(int j = 0; j < COARSE_LOOKUP_SIZE; j++)
	{
	#pragma HLS UNROLL
		local_columnid_lut[j] = columnid_lut[j+first_index];
	}

	// 再对这粗区域中的32个数据，进行精确查找。 COARSE_LOOKUP_SIZE
	ap_uint<1> sign[COARSE_LOOKUP_SIZE];
	loop_fine_compare:
	for(int j = 0; j < COARSE_LOOKUP_SIZE; j++)
	{
	#pragma HLS UNROLL
		if( (j+first_index) < quadrant_max)
		{
			if(input_lut_value >= local_columnid_lut[j])
				sign[j] = 0;
			else
				sign[j] = 1;
		}else sign[j] = 1;
	}

	loop_fine_select:
	for(int j = 0; j < COARSE_LOOKUP_SIZE-1; j++)
	{
	#pragma HLS UNROLL
		if(j == 0 && sign[j] == 1)
			temp_columnid = first_index+j-1;
		else if(sign[j] == 0 && sign[j+1] == 1)
			temp_columnid = first_index+j;
		else if(j == (COARSE_LOOKUP_SIZE-2) && sign[j] == 0)
			temp_columnid = first_index+COARSE_LOOKUP_SIZE-1;
	}
	
#ifdef DEBUG_TEST	
	if(temp_columnid != columnid)
	{
		std::cout << "origin, temp_columnid, temp_columnid2_origin, temp_columnid2: " << columnid << " " << temp_columnid << " " << temp_columnid2_origin << " " << temp_columnid2 << " " 
		<< quadrant_min << " " << quadrant_max << " " << std::endl;
	}
#endif

	return temp_columnid;
}

int find_columnid(My_PointXYZI_HW input_point)
{
#pragma HLS inline
	int temp_columnid;
	int current_quadrant_min;
	int current_quadrant_max;
	int current_coarse_start_index;

	type_temp_hw temp_x = input_point.x;
	type_temp_hw temp_y = input_point.y;
	
	type_temp_hw temp_value = 0;
    if(temp_x != 0) 
       temp_value = temp_y / temp_x;
	
	if(temp_x < 0 and temp_y == 0)
		temp_columnid = 0;
	else if(temp_x == 0 && temp_y < 0)
		temp_columnid = COLUMNID_LUT_SIZE_1;
	else if(temp_x >= 0 && temp_y == 0)
		temp_columnid = COLUMNID_LUT_SIZE_2;
	else if(temp_x == 0 && temp_y >= 0)
		temp_columnid = COLUMNID_LUT_SIZE_3;
	else
	{
		if(temp_x < 0 and temp_y < 0)
		{
			current_quadrant_min = 0;
			current_quadrant_max = COLUMNID_LUT_SIZE_1;
			current_coarse_start_index = 0;
			// temp_columnid = find_columnid_in_quadrant(temp_value, 0, COLUMNID_LUT_SIZE_1, 0);
		}
		else if(temp_x > 0 && temp_y < 0)
		{
			current_quadrant_min = COLUMNID_LUT_SIZE_1;
			current_quadrant_max = COLUMNID_LUT_SIZE_2;
			current_coarse_start_index = 15;
			// temp_columnid = find_columnid_in_quadrant(temp_value, COLUMNID_LUT_SIZE_1, COLUMNID_LUT_SIZE_2, 15);
		}
		else if(temp_x > 0 && temp_y > 0)
		{
			current_quadrant_min = COLUMNID_LUT_SIZE_2;
			current_quadrant_max = COLUMNID_LUT_SIZE_3;
			current_coarse_start_index = 30;
			// temp_columnid = find_columnid_in_quadrant(temp_value, COLUMNID_LUT_SIZE_2, COLUMNID_LUT_SIZE_3, 30);
		}
		else if(temp_x < 0 && temp_y > 0)
		{
			current_quadrant_min = COLUMNID_LUT_SIZE_3;
			current_quadrant_max = COLUMNID_LUT_SIZE;
			current_coarse_start_index = 45;
			// temp_columnid = find_columnid_in_quadrant(temp_value, COLUMNID_LUT_SIZE_3, COLUMNID_LUT_SIZE, 45);
		}
		temp_columnid = find_columnid_in_quadrant(temp_value, current_quadrant_min, current_quadrant_max, current_coarse_start_index);
	}

	return temp_columnid;
}