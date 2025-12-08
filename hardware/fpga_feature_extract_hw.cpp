#undef __ARM_NEON__
#undef __ARM_NEON
#include "registration_fpga.h"
#define __ARM_NEON__
#define __ARM_NEON


static type_picked_hw picked_one = type_picked_hw(1);
static type_picked_hw picked_zero = type_picked_hw(0);
static type_curvature_hw curvature_zero = type_curvature_hw(0);
static type_ground_hw ground_one = type_ground_hw(1);
static type_ground_hw ground_zero = type_ground_hw(0);

/* ******************************************* Debug the stream ****************************************************************** */

static void compute_ground_to_output(
					hls::stream<My_PointXYZI_HW> ground_out_point_stream[PARAL_NUM],
					hls::stream<type_range_hw> ground_out_range_stream[PARAL_NUM],
					hls::stream<type_curvature_hw> ground_out_curvature_stream[PARAL_NUM],
					hls::stream<type_picked_hw> ground_out_picked_stream[PARAL_NUM],
					hls::stream<type_sortind_hw> ground_out_sortind_stream[PARAL_NUM],
					hls::stream<type_ground_hw> ground_out_ground_stream[PARAL_NUM],

					My_PointXYZI_Port* rangeimage_point, type_point_port_hw* rangeimage_range, type_picked_hw *rangeimage_lessflat, type_ground_hw *rangeimage_ground,
					My_PointOutFeatureLocation *rangeimage_flat,
					My_PointOutFeatureLocation *rangeimage_sharp,
					My_PointOutFeatureLocation *rangeimage_lesssharp)
{
	DEBUG_OPERATION(int write_count = 0;);
	DEBUG_OPERATION(int read_count = 0;);
	My_PointXYZI_HW point_read_from_stream[PARAL_NUM];
	type_range_hw range_read_from_stream[PARAL_NUM];
	type_curvature_hw curvature_read_from_stream[PARAL_NUM];
	type_picked_hw picked_read_from_stream[PARAL_NUM];
	type_sortind_hw sortind_read_from_stream[PARAL_NUM];
	type_ground_hw ground_read_from_stream[PARAL_NUM];
#pragma HLS array_partition variable=point_read_from_stream complete dim=0
#pragma HLS array_partition variable=range_read_from_stream complete dim=0
#pragma HLS array_partition variable=curvature_read_from_stream complete dim=0
#pragma HLS array_partition variable=picked_read_from_stream complete dim=0
#pragma HLS array_partition variable=sortind_read_from_stream complete dim=0
#pragma HLS array_partition variable=ground_read_from_stream complete dim=0
// #pragma HLS aggregate variable=point_read_from_stream
	loop_debug_out_horizon:
	for(int j = 0; j < Horizon_SCAN; j++)
	{
		loop_debug_out_scans:
		for(int i = 0; i < N_SCANS; i+=PARAL_NUM)
		{
#pragma HLS PIPELINE II=1
			for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
			{
	#pragma HLS UNROLL
				point_read_from_stream[para_i] = ground_out_point_stream[para_i].read();
				range_read_from_stream[para_i] = ground_out_range_stream[para_i].read();
				curvature_read_from_stream[para_i] = ground_out_curvature_stream[para_i].read();
				picked_read_from_stream[para_i] = ground_out_picked_stream[para_i].read();
				sortind_read_from_stream[para_i] = ground_out_sortind_stream[para_i].read();
				ground_read_from_stream[para_i] = ground_out_ground_stream[para_i].read();
				
				My_PointXYZI_Port tobe_write_point;
				tobe_write_point.x = point_read_from_stream[para_i].x;
				tobe_write_point.y = point_read_from_stream[para_i].y;
				tobe_write_point.z = point_read_from_stream[para_i].z;
				tobe_write_point.intensity = point_read_from_stream[para_i].intensity;
				rangeimage_point[j*N_SCANS + i+para_i] = tobe_write_point;
				rangeimage_range[j*N_SCANS + i+para_i] = range_read_from_stream[para_i];

				if(curvature_read_from_stream[para_i] < 0.1 && picked_read_from_stream[para_i] == 0 &&  range_read_from_stream[para_i] > 2 && ( ground_read_from_stream[para_i] == 1 ||  ground_read_from_stream[para_i] == 2 ))
				{
					rangeimage_lessflat[j*N_SCANS + i+para_i] = picked_one;
				}
				else
				{
					rangeimage_lessflat[j*N_SCANS + i+para_i] = picked_zero;
				}

				if( ground_read_from_stream[para_i] == 1 )
				{
					rangeimage_ground[j*N_SCANS + i+para_i] = ground_one;
				}
				else
				{
					rangeimage_ground[j*N_SCANS + i+para_i] = ground_zero;
				}		
			}
			DEBUG_OPERATION(read_count ++;);
			DEBUG_OPERATION(write_count ++;);
		}
	}

	DEBUG_OPERATION(int second_write_count = 0;);
	DEBUG_OPERATION(int second_read_count = 0;);
	loop_out_flat_sharp_result_result_main:
	for(int j = 0; j < EXTRACT_SEGMENT; j++)	// 分成了6段。
	{
		for(int i = 0; i < N_SCANS/PARAL_NUM; i++)
		{
#pragma HLS PIPELINE
			for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
			{
				for(int m = 0; m < FLAT_SIZE; m++)
				{
					int temp_value = j*N_SCANS*FLAT_SIZE+i*PARAL_NUM*FLAT_SIZE+para_i*FLAT_SIZE+m;
					My_PointOutFeatureLocation temp_out;
#pragma HLS aggregate variable=temp_out
					temp_out.scanId = temp_value % N_SCANS;
					temp_out.columnId = temp_value % Horizon_SCAN;
					rangeimage_flat[temp_value] = temp_out;
				}

				for(int m = 0; m < SHARP_SIZE; m++)
				{
					int temp_value = j*N_SCANS*SHARP_SIZE+i*PARAL_NUM*SHARP_SIZE+para_i*SHARP_SIZE+m;
					My_PointOutFeatureLocation temp_out;
#pragma HLS aggregate variable=temp_out
					temp_out.scanId = (temp_value+100) % N_SCANS;
					temp_out.columnId = (temp_value+1000) % Horizon_SCAN;
					rangeimage_sharp[temp_value] = temp_out;
				}

				for(int m = 0; m < LESS_SHARP_SIZE; m++)
				{
					int temp_value = j*N_SCANS*LESS_SHARP_SIZE+i*PARAL_NUM*LESS_SHARP_SIZE+para_i*LESS_SHARP_SIZE+m;
					My_PointOutFeatureLocation temp_out;
#pragma HLS aggregate variable=temp_out
					temp_out.scanId = (temp_value+50) % N_SCANS;;
					temp_out.columnId = (temp_value+50) % Horizon_SCAN;
					rangeimage_lesssharp[temp_value] = temp_out;
				}

			}
			DEBUG_OPERATION(second_write_count ++;);
			DEBUG_OPERATION(second_read_count ++;);
		}
	}

	DEBUG_LOG("compute_ground_to_output READ WRITE: " << read_count << " " << write_count << " " 
	<< second_read_count << " " << second_write_count << " " );
}


static void integrate_features_to_output(
					hls::stream<My_PointXYZI_HW> registrate_point_cloud_point_stream[PARAL_NUM],
					hls::stream<type_range_hw> registrate_point_cloud_point_range_stream[PARAL_NUM],
					hls::stream<My_PointFeature_hw> registrate_point_cloud_feature_stream[PARAL_NUM],

					My_PointXYZI_Port* rangeimage_point, type_point_port_hw* rangeimage_range, type_picked_hw *rangeimage_lessflat, type_ground_hw *rangeimage_ground,
					My_PointOutFeatureLocation *rangeimage_flat,
					My_PointOutFeatureLocation *rangeimage_sharp,
					My_PointOutFeatureLocation *rangeimage_lesssharp)
{
	My_PointXYZI_HW point_read_from_stream[PARAL_NUM];
	type_range_hw range_read_from_stream[PARAL_NUM];
	My_PointFeature_hw features_read_from_stream[PARAL_NUM];
#pragma HLS array_partition variable=point_read_from_stream complete dim=0
#pragma HLS array_partition variable=range_read_from_stream complete dim=0
#pragma HLS array_partition variable=features_read_from_stream complete dim=0

	for(int j = 0; j < Horizon_SCAN; j++)
	{
		for(int i = 0; i < N_SCANS; i+=PARAL_NUM)
		{
			for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
			{
	#pragma HLS UNROLL
				point_read_from_stream[para_i] = registrate_point_cloud_point_stream[para_i].read();
				range_read_from_stream[para_i] = registrate_point_cloud_point_range_stream[para_i].read();
				features_read_from_stream[para_i] = registrate_point_cloud_feature_stream[para_i].read();
				
				rangeimage_point[j*N_SCANS + i+para_i].x = point_read_from_stream[para_i].x;
				rangeimage_point[j*N_SCANS + i+para_i].y = point_read_from_stream[para_i].y;
				rangeimage_point[j*N_SCANS + i+para_i].z = point_read_from_stream[para_i].z;
				rangeimage_point[j*N_SCANS + i+para_i].intensity = point_read_from_stream[para_i].intensity;
				rangeimage_range[j*N_SCANS + i+para_i] = range_read_from_stream[para_i];

				if(features_read_from_stream[para_i].curvature < 0.1 && features_read_from_stream[para_i].picked == 0 &&  range_read_from_stream[para_i] > 2 && ( features_read_from_stream[para_i].ground == 1 ||  features_read_from_stream[para_i].ground == 2 ))
				{
					rangeimage_lessflat[j*N_SCANS + i+para_i] = picked_one;
				}
				else
				{
					rangeimage_lessflat[j*N_SCANS + i+para_i] = picked_zero;
				}

				if( features_read_from_stream[para_i].ground == 1 )
				{
					rangeimage_ground[j*N_SCANS + i+para_i] = ground_one;
				}
				else
				{
					rangeimage_ground[j*N_SCANS + i+para_i] = ground_zero;
				}		
			}
		}
	}

	loop_out_flat_sharp_result_result_main:
	for(int j = 0; j < EXTRACT_SEGMENT; j++)	// 分成了6段。
	{
		for(int i = 0; i < N_SCANS/PARAL_NUM; i++)
		{
			for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
			{
				for(int m = 0; m < FLAT_SIZE; m++)
				{
					int temp_value = j*N_SCANS*FLAT_SIZE+i*PARAL_NUM*FLAT_SIZE+para_i*FLAT_SIZE+m;
					rangeimage_flat[temp_value].scanId = temp_value % N_SCANS;
					rangeimage_flat[temp_value].columnId = temp_value % Horizon_SCAN;
				}

				for(int m = 0; m < SHARP_SIZE; m++)
				{
					int temp_value = j*N_SCANS*SHARP_SIZE+i*PARAL_NUM*SHARP_SIZE+para_i*SHARP_SIZE+m;
					rangeimage_sharp[temp_value].scanId =  (temp_value+100) % N_SCANS;
					rangeimage_sharp[temp_value].columnId =   (temp_value+1000) % Horizon_SCAN;
				}

				for(int m = 0; m < LESS_SHARP_SIZE; m++)
				{
					int temp_value = j*N_SCANS*LESS_SHARP_SIZE+i*PARAL_NUM*LESS_SHARP_SIZE+para_i*LESS_SHARP_SIZE+m;
					rangeimage_lesssharp[temp_value].scanId = (temp_value+50) % N_SCANS;;
					rangeimage_lesssharp[temp_value].columnId = (temp_value+50) % N_SCANS;;
				}

			}
		}
	}
}

static void allocate_image_to_output(
					hls::stream<My_PointXYZI_HW> registrate_point_cloud_stream[ALLOCATE_PARAL_NUM],
					hls::stream<type_range_hw> registrate_point_cloud_range_stream[ALLOCATE_PARAL_NUM],
					My_PointXYZI_Port* rangeimage_point, type_point_port_hw* rangeimage_range, type_picked_hw *rangeimage_lessflat, type_ground_hw *rangeimage_ground,
					My_PointOutFeatureLocation *rangeimage_flat,
					My_PointOutFeatureLocation *rangeimage_sharp,
					My_PointOutFeatureLocation *rangeimage_lesssharp
)
{
	type_range_hw range_read_from_stream[ALLOCATE_PARAL_NUM];
#pragma HLS array_partition variable=range_read_from_stream complete dim=0
	My_PointXYZI_HW point_read_from_stream[ALLOCATE_PARAL_NUM];
#pragma HLS array_partition variable=point_read_from_stream complete dim=0

	for(int j = 0; j < Horizon_SCAN; j++)
	{
		for(int i = 0; i < N_SCANS; i+=ALLOCATE_PARAL_NUM)
		{
			for(int para_i = 0; para_i < ALLOCATE_PARAL_NUM; para_i ++)
			{
	#pragma HLS UNROLL
				range_read_from_stream[para_i] = registrate_point_cloud_range_stream[para_i].read();
				point_read_from_stream[para_i] = registrate_point_cloud_stream[para_i].read();
				rangeimage_point[j*N_SCANS + i+para_i].x = point_read_from_stream[para_i].x;
				rangeimage_point[j*N_SCANS + i+para_i].y = point_read_from_stream[para_i].y;
				rangeimage_point[j*N_SCANS + i+para_i].z = point_read_from_stream[para_i].z;
				rangeimage_point[j*N_SCANS + i+para_i].intensity = point_read_from_stream[para_i].intensity;
				rangeimage_range[j*N_SCANS + i+para_i] = range_read_from_stream[para_i];
				rangeimage_lessflat[j*N_SCANS + i+para_i] = i % 10;
				rangeimage_ground[j*N_SCANS + i+para_i] = i % 10;
			}
		}
	}

	loop_out_flat_sharp_result_result_main:
	for(int j = 0; j < EXTRACT_SEGMENT; j++)	// 分成了6段。
	{
		for(int i = 0; i < N_SCANS/PARAL_NUM; i++)
		{
			for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
			{
				for(int m = 0; m < FLAT_SIZE; m++)
				{
					int temp_value = j*N_SCANS*FLAT_SIZE+i*PARAL_NUM*FLAT_SIZE+para_i*FLAT_SIZE+m;
					rangeimage_flat[temp_value].scanId = temp_value % N_SCANS;
					rangeimage_flat[temp_value].columnId = temp_value % Horizon_SCAN;
				}

				for(int m = 0; m < SHARP_SIZE; m++)
				{
					int temp_value = j*N_SCANS*SHARP_SIZE+i*PARAL_NUM*SHARP_SIZE+para_i*SHARP_SIZE+m;
					rangeimage_sharp[temp_value].scanId =  (temp_value+100) % N_SCANS;
					rangeimage_sharp[temp_value].columnId =   (temp_value+1000) % Horizon_SCAN;
				}

				for(int m = 0; m < LESS_SHARP_SIZE; m++)
				{
					int temp_value = j*N_SCANS*LESS_SHARP_SIZE+i*PARAL_NUM*LESS_SHARP_SIZE+para_i*LESS_SHARP_SIZE+m;
					rangeimage_lesssharp[temp_value].scanId = (temp_value+50) % N_SCANS;;
					rangeimage_lesssharp[temp_value].columnId = (temp_value+50) % N_SCANS;;
				}

			}
		}
	}
}

static void cache_rangeimage_to_output(
					hls::stream<My_PointXYZI_HW> cache_image_point_stream[PARAL_NUM],
					hls::stream<type_range_hw> cache_image_range_stream[PARAL_NUM],
					My_PointXYZI_Port* rangeimage_point, type_point_port_hw* rangeimage_range, type_picked_hw *rangeimage_lessflat, type_ground_hw *rangeimage_ground,
					My_PointOutFeatureLocation *rangeimage_flat,
					My_PointOutFeatureLocation *rangeimage_sharp,
					My_PointOutFeatureLocation *rangeimage_lesssharp)
{
	type_range_hw range_read_from_stream[ALLOCATE_PARAL_NUM];
#pragma HLS array_partition variable=range_read_from_stream complete dim=0
	My_PointXYZI_HW point_read_from_stream[ALLOCATE_PARAL_NUM];
#pragma HLS array_partition variable=point_read_from_stream complete dim=0

	for(int j = 0; j < Horizon_SCAN; j++)
	{
		for(int i = 0; i < N_SCANS; i+=PARAL_NUM)
		{
			for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
			{
	#pragma HLS UNROLL
				range_read_from_stream[para_i] = cache_image_range_stream[para_i].read();
				point_read_from_stream[para_i] = cache_image_point_stream[para_i].read();
				rangeimage_point[j*N_SCANS + i+para_i].x = point_read_from_stream[para_i].x;
				rangeimage_point[j*N_SCANS + i+para_i].y = point_read_from_stream[para_i].y;
				rangeimage_point[j*N_SCANS + i+para_i].z = point_read_from_stream[para_i].z;
				rangeimage_point[j*N_SCANS + i+para_i].intensity = point_read_from_stream[para_i].intensity;
				rangeimage_range[j*N_SCANS + i+para_i] = range_read_from_stream[para_i];
				rangeimage_lessflat[j*N_SCANS + i+para_i] = i % 10;
				rangeimage_ground[j*N_SCANS + i+para_i] = i % 10;
			}
		}
	}

	loop_out_flat_sharp_result_result_main:
	for(int j = 0; j < EXTRACT_SEGMENT; j++)	// 分成了6段。
	{
		for(int i = 0; i < N_SCANS/PARAL_NUM; i++)
		{
			for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
			{
				for(int m = 0; m < FLAT_SIZE; m++)
				{
					int temp_value = j*N_SCANS*FLAT_SIZE+i*PARAL_NUM*FLAT_SIZE+para_i*FLAT_SIZE+m;
					rangeimage_flat[temp_value].scanId = temp_value % N_SCANS;
					rangeimage_flat[temp_value].columnId = temp_value % Horizon_SCAN;
				}

				for(int m = 0; m < SHARP_SIZE; m++)
				{
					int temp_value = j*N_SCANS*SHARP_SIZE+i*PARAL_NUM*SHARP_SIZE+para_i*SHARP_SIZE+m;
					rangeimage_sharp[temp_value].scanId =  (temp_value+100) % N_SCANS;
					rangeimage_sharp[temp_value].columnId =   (temp_value+1000) % Horizon_SCAN;
				}

				for(int m = 0; m < LESS_SHARP_SIZE; m++)
				{
					int temp_value = j*N_SCANS*LESS_SHARP_SIZE+i*PARAL_NUM*LESS_SHARP_SIZE+para_i*LESS_SHARP_SIZE+m;
					rangeimage_lesssharp[temp_value].scanId = (temp_value+50) % N_SCANS;;
					rangeimage_lesssharp[temp_value].columnId = (temp_value+50) % N_SCANS;;
				}

			}
		}
	}
}

static void compute_curvature_to_output(
					hls::stream<My_PointXYZI_HW> curvature_out_point_stream[PARAL_NUM],
					hls::stream<type_range_hw> curvature_out_range_stream[PARAL_NUM],
					hls::stream<type_curvature_hw> curvature_out_curvature_stream[PARAL_NUM],
					hls::stream<type_picked_hw> curvature_out_picked_stream[PARAL_NUM],
					My_PointXYZI_Port* rangeimage_point, type_point_port_hw* rangeimage_range, type_picked_hw *rangeimage_lessflat, type_ground_hw *rangeimage_ground,
					My_PointOutFeatureLocation *rangeimage_flat,
					My_PointOutFeatureLocation *rangeimage_sharp,
					My_PointOutFeatureLocation *rangeimage_lesssharp)
{
	My_PointXYZI_HW point_read_from_stream[PARAL_NUM];
	type_range_hw range_read_from_stream[PARAL_NUM];
	type_curvature_hw curvature_read_from_stream[PARAL_NUM];
	type_picked_hw picked_read_from_stream[PARAL_NUM];
#pragma HLS array_partition variable=range_read_from_stream complete dim=0
#pragma HLS array_partition variable=point_read_from_stream complete dim=0
#pragma HLS array_partition variable=curvature_read_from_stream complete dim=0
#pragma HLS array_partition variable=picked_read_from_stream complete dim=0

	for(int j = 0; j < Horizon_SCAN; j++)
	{
		for(int i = 0; i < N_SCANS; i+=PARAL_NUM)
		{
			for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
			{
	#pragma HLS UNROLL
				range_read_from_stream[para_i] = curvature_out_range_stream[para_i].read();
				point_read_from_stream[para_i] = curvature_out_point_stream[para_i].read();
				curvature_read_from_stream[para_i] = curvature_out_curvature_stream[para_i].read();
				picked_read_from_stream[para_i] = curvature_out_picked_stream[para_i].read();
				rangeimage_point[j*N_SCANS + i+para_i].x = point_read_from_stream[para_i].x;
				rangeimage_point[j*N_SCANS + i+para_i].y = point_read_from_stream[para_i].y;
				rangeimage_point[j*N_SCANS + i+para_i].z = point_read_from_stream[para_i].z;
				rangeimage_point[j*N_SCANS + i+para_i].intensity = point_read_from_stream[para_i].intensity;
				rangeimage_range[j*N_SCANS + i+para_i] = range_read_from_stream[para_i];

				if(curvature_read_from_stream[para_i] < 0.01 && picked_read_from_stream[para_i] == 0 &&  range_read_from_stream[para_i] > 2)
				{
					rangeimage_lessflat[j*N_SCANS + i+para_i] = picked_one;
					rangeimage_ground[j*N_SCANS + i+para_i] = ground_one;
				}
				else
				{
					rangeimage_lessflat[j*N_SCANS + i+para_i] = picked_zero;
					rangeimage_ground[j*N_SCANS + i+para_i] = ground_zero;
				}
				
			}
		}
	}

	loop_out_flat_sharp_result_result_main:
	for(int j = 0; j < EXTRACT_SEGMENT; j++)	// 分成了6段。
	{
		for(int i = 0; i < N_SCANS/PARAL_NUM; i++)
		{
			for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
			{
				for(int m = 0; m < FLAT_SIZE; m++)
				{
					int temp_value = j*N_SCANS*FLAT_SIZE+i*PARAL_NUM*FLAT_SIZE+para_i*FLAT_SIZE+m;
					rangeimage_flat[temp_value].scanId = temp_value % N_SCANS;
					rangeimage_flat[temp_value].columnId = temp_value % Horizon_SCAN;
				}

				for(int m = 0; m < SHARP_SIZE; m++)
				{
					int temp_value = j*N_SCANS*SHARP_SIZE+i*PARAL_NUM*SHARP_SIZE+para_i*SHARP_SIZE+m;
					rangeimage_sharp[temp_value].scanId =  (temp_value+100) % N_SCANS;
					rangeimage_sharp[temp_value].columnId =   (temp_value+1000) % Horizon_SCAN;
				}

				for(int m = 0; m < LESS_SHARP_SIZE; m++)
				{
					int temp_value = j*N_SCANS*LESS_SHARP_SIZE+i*PARAL_NUM*LESS_SHARP_SIZE+para_i*LESS_SHARP_SIZE+m;
					rangeimage_lesssharp[temp_value].scanId = (temp_value+50) % N_SCANS;;
					rangeimage_lesssharp[temp_value].columnId = (temp_value+50) % N_SCANS;;
				}

			}
		}
	}
}

/* ******************************************* Normal code ****************************************************************** */

static void array_to_stream(My_PointXYZI_Port* laserCloudInArray, int cloudSize, hls::stream<My_PointXYZI_HW> &laserCloudInArray_stream)
{
	DEBUG_OPERATION(int write_count = 0;);
	My_PointXYZI_HW temp_in_point;
loop_read_input_array:
	for (int i = 0; i < cloudSize; i++)
	{
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=110000 max=160000
		temp_in_point.x = laserCloudInArray[i].x;
		temp_in_point.y = laserCloudInArray[i].y;
		temp_in_point.z = laserCloudInArray[i].z;
		temp_in_point.intensity = laserCloudInArray[i].intensity;
		laserCloudInArray_stream.write(temp_in_point);
		DEBUG_OPERATION(write_count++;)
	}
	// temp_in_point.intensity = -1;
	// laserCloudInArray_stream.write(temp_in_point);
	// write_count++;
	DEBUG_LOG("write_count = " << write_count <<  " cloudSize = " << cloudSize);
}

static void allocate_rangeimage(hls::stream<My_PointXYZI_HW> &laserCloudInArray_stream, int &cloudSize, type_point_port_hw start_point_ori, type_point_port_hw end_point_ori,
		hls::stream<My_PointXYZI_HW> &point_stream, hls::stream<type_range_hw> &range_stream,
		hls::stream<type_scanid> &scanid_stream, hls::stream<type_scanid> &column_id_stream, hls::stream<type_picked_hw> &end_stream )
{
	int read_count = 0;		// 统计读了多少次
	int write_count = 0;	// 统计写了多少次

	My_PointXYZI_HW point;		// 正在处理的点
	int point_scanID;	// 正在处理的点的 scanid
	int point_columnID;	// 正在处理的点的 columnid
	int count = cloudSize;	//计数点云有效点的个数
	int scanid_remove_point_size = 0;	// 由于scanid超出范围被去除的点的个数
	int columnid_remove_point_size = 0;	// 由于columnid 超出范围
	int rangeimage_remove_point_size = 0;	// 由于rangeimage过程中被去除的点的个数
	int rangeimage_overlay_point_size = 0;	// rangeimage 中重覆盖的点
	ap_uint<1> halfPassed = 0;  // judge whether the point's z/x angle over pi； if over, compare with end_ori.

	int current_column_id;	// 当前 buffer 正在处理的真实的 column
	int current_buffer_column_id = 0;	// 当前 buffer 的计数，恒等于 current_column_id % BUFFER_LINES

	int read_i = 0;		// 读取 i, 目前没用
	ap_uint<1> pp_read_i = 0;	// 乒乓buffer 待读取的index
	ap_uint<1> pp_process_i = 1;	// 乒乓buffer 待处理的index

	My_PointXYZI_HW input_pp_buffer[2];	// 输入点的 乒乓 buffer
#pragma HLS array_partition variable=input_pp_buffer complete dim=0

	input_pp_buffer[pp_read_i] = laserCloudInArray_stream.read();
	pp_process_i = pp_read_i;
	pp_read_i = ~pp_read_i;
	read_count++;

	type_angle_hw startOri = start_point_ori;
	type_angle_hw endOri = end_point_ori;

	loop_allocate_rangeimage_main:
	for(int i = 0; i < (cloudSize-1); i++)	// 前面已经读了一次了
	{
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=110000 max=160000

		// 1. 读取数据到 linebuffer 中
		input_pp_buffer[pp_read_i] = laserCloudInArray_stream.read();
		read_count++;
		pp_process_i = ~pp_read_i;
		pp_read_i = ~pp_read_i;

		// 处理 processing_buffer ,即处理之前缓存的点
		point.x = input_pp_buffer[pp_process_i].x;
		point.y = input_pp_buffer[pp_process_i].y;
		point.z = input_pp_buffer[pp_process_i].z;
		point.intensity = input_pp_buffer[pp_process_i].intensity;
		// todo:

		int point_scanID = find_scanid(point);
		int point_columnID = find_columnid(point);

		type_point_hw relTime = point_columnID / Horizon_SCAN;
		//点强度=线号+点相对时间（即一个整数+一个小数，整数部分是线号，小数部分是该点的相对时间）,匀速扫描：根据当前扫描的角度和扫描周期计算相对扫描起始位置的时间
		point.intensity = point_scanID + scanPeriod_hw * relTime;

		// todo:
		type_temp_hw temp_dist = point.x * point.x + point.y * point.y + point.z * point.z;
		type_range_hw range;
		if(temp_dist > 0)
			range = my_sqrt(temp_dist);
		else range = 0;
		
		// write to output
		point_stream.write(point);
		range_stream.write(range);
		scanid_stream.write(point_scanID);
		column_id_stream.write(point_columnID);
		end_stream.write(picked_zero);
		write_count++;
	}

	end_stream.write(picked_one);
	DEBUG_LOG("allocate_rangeimage read_count = " << read_count << " write_count = " << write_count);
}

static void organize_point_cloud(hls::stream<My_PointXYZI_HW> &point_stream, hls::stream<type_range_hw> &range_stream,
			hls::stream<type_scanid> &scanid_stream, hls::stream<type_scanid> &column_id_stream, hls::stream<type_picked_hw> &end_stream,
			hls::stream<My_PointXYZI_HW> registrate_point_cloud_stream[ALLOCATE_PARAL_NUM], hls::stream<type_range_hw> registrate_point_cloud_range_stream[ALLOCATE_PARAL_NUM])
{
	int read_count = 0;		// 统计读了多少次
	int write_count = 0;	// 统计写了多少次

	My_PointXYZI_HW point_from_stream;
	type_range_hw range_from_stream;
	type_scanid scanid_from_stream;
	type_scanid column_id_from_stream;
	type_picked_hw end_from_stream;

	type_picked_hw read_flag = 1;
	int out_column_id = 0;
	int out_scanID = 0;
	My_PointXYZI_HW raw_point_line_buffer[N_SCANS][BUFFER_LINES];	// 正在处理点的buffer, 此处用了多层buffer.
	type_range_hw raw_range_line_buffer[N_SCANS][BUFFER_LINES];
#pragma HLS array_partition variable=raw_point_line_buffer cyclic factor=ALLOCATE_PARAL_NUM dim=1
#pragma HLS array_partition variable=raw_point_line_buffer complete dim=2
#pragma HLS array_partition variable=raw_range_line_buffer cyclic factor=ALLOCATE_PARAL_NUM dim=1
#pragma HLS array_partition variable=raw_range_line_buffer complete dim=2

	ap_uint<1> insert_flag_buffer[N_SCANS][Horizon_SCAN];	// 用一个 64*1800 flag数组来避免数据清空操作，从而消除数据依赖。
#pragma HLS array_partition variable=insert_flag_buffer cyclic factor=ALLOCATE_PARAL_NUM dim=1
#pragma HLS array_partition variable=insert_flag_buffer cyclic factor=BUFFER_LINES dim=2
#pragma HLS bind_storage variable=insert_flag_buffer type=RAM_1WNR		// 这个变量为啥需要 1WNR?

	int reset_i = 0;
	int reset_j = BUFFER_LINES;

	const int HALF_BUFFER_LINES = BUFFER_LINES/2;
		// 初始化缓冲数组。
loop_allocate_rangeimage_reset_array:
	for(int i = 0; i < N_SCANS; i++)
	{
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=BUFFER_LINES max=BUFFER_LINES
		for(int j = 0; j < BUFFER_LINES*2; j++)
		{
			insert_flag_buffer[i][j] = 0;
		}
	}

	My_PointXYZI_HW default_out_point[ALLOCATE_PARAL_NUM];
	type_range_hw default_out_range[ALLOCATE_PARAL_NUM];
#pragma HLS array_partition variable=default_out_point complete dim=0
#pragma HLS array_partition variable=default_out_range complete dim=0

	for(int i = 0; i < ALLOCATE_PARAL_NUM; i++)
	{
#pragma HLS UNROLL
		default_out_point[i].x = 0;
		default_out_point[i].y = 0;
		default_out_point[i].z = 0;
		default_out_point[i].intensity = 0;
		default_out_range[i] = -1;
	}

	end_from_stream = end_stream.read();
loop_organize_points_main:
	while(end_from_stream == 0)
	{
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=110000 max=200000
#pragma HLS dependence variable=raw_point_line_buffer intra false
#pragma HLS dependence variable=raw_range_line_buffer intra false
#pragma HLS dependence variable=insert_flag_buffer intra false
#pragma HLS dependence variable=insert_flag_buffer inter false
#pragma HLS dependence variable=read_flag intra false
		// DEBUG_LOG("read_flag: " << read_flag << " column_id_from_stream: " << column_id_from_stream << " out_column_id: " << out_column_id);
		if(read_flag == 1)
		{
			point_from_stream = point_stream.read();
			range_from_stream = range_stream.read();
			scanid_from_stream = scanid_stream.read();
			column_id_from_stream = column_id_stream.read();
			end_from_stream = end_stream.read();
			read_count++;

			// store data in the buffer
			raw_point_line_buffer[scanid_from_stream][column_id_from_stream%BUFFER_LINES] = point_from_stream;
			raw_range_line_buffer[scanid_from_stream][column_id_from_stream%BUFFER_LINES] = range_from_stream;
			insert_flag_buffer[scanid_from_stream][column_id_from_stream] = 1;

			for(int j = 1; j < BUFFER_LINES; j++)
			{
				if( reset_i < N_SCANS && (column_id_from_stream+BUFFER_LINES+j) < Horizon_SCAN)
				{
					insert_flag_buffer[reset_i][column_id_from_stream+BUFFER_LINES+j] = 0;
				}
			}
		}
		else
		{
			for(int j = 0; j < BUFFER_LINES; j++)
			{
				if( reset_i < N_SCANS && (column_id_from_stream+BUFFER_LINES+j) < Horizon_SCAN)
				{
					insert_flag_buffer[reset_i][column_id_from_stream+BUFFER_LINES+j] = 0;
				}
			}
		}
		if( (reset_i + 1) == N_SCANS )
		{
			reset_i = 0;
		}
		else
		{
			reset_i = reset_i + 1;
		}

		// judge read_flag
		if((column_id_from_stream - out_column_id) > HALF_BUFFER_LINES)
			read_flag = 0;
		else
			read_flag = 1;
		
		if( ( out_column_id < column_id_from_stream ) && my_abs(column_id_from_stream - out_column_id) > 2)
		{
			// only write
			for(int para_i = 0; para_i < ALLOCATE_PARAL_NUM; para_i ++)
			{
				int temp_out_scanid = out_scanID + para_i;
				int temp_out_column_buffer_id = out_column_id % BUFFER_LINES;
				if(insert_flag_buffer[temp_out_scanid][out_column_id] == 1)
				{
					registrate_point_cloud_stream[para_i].write( raw_point_line_buffer[temp_out_scanid][temp_out_column_buffer_id] );
					registrate_point_cloud_range_stream[para_i].write( raw_range_line_buffer[temp_out_scanid][temp_out_column_buffer_id] );
				}
				else
				{
					registrate_point_cloud_stream[para_i].write( default_out_point[para_i] );
					registrate_point_cloud_range_stream[para_i].write( default_out_range[para_i] );
				}
				write_count++;
			}
			if( (out_scanID + ALLOCATE_PARAL_NUM) >= N_SCANS)
			{
				out_scanID = 0;
				out_column_id ++;
			}
			else
			{
				out_scanID += ALLOCATE_PARAL_NUM;
			}
		}
	}

	// write the residual points
	type_picked_hw out_flag = 1;
loop_write_residual_data:
	while(out_flag == 1)
	{
#pragma HLS PIPELINE ii=1
#pragma HLS loop_tripcount min=64 max=128	
		for(int para_i = 0; para_i < ALLOCATE_PARAL_NUM; para_i ++)
		{
			int temp_out_scanid = out_scanID + para_i;
			int temp_out_column_buffer_id = out_column_id % BUFFER_LINES;
			if(insert_flag_buffer[temp_out_scanid][out_column_id] == 1)
			{
				registrate_point_cloud_stream[para_i].write( raw_point_line_buffer[temp_out_scanid][temp_out_column_buffer_id] );
				registrate_point_cloud_range_stream[para_i].write( raw_range_line_buffer[temp_out_scanid][temp_out_column_buffer_id] );
			}
			else
			{
				registrate_point_cloud_stream[para_i].write( default_out_point[para_i] );
				registrate_point_cloud_range_stream[para_i].write( default_out_range[para_i] );
			}
			write_count++;
		}
		if( (out_scanID + ALLOCATE_PARAL_NUM) >= N_SCANS)
		{
			out_scanID = 0;

			if(out_column_id >= (Horizon_SCAN-1))
				out_flag = 0;
			else
				out_column_id ++;
		}
		else
		{
			out_scanID += ALLOCATE_PARAL_NUM;
		}
	}

	DEBUG_LOG("organize_point_cloud read_count = " << read_count << " write_count = " << write_count);
}

static void complem_function(int first_one_index, int last_one_index, int zero_count, My_PointXYZI_HW *point, type_range_hw *depth)
{
	// std::cout << first_one_index << " " << point[first_one_index].x << " " << point[first_one_index].y << " " << point[first_one_index].z << " " << point[first_one_index].intensity << std::endl;
	type_point_hw x_increm = ( point[last_one_index].x - point[first_one_index].x ) / (zero_count+1);
	type_point_hw y_increm = ( point[last_one_index].y - point[first_one_index].y ) / (zero_count+1);
	type_point_hw z_increm = ( point[last_one_index].z - point[first_one_index].z ) / (zero_count+1);
	type_point_hw intensity_increm = ( point[last_one_index].intensity - point[first_one_index].intensity ) / (zero_count+1);
	for(int i = first_one_index+1; i < last_one_index; i++)
	{
		point[i].x = point[i-1].x + x_increm;
		point[i].y = point[i-1].y + y_increm;
		point[i].z = point[i-1].z + z_increm;
		point[i].intensity = point[i-1].intensity + intensity_increm;
		depth[i] = my_sqrt(point[i].x * point[i].x + point[i].y * point[i].y + point[i].z * point[i].z);

		// std::cout << i << " " << point[i].x << " " << point[i].y << " " << point[i].z << " "  << point[i].intensity << std::endl;
	}
	// std::cout << last_one_index << " " << point[last_one_index].x << " " << point[last_one_index].y << " " << point[last_one_index].z << " " << point[last_one_index].intensity << std::endl;
}

static void comple_point(hls::stream<My_PointXYZI_HW> registrate_point_cloud_stream[ALLOCATE_PARAL_NUM],
					hls::stream<type_range_hw> registrate_point_cloud_range_stream[ALLOCATE_PARAL_NUM],
					hls::stream<My_PointXYZI_HW> comple_point_cloud_stream[ALLOCATE_PARAL_NUM],
					hls::stream<type_range_hw> comple_point_cloud_range_stream[ALLOCATE_PARAL_NUM])
{
	My_PointXYZI_HW point_cloud_matrix[N_SCANS][Horizon_SCAN];
	type_range_hw range_matrix[N_SCANS][Horizon_SCAN];

	for(int j = 0; j < Horizon_SCAN; j++)
	{
		for(int i = 0; i < N_SCANS; i+=ALLOCATE_PARAL_NUM)
		{
			for(int para_i = 0; para_i < ALLOCATE_PARAL_NUM; para_i ++)
			{
				point_cloud_matrix[i+para_i][j] = registrate_point_cloud_stream[para_i].read();
				range_matrix[i+para_i][j] = registrate_point_cloud_range_stream[para_i].read();
			}
		}
	}

	const int window_size = 5;
	const int pad = window_size/2;

	std::cout << "window_size,pad: " << window_size << " " << pad << std::endl;  
	type_range_hw depth[window_size];
	My_PointXYZI_HW point[window_size];	
	ap_uint<1> cmp[window_size];	

	int status_count[7] = {0};

	std::cout << "status_count init: " << status_count[0] << " " << status_count[1] << " "<< status_count[2] << " "
	<< status_count[3] << " "<< status_count[4] << " "<< status_count[5] << " "<< status_count[6] << " " << std::endl;

	for(int i = 0; i < N_SCANS; i++)
	{
		for(int j = pad; j < Horizon_SCAN-pad; j++)
		{

			for(int k = 0; k < window_size; k++)
			{
				depth[k] = range_matrix[i][j-pad+k];
				point[k] = point_cloud_matrix[i][j-pad+k];
			}

			// 记录首先两个1之间0的个数。
			int compl_seq_count = 0;
			bool record = false;

			for(int k = 0; k < window_size; k++)
			{
				if(depth[k] > 2)
					cmp[k] = 1;
				else
					cmp[k] = 0;
			}

			// 一共有6种情况，分别是 101xx, 1001x, 10001, 0101x, 01001, 00101
			if(cmp[0] == 1 && cmp[1] == 0 && cmp[2] == 1)
			{
				status_count[0]++;
				complem_function(0, 2, 1, point, depth);
			}
			else if(cmp[0] == 0 && cmp[1] == 0 && cmp[2] == 1 && cmp[3] == 0 && cmp[4] == 0)
			{
				// 这种情况下，这个1不配和后面的点进行约束，均匀化中间的点。它自己就是不可靠的。
				status_count[6]++;
				depth[2] = -1;
				point[2].x = 0;
				point[2].y = 0;
				point[2].z = 0;
				point[2].intensity = 0;
			}
			else if(cmp[0] == 1 && cmp[1] == 0 && cmp[2] == 0 && cmp[3] == 1)
			{
				status_count[1]++;
				complem_function(0, 3, 2, point, depth);
			}
			else if(cmp[0] == 1 && cmp[1] == 0 && cmp[2] == 0 && cmp[3] == 0 && cmp[4] == 1)
			{
				status_count[2]++;
				complem_function(0, 4, 3, point, depth);
			}

			else if(cmp[0] == 0 && cmp[1] == 1 && cmp[2] == 0 && cmp[3] == 1)
			{
				status_count[3]++;
				complem_function(1, 3, 1, point, depth);
			}
			else if(cmp[0] == 0 && cmp[1] == 1 && cmp[2] == 0 && cmp[3] == 0 && cmp[4] == 1)
			{
				status_count[4]++;
				complem_function(1, 4, 2, point, depth);
			}
			else if(cmp[0] == 0 && cmp[1] == 0 && cmp[2] == 1 && cmp[3] == 0 && cmp[4] == 1)
			{
				status_count[5]++;
				complem_function(2, 4, 1, point, depth);
			}

			for(int k = 0; k < window_size; k++)
			{
				range_matrix[i][j-pad+k] = depth[k];
				point_cloud_matrix[i][j-pad+k] = point[k];
			}

		}
	}


	for(int j = 0; j < Horizon_SCAN; j++)
	{
		for(int i = 0; i < N_SCANS; i+=ALLOCATE_PARAL_NUM)
		{
			for(int para_i = 0; para_i < ALLOCATE_PARAL_NUM; para_i ++)
			{
				comple_point_cloud_stream[para_i].write(point_cloud_matrix[i+para_i][j]);
				comple_point_cloud_range_stream[para_i].write(range_matrix[i+para_i][j]);
			}
		}
	}

	std::cout << "status_count end: " << status_count[0] << " " << status_count[1] << " "<< status_count[2] << " "
	<< status_count[3] << " "<< status_count[4] << " "<< status_count[5] << " " << std::endl;

}

/*
目前 II=1.
总体思路： 用 linebuffer[N_SCANS] 从数据流中读取数据。每次读取 ALLOCATE_PARAL_NUM
		  用windowbuffer[ALLOCATE_PARAL_NUM*2] 从 linebuffer 中读取数据。待处理数据为前 ALLOCATE_PARAL_NUM 个。后四个为 +1作用。
		  注意： linebuffer 某一周期读入的数据起点假设为 N， windowbuffer 中 [0:ALLOCATE_PARAL_NUM-1] 数据被处理并流出。
		  		之后更新windowbuffer, windowbuffer[0:ALLOCATE_PARAL_NUM-1] <-- windowbuffer[ALLOCATE_PARAL_NUM:ALLOCATE_PARAL_NUM*2-1], windowbuffer[ALLOCATE_PARAL_NUM:ALLOCATE_PARAL_NUM*2-1] <--- linebuffer[N-ALLOCATE_PARAL_NUM:N-1]
*/
static void update_range(
					hls::stream<My_PointXYZI_HW> registrate_point_cloud_stream[ALLOCATE_PARAL_NUM],
					hls::stream<type_range_hw> registrate_point_cloud_range_stream[ALLOCATE_PARAL_NUM],
					hls::stream<My_PointXYZI_HW> update_registrate_point_cloud_point_stream[ALLOCATE_PARAL_NUM],
					hls::stream<type_range_hw> update_registrate_point_cloud_range_stream[ALLOCATE_PARAL_NUM]
)
{
	int write_count = 0;
	int read_count = 0;
	int line_buffer_read_i = 0;
	const int RANGE_DELAY_READ_COUNT = 2;	// 延迟个数，即读取几次数据后开始处理数据并写入数据。比如此处为2，那么读入0和1之后，读2的同时处理0，写0到流中.然后读3的时候处理1，写1到流中.
	const int RANGE_DELAY_NUM = ALLOCATE_PARAL_NUM*RANGE_DELAY_READ_COUNT;	// 延迟个数乘以并行度。延迟的数据数目。。

	type_range_hw range_read_from_stream[ALLOCATE_PARAL_NUM];
#pragma HLS array_partition variable=range_read_from_stream complete dim=0
	My_PointXYZI_HW point_read_from_stream[ALLOCATE_PARAL_NUM];
#pragma HLS array_partition variable=point_read_from_stream complete dim=0

	My_PointXYZI_HW point_line_buffer[N_SCANS];
#pragma HLS array_partition variable=point_line_buffer cyclic factor=ALLOCATE_PARAL_NUM*3+1  dim=0
	type_range_hw range_line_buffer[N_SCANS];
#pragma HLS array_partition variable=range_line_buffer cyclic factor=ALLOCATE_PARAL_NUM*3+1  dim=0

	type_range_hw range_window_buffer[ALLOCATE_PARAL_NUM*2];	// 考虑+1
#pragma HLS array_partition variable=range_window_buffer complete dim=0
	type_range_hw out_buffer[ALLOCATE_PARAL_NUM];
#pragma HLS array_partition variable=out_buffer complete dim=0
	My_PointXYZI_HW point_window_buffer[ALLOCATE_PARAL_NUM*2];	// 考虑+1
#pragma HLS array_partition variable=point_window_buffer complete dim=0
	My_PointXYZI_HW point_out_buffer[ALLOCATE_PARAL_NUM];
#pragma HLS array_partition variable=point_out_buffer complete dim=0
	int window_buffer_read_i = 0;

	// 读数据.先缓存两次数据
	loop_update_range_precache_data:
	for(int pre_i = 0; pre_i < RANGE_DELAY_NUM; pre_i+=ALLOCATE_PARAL_NUM)
	{
#pragma HLS PIPELINE II=1
// #pragma HLS dependence variable=range_line_buffer intra false
		for(int para_i = 0; para_i < ALLOCATE_PARAL_NUM; para_i ++)
		{
#pragma HLS UNROLL
			range_read_from_stream[para_i] = registrate_point_cloud_range_stream[para_i].read();
			range_line_buffer[pre_i + para_i] = range_read_from_stream[para_i];
			range_window_buffer[pre_i + para_i] = range_read_from_stream[para_i];
			point_read_from_stream[para_i] = registrate_point_cloud_stream[para_i].read();
			point_line_buffer[pre_i + para_i] = point_read_from_stream[para_i];
			point_window_buffer[pre_i + para_i] = point_read_from_stream[para_i];
		}
		read_count++;
	}
	window_buffer_read_i = ALLOCATE_PARAL_NUM*RANGE_DELAY_READ_COUNT;
	line_buffer_read_i = ALLOCATE_PARAL_NUM*RANGE_DELAY_READ_COUNT;

	// 再从流中读取一次数据，下次更新给 windowbuffer
	for(int para_i = 0; para_i < ALLOCATE_PARAL_NUM; para_i ++)
	{
#pragma HLS UNROLL
		range_read_from_stream[para_i] = registrate_point_cloud_range_stream[para_i].read();
		range_line_buffer[line_buffer_read_i + para_i] = range_read_from_stream[para_i];
		point_read_from_stream[para_i] = registrate_point_cloud_stream[para_i].read();
		point_line_buffer[line_buffer_read_i + para_i] = point_read_from_stream[para_i];
	}
	line_buffer_read_i += ALLOCATE_PARAL_NUM;
	read_count++;

loop_update_range_main:
	for(int i = read_count; i < N_SCANS*Horizon_SCAN/ALLOCATE_PARAL_NUM; i++)
	{
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=110000 max=160000
// #pragma HLS dependence variable=range_line_buffer intra false
// // #pragma HLS dependence variable=range_line_buffer inter false
		// 读数据
		for(int para_i = 0; para_i < ALLOCATE_PARAL_NUM; para_i ++)
		{
#pragma HLS UNROLL
			range_read_from_stream[para_i] = registrate_point_cloud_range_stream[para_i].read();
			range_line_buffer[line_buffer_read_i + para_i] = range_read_from_stream[para_i];
			point_read_from_stream[para_i] = registrate_point_cloud_stream[para_i].read();
			point_line_buffer[line_buffer_read_i + para_i] = point_read_from_stream[para_i];

			// std::cout << read_count << " " << line_buffer_read_i + para_i << " "
			// << " " << range_line_buffer[line_buffer_read_i + para_i]
			// << " " << point_line_buffer[line_buffer_read_i + para_i].x << " " << point_line_buffer[line_buffer_read_i + para_i].y<< " "
			// << point_line_buffer[line_buffer_read_i + para_i].z << " " << point_line_buffer[line_buffer_read_i + para_i].intensity << " " << std::endl;
		}
		read_count++;

		// 处理数据
		for(int para_i = 0; para_i < ALLOCATE_PARAL_NUM; para_i ++)
		{
			if(window_buffer_read_i >=  RANGE_DELAY_NUM && window_buffer_read_i < (groundScanInd+RANGE_DELAY_NUM))
			{
				int scan_i = window_buffer_read_i + para_i - RANGE_DELAY_NUM;
				if(range_window_buffer[para_i] == -1)
				{
					if( scan_i == 0)
					{
						out_buffer[para_i] = -2;
					}
					else // 判断前面为-2的情况。 这块判断暂时先不加上，对结果影响很小。这块影响整体的硬件并行。
					{
						out_buffer[para_i] = range_window_buffer[para_i];
					}
				}
				else
				{
					// 这一步还挺重要。。去除了会接近100个平面点，同时这些点会给角点。。这一步对提取地面也很重要，缺少这一步有的地面点会提取不出来。见20210901微信图片
					if (range_window_buffer[para_i] > range_window_buffer[para_i+1]) // 如果后一个点比前一个点range小，那么就有问题，出现间断点
					{
						out_buffer[para_i] = -2;
					}
					else
					{
						out_buffer[para_i] = range_window_buffer[para_i];
					}
				}
			}
			else
			{
				out_buffer[para_i] = range_window_buffer[para_i];
			}
			point_out_buffer[para_i] = point_window_buffer[para_i];
		}
		// DEBUG_LOG("range_window_buffer: " << range_window_buffer[0] << " " << range_window_buffer[1] << " " << range_window_buffer[2] << " "
		// << range_window_buffer[3] << " " << range_window_buffer[4] << " " << range_window_buffer[5] << " " << range_window_buffer[6] << " "
		// <<  " out_buffer: " << out_buffer[0] << " " << out_buffer[1] << " "  << out_buffer[2] << " "  << out_buffer[3] << " ");

		// 输出数据,此处必须用read_count，因为后面是连续输出的
		for(int para_i = 0; para_i < ALLOCATE_PARAL_NUM; para_i ++)
		{
#pragma HLS UNROLL
			update_registrate_point_cloud_range_stream[para_i].write(out_buffer[para_i]);
			update_registrate_point_cloud_point_stream[para_i].write(point_out_buffer[para_i]);
		}
		write_count++;

		// 更新 windowbuffer
		for(int para_i = 0; para_i < ALLOCATE_PARAL_NUM; para_i ++)
		{
#pragma HLS UNROLL
			range_window_buffer[para_i] = range_window_buffer[para_i+ALLOCATE_PARAL_NUM];	// 左移 ALLOCATE_PARAL_NUM 位
			range_window_buffer[para_i+ALLOCATE_PARAL_NUM] = range_line_buffer[para_i+window_buffer_read_i];	// 新数据插入后 ALLOCATE_PARAL_NUM 位
			point_window_buffer[para_i] = point_window_buffer[para_i+ALLOCATE_PARAL_NUM];	// 左移 ALLOCATE_PARAL_NUM 位
			point_window_buffer[para_i+ALLOCATE_PARAL_NUM] = point_line_buffer[para_i+window_buffer_read_i];	// 新数据插入后 ALLOCATE_PARAL_NUM 位
		}
		if((window_buffer_read_i + ALLOCATE_PARAL_NUM) >= N_SCANS)
			window_buffer_read_i = 0;
		else
			window_buffer_read_i = window_buffer_read_i + ALLOCATE_PARAL_NUM;

		// 处理读取的索引
		if((line_buffer_read_i + ALLOCATE_PARAL_NUM) >= N_SCANS)
			line_buffer_read_i = 0;
		else
			line_buffer_read_i = line_buffer_read_i + ALLOCATE_PARAL_NUM;
	}
	// DEBUG_LOG( "update_range after main loop read_count = " << read_count << " write_count = " << write_count);

	// 直接输出 windowbuffer 中的数据
	loop_update_range_final_write:
	for(int delay_i = 0; delay_i < RANGE_DELAY_NUM; delay_i += ALLOCATE_PARAL_NUM)
	{
#pragma HLS PIPELINE
		for(int para_i = 0; para_i < ALLOCATE_PARAL_NUM; para_i ++)
		{
#pragma HLS UNROLL
			update_registrate_point_cloud_range_stream[para_i].write(range_window_buffer[para_i + delay_i]);
			update_registrate_point_cloud_point_stream[para_i].write(point_window_buffer[para_i + delay_i]);
		}
		write_count++;
	}

	// 最后一次读取的数据。此没有缓存到windowbuffer中，故直接输出
	loop_update_range_final2_write:
	for(int para_i = 0; para_i < ALLOCATE_PARAL_NUM; para_i ++)
	{
#pragma HLS UNROLL
		update_registrate_point_cloud_range_stream[para_i].write(range_read_from_stream[para_i]);
		update_registrate_point_cloud_point_stream[para_i].write(point_read_from_stream[para_i]);
	}
	write_count++;

	DEBUG_LOG( "update_range read_count = " << read_count << " write_count = " << write_count);
	DEBUG_GETCHAR;

}

void cache_rangeimage(hls::stream<My_PointXYZI_HW> registrate_point_cloud_stream[ALLOCATE_PARAL_NUM],
					hls::stream<type_range_hw> registrate_point_cloud_range_stream[ALLOCATE_PARAL_NUM],
					hls::stream<My_PointXYZI_HW> cache_image_point_stream[PARAL_NUM],
					hls::stream<type_range_hw> cache_image_range_stream[PARAL_NUM])
{
	int write_count = 0;
	int read_count = 0;
	My_PointXYZI_HW point_read_from_stream[ALLOCATE_PARAL_NUM];
	type_range_hw range_read_from_stream[ALLOCATE_PARAL_NUM];
#pragma HLS array_partition variable=point_read_from_stream complete dim=0
#pragma HLS array_partition variable=range_read_from_stream complete dim=0

	loop_cache_rangeimage_main:
	for(int loop_i = 0; loop_i < N_SCANS*Horizon_SCAN/ALLOCATE_PARAL_NUM; loop_i++)
	{
#pragma HLS PIPELINE

		for(int i = 0; i < ALLOCATE_PARAL_NUM; i ++)
		{
			point_read_from_stream[i] = registrate_point_cloud_stream[i].read();
			range_read_from_stream[i] = registrate_point_cloud_range_stream[i].read();

			// DEBUG_LOG("compute_features input i,j,x,y,z,range: " << point_read_from_stream[i].x << " "
			// 	<< point_read_from_stream[i].y << " " << point_read_from_stream[i].z << " "
			// 	<< point_read_from_stream[i].intensity << " " << range_read_from_stream[i] << " ");
		}
		read_count++;
		for(int batch_i = 0; batch_i < ALLOCATE_PARAL_NUM; batch_i+=PARAL_NUM)
		{
			for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
			{
				cache_image_point_stream[para_i].write(point_read_from_stream[batch_i+para_i]);
				cache_image_range_stream[para_i].write(range_read_from_stream[batch_i+para_i]);
			}
			write_count++;
		}

	}
	DEBUG_LOG( "cache_rangeimage read_count = " << read_count << " write_count = " << write_count);
	DEBUG_GETCHAR;

}

static void compute_curvature(
					hls::stream<My_PointXYZI_HW> registrate_point_cloud_stream[PARAL_NUM],
					hls::stream<type_range_hw> registrate_point_cloud_range_stream[PARAL_NUM],

					hls::stream<My_PointXYZI_HW> curvature_out_point_stream[PARAL_NUM],
					hls::stream<type_range_hw> curvature_out_range_stream[PARAL_NUM],
					hls::stream<type_curvature_hw> curvature_out_curvature_stream[PARAL_NUM],
					hls::stream<type_picked_hw> curvature_out_picked_stream[PARAL_NUM])
{
	int write_count = 0;
	int read_count = 0;
	int process_bias = 11;

	const int PICKED_BUFFER_SIZE = Horizon_SCAN;

	My_PointXYZI_HW rangeimage_line_buffer[N_SCANS][N_buffer];
	type_curvature_hw rangeimage_curvature_buffer[N_SCANS][N_buffer];
	type_picked_hw rangeimage_picked_buffer[N_SCANS][PICKED_BUFFER_SIZE];
	type_range_hw rangeimage_range_buffer[N_SCANS][N_buffer];
#pragma HLS aggregate variable=rangeimage_line_buffer
#pragma HLS array_partition variable=rangeimage_line_buffer cyclic factor=PARAL_NUM dim=1
#pragma HLS array_partition variable=rangeimage_line_buffer factor=N_buffer dim=2
// 用 URAM 的话，为了组成双端口，会用32个URAM, 有点得不偿失。而且速度会变慢。暂时先不用。
// #pragma HLS bind_storage variable=rangeimage_line_buffer type=RAM_T2P impl=URAM 		
#pragma HLS array_partition variable=rangeimage_curvature_buffer cyclic factor=PARAL_NUM dim=1
#pragma HLS array_partition variable=rangeimage_curvature_buffer factor=N_buffer dim=2
#pragma HLS array_partition variable=rangeimage_picked_buffer cyclic factor=PARAL_NUM dim=1
#pragma HLS array_partition variable=rangeimage_picked_buffer cyclic factor=N_buffer dim=2
// #pragma HLS array_partition variable=rangeimage_range_buffer cyclic factor=PARAL_NUM dim=1
// #pragma HLS array_partition variable=rangeimage_range_buffer cyclic factor=4 dim=2
#pragma HLS bind_storage variable=rangeimage_range_buffer type=RAM_1WNR impl=bram

	My_PointXYZI_HW point_read_from_stream[PARAL_NUM];
	type_range_hw range_read_from_stream[PARAL_NUM];
#pragma HLS array_partition variable=point_read_from_stream complete dim=0
#pragma HLS array_partition variable=range_read_from_stream complete dim=0

	type_range_hw depth1, depth2;
	type_range_hw out_range;
	type_picked_hw out_picked;


	My_PointXYZI_HW rangeimage_window_buffer[PARAL_NUM][N_buffer];	// 只是单纯的将数据从linebuffer提取出来，让HLS更好分析。
#pragma HLS array_partition variable=rangeimage_window_buffer complete dim=0

	// 缓存前11列数据。之后每次读取新的数据，处理前面11个数据。
loop_reset_compute_curvature_buffer:
	for(int j = 0; j < process_bias; j++)
	{
		for(int i = 0; i < N_SCANS; i+=PARAL_NUM)
		{
#pragma HLS PIPELINE
#pragma HLS loop_tripcount min=N_SCANS/PARAL_NUM max=N_SCANS/PARAL_NUM
			for(int para_i = 0; para_i < PARAL_NUM; para_i++)
			{
				point_read_from_stream[para_i] = registrate_point_cloud_stream[para_i].read();
				rangeimage_line_buffer[i+para_i][j] = point_read_from_stream[para_i];
				range_read_from_stream[para_i] = registrate_point_cloud_range_stream[para_i].read();
				rangeimage_range_buffer[i+para_i][j] = range_read_from_stream[para_i];

				rangeimage_curvature_buffer[i+para_i][j] = 0;
				rangeimage_picked_buffer[i+para_i][j] = 0;

				// curvature_out_range_stream[para_i].write(range_read_from_stream[para_i]);
			}

		}
		read_count++;
	}

	int current_column_id = process_bias;	// 大循环中第一个要读取的列数目
	int current_line_buffer_id = current_column_id%N_buffer;
	// int current_write_id = -1;
	
	int read_i = 0;
	// int current_process_column_id = 5;	// 初始处理列数是5
loop_compute_curvature_main:
	for(int loop_i = 0; loop_i < (Horizon_SCAN-process_bias)*N_SCANS/PARAL_NUM; loop_i++)
	{
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=Horizon_SCAN*N_SCANS/PARAL_NUM max=Horizon_SCAN*N_SCANS/PARAL_NUM
#pragma HLS dependence variable=rangeimage_picked_buffer intra false
#pragma HLS dependence variable=rangeimage_picked_buffer inter false
// #pragma HLS dependence variable=rangeimage_curvature_buffer intra false
// #pragma HLS dependence variable=rangeimage_curvature_buffer inter false
#pragma HLS dependence variable=rangeimage_line_buffer intra false
#pragma HLS dependence variable=rangeimage_range_buffer intra false



/************************************* 2.1从上一级 stream 中缓存数据 到 line buffer **************************************************************************/
		// DEBUG_LOG( "reading stream...... with current_column_id = " << current_column_id );
		loop_compute_curvature_read:
		for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
		{
#pragma HLS UNROLL
			point_read_from_stream[para_i] = registrate_point_cloud_stream[para_i].read();
			rangeimage_line_buffer[read_i+para_i][current_line_buffer_id] = point_read_from_stream[para_i];
			range_read_from_stream[para_i] = registrate_point_cloud_range_stream[para_i].read();
			rangeimage_range_buffer[read_i+para_i][current_line_buffer_id] = range_read_from_stream[para_i];

			// rangeimage_picked_buffer[read_i+para_i][(current_column_id)] = 0;

			// curvature_out_range_stream[para_i].write(range_read_from_stream[para_i]);

			// window buffer 
			for(int temp_j = 0; temp_j < 11; temp_j++)
			{
				// rangeimage_window_buffer[para_i][temp_j] = rangeimage_line_buffer[read_i+para_i][(current_column_id+1+temp_j)%N_buffer];
				rangeimage_window_buffer[para_i][temp_j] = rangeimage_line_buffer[read_i+para_i][(current_column_id-11+temp_j)%N_buffer];
			}

			// DEBUG_LOG("compute_features input i,j,x,y,z,range: " << read_i+para_i << " " << current_column_id << " " << rangeimage_line_buffer[read_i+para_i][current_line_buffer_id].x << " "
			// 	<< rangeimage_line_buffer[read_i+para_i][current_line_buffer_id].y << " " << rangeimage_line_buffer[read_i+para_i][current_line_buffer_id].z << " "
			// 	<< rangeimage_line_buffer[read_i+para_i][current_line_buffer_id].intensity << " " << rangeimage_line_range_buffer[read_i+para_i][current_line_buffer_id] << " ");
		}

/*********************************************************** 2.2 对于每条线，处理每个column **************************************************************************/
		// current_process_column_id = current_column_id - 5 -1;	// -1 是前一列。 -5 是为了满足曲率计算需要的后五个点会进入
		// int current_process_column_buffer_id = (current_column_id - 6)%N_buffer;
		
		loop_compute_curvature_each_line:
		for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
		{
#pragma HLS UNROLL
#pragma HLS loop_tripcount min=PARAL_NUM max=PARAL_NUM
// #pragma HLS UNROLL
			int scan_i = read_i + para_i; // 读那个scan的数据，就处理其前一 column 数据。

/*********************************************************** 2.2.1 对于每条线，处理曲率 **************************************************************************/

			// todo:
			type_curvature_hw diffX_c = rangeimage_window_buffer[para_i][0].x + rangeimage_window_buffer[para_i][1].x
				+ rangeimage_window_buffer[para_i][2].x + rangeimage_window_buffer[para_i][3].x
				+ rangeimage_window_buffer[para_i][4].x - 10 * rangeimage_window_buffer[para_i][5].x
				+ rangeimage_window_buffer[para_i][6].x + rangeimage_window_buffer[para_i][7].x
				+ rangeimage_window_buffer[para_i][8].x + rangeimage_window_buffer[para_i][9].x
				+ rangeimage_window_buffer[para_i][10].x;
			type_curvature_hw diffY_c = rangeimage_window_buffer[para_i][0].y + rangeimage_window_buffer[para_i][1].y
				+ rangeimage_window_buffer[para_i][2].y + rangeimage_window_buffer[para_i][3].y
				+ rangeimage_window_buffer[para_i][4].y - 10 * rangeimage_window_buffer[para_i][5].y
				+ rangeimage_window_buffer[para_i][6].y + rangeimage_window_buffer[para_i][7].y
				+ rangeimage_window_buffer[para_i][8].y + rangeimage_window_buffer[para_i][9].y
				+ rangeimage_window_buffer[para_i][10].y;
			type_curvature_hw diffZ_c = rangeimage_window_buffer[para_i][0].z + rangeimage_window_buffer[para_i][1].z
				+ rangeimage_window_buffer[para_i][2].z + rangeimage_window_buffer[para_i][3].z
				+ rangeimage_window_buffer[para_i][4].z - 10 * rangeimage_window_buffer[para_i][5].z
				+ rangeimage_window_buffer[para_i][6].z + rangeimage_window_buffer[para_i][7].z
				+ rangeimage_window_buffer[para_i][8].z + rangeimage_window_buffer[para_i][9].z
				+ rangeimage_window_buffer[para_i][10].z;

			// todo:
			type_curvature_hw cloudCurvature_ri = diffX_c * diffX_c + diffY_c * diffY_c + diffZ_c * diffZ_c;

			rangeimage_curvature_buffer[scan_i][(current_column_id - 6)%N_buffer] = cloudCurvature_ri;

			range_io:{
				#pragma HLS protocol mode=floating
				depth1 = rangeimage_range_buffer[scan_i][(current_column_id - 6)%N_buffer];
				depth2 = rangeimage_range_buffer[scan_i][(current_column_id - 5)%N_buffer];
				out_range = rangeimage_range_buffer[scan_i][(current_column_id-12)%N_buffer];
			}

			picked_io:{
				#pragma HLS protocol mode=floating
				rangeimage_picked_buffer[scan_i][(current_column_id)] = 0;
				out_picked = rangeimage_picked_buffer[scan_i][(current_column_id-12)];
				if( depth1 > 2 && depth2 > 2 )
				{
					if (depth1 - depth2 > depth1*(type_range_hw)0.01){
						rangeimage_picked_buffer[scan_i][(current_column_id - 11)] = 1;
						rangeimage_picked_buffer[scan_i][(current_column_id - 10)] = 1;
						rangeimage_picked_buffer[scan_i][(current_column_id - 9)] = 1;
						rangeimage_picked_buffer[scan_i][(current_column_id - 8)] = 1;
						rangeimage_picked_buffer[scan_i][(current_column_id - 7)] = 1;
						rangeimage_picked_buffer[scan_i][(current_column_id - 6)] = 1;
					}else if (depth2 - depth1 > depth1*(type_range_hw)0.01){
						rangeimage_picked_buffer[scan_i][(current_column_id - 6)] = 1;
						rangeimage_picked_buffer[scan_i][(current_column_id - 5)] = 1;
						rangeimage_picked_buffer[scan_i][(current_column_id - 4)] = 1;
						rangeimage_picked_buffer[scan_i][(current_column_id - 3)] = 1;
						rangeimage_picked_buffer[scan_i][(current_column_id - 2)] = 1;
						rangeimage_picked_buffer[scan_i][(current_column_id - 1)] = 1;
						// rangeimage_picked_buffer[scan_i][(current_column_id )] = 1;
					}
				}
			}

			/*********************************************************** 2.3 输出处理完毕的数据 **************************************************************************/
			if(current_column_id >= 12)
			{
				int current_write_buffer_id = (current_column_id-12)%N_buffer;

				curvature_out_point_stream[para_i].write(rangeimage_line_buffer[scan_i][current_write_buffer_id]);
				curvature_out_range_stream[para_i].write(out_range);
				curvature_out_curvature_stream[para_i].write(rangeimage_curvature_buffer[scan_i][current_write_buffer_id]);
				curvature_out_picked_stream[para_i].write(out_picked);
			
			}
		}// for each scan_i

/***************************************************** 2.4 为下次读取数据做准备 ******************************************************************/

		if((read_i + PARAL_NUM) >= N_SCANS)
		{
			read_i = 0;
			read_count++;
			if(current_column_id >= 12)
				write_count++;

			current_column_id ++;
			if( (current_line_buffer_id+1) == N_buffer)
				current_line_buffer_id = 0;
			else
				current_line_buffer_id = current_line_buffer_id + 1;

		}
		else
		{
			read_i = read_i + PARAL_NUM;
		}

	}	// while read flag

	// DEBUG_LOG("compute_curvature OUT LOOP with current_column_id: " << current_column_id << " current_write_id = " << current_write_id << " read_i = " << read_i << " write_count = " << write_count
	//  << " read_count = " << read_count);
	 DEBUG_GETCHAR;

	// 输出 rangeimage_line_buffer 中的所有信息。
	loop_out_final_compute_curvature_stream:
	for(int column_i = (current_column_id-10-1-1); column_i < Horizon_SCAN; column_i++)
	{
#pragma HLS loop_tripcount min=10 max=10
		loop_out_final_scan_compute_curvature_stream:
		for(int out_i = 0; out_i < N_SCANS; out_i+=PARAL_NUM)
		{
#pragma HLS PIPELINE
			for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
			{
#pragma HLS UNROLL
				int current_write_buffer_id = column_i%N_buffer;

				curvature_out_point_stream[para_i].write(rangeimage_line_buffer[out_i + para_i][current_write_buffer_id]);
				curvature_out_range_stream[para_i].write(rangeimage_range_buffer[out_i + para_i][current_write_buffer_id]);
				curvature_out_picked_stream[para_i].write(rangeimage_picked_buffer[out_i + para_i][current_write_buffer_id]);
				
				if(current_write_buffer_id < Horizon_SCAN-6)
					curvature_out_curvature_stream[para_i].write(rangeimage_curvature_buffer[out_i + para_i][current_write_buffer_id]);
				else
					curvature_out_curvature_stream[para_i].write(curvature_zero);

			}
		}
		write_count++;
	}
	DEBUG_LOG( "compute_curvature read_count = " << read_count << " write_count = " << write_count);
	DEBUG_GETCHAR;
}

static void compute_ground(
					hls::stream<My_PointXYZI_HW> curvature_out_point_stream[PARAL_NUM],
					hls::stream<type_range_hw> curvature_out_range_stream[PARAL_NUM],
					hls::stream<type_curvature_hw> curvature_out_curvature_stream[PARAL_NUM],
					hls::stream<type_picked_hw> curvature_out_picked_stream[PARAL_NUM],

					hls::stream<My_PointXYZI_HW> ground_out_point_stream[PARAL_NUM],
					hls::stream<type_range_hw> ground_out_range_stream[PARAL_NUM],
					hls::stream<type_curvature_hw> ground_out_curvature_stream[PARAL_NUM],
					hls::stream<type_picked_hw> ground_out_picked_stream[PARAL_NUM],
					hls::stream<type_sortind_hw> ground_out_sortind_stream[PARAL_NUM],
					hls::stream<type_ground_hw> ground_out_ground_stream[PARAL_NUM]
)
{
	int write_count = 0;
	int read_count = 0;
	// 寻找地面
	int ground_point_count = 0.0;
	// const int AVERAGE_GROUND_SIZE = 2;
	My_Ground_HW ground_average_height[N_SCANS];
	type_point_hw sensorMountAngle = 0.0;
	int process_bias = 1;


	for(int i = 0; i < N_SCANS; i++)
	{
		ground_average_height[i].height = -lidar_height_hw;
		ground_average_height[i].count = 0;
	}


	const int N_ground_buffer = 4;	// 一个读，一个处理，前三个来计算局部平均地面高度
	const int ground_window_buffer_length = 6+PARAL_NUM;	// 用来计算局部平均地面高度
	const type_curvature_hw DEFAULT_WINDOW_GROUND = -99.9;		// 默认的地面数据
	type_curvature_hw rangeimage_ground_height_window_buffer[ground_window_buffer_length][3];	// 存储处理点前三列的ground的z数据。只有ground=1才有数据，否则为0。 0表示不是地面点。
#pragma HLS array_partition variable=rangeimage_ground_height_window_buffer complete dim=0

	type_ground_hw rangeimage_ground_buffer[N_SCANS][N_ground_buffer];
#pragma HLS array_partition variable=rangeimage_ground_buffer cyclic factor=ground_window_buffer_length dim=1
#pragma HLS array_partition variable=rangeimage_ground_buffer complete dim=2

	My_PointXYZI_HW point_read_from_stream[PARAL_NUM];
	type_range_hw range_read_from_stream[PARAL_NUM];
	type_curvature_hw curvature_read_from_stream[PARAL_NUM];
	type_picked_hw picked_read_from_stream[PARAL_NUM];
	type_sortind_hw this_sortind[PARAL_NUM];
#pragma HLS array_partition variable=point_read_from_stream complete dim=0
#pragma HLS array_partition variable=range_read_from_stream complete dim=0
#pragma HLS array_partition variable=curvature_read_from_stream complete dim=0
#pragma HLS array_partition variable=picked_read_from_stream complete dim=0

	My_PointXYZI_HW last_point[PARAL_NUM];
	type_range_hw last_range[PARAL_NUM];
	type_curvature_hw last_curvature[PARAL_NUM];
	type_picked_hw last_picked[PARAL_NUM];
	type_sortind_hw last_sortind[PARAL_NUM];

	int current_column_id = 0;	// 当前读取数据的 column id
	int read_i = 0;
	int current_line_buffer_id = 0;

loop_compute_ground_main:
	for(int loop_i = 0; loop_i < (Horizon_SCAN)*N_SCANS/PARAL_NUM; loop_i ++)	// 读 (Horizon_SCAN-1) 列，读完所有数据
	{
#pragma HLS PIPELINE II=1
// #pragma HLS loop_tripcount min=Horizon_SCAN*N_SCANS/PARAL_NUM max=Horizon_SCAN*N_SCANS/PARAL_NUM
#pragma HLS dependence variable=ground_average_height intra false
#pragma HLS dependence variable=ground_average_height inter false	// 这个必须得加上，要不然II会很大。

		loop_compute_ground_read:
		for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
		{
#pragma HLS UNROLL
			point_read_from_stream[para_i] = curvature_out_point_stream[para_i].read();
			range_read_from_stream[para_i] = curvature_out_range_stream[para_i].read();
			curvature_read_from_stream[para_i] = curvature_out_curvature_stream[para_i].read();
			picked_read_from_stream[para_i] = curvature_out_picked_stream[para_i].read();
			this_sortind[para_i] = current_column_id;
			read_count++;
		}

		for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
		{
			int scan_i = read_i + para_i; // 读那个scan的数据，就处理其前一 column 数据。
			type_ground_hw current_point_ground = 0;

			My_PointXYZI_HW this_point, nextline_point;
			this_point.x = last_point[para_i].x;
			this_point.y = last_point[para_i].y;
			this_point.z = last_point[para_i].z;
			nextline_point.x = point_read_from_stream[para_i].x;
			nextline_point.y = point_read_from_stream[para_i].y;
			nextline_point.z = point_read_from_stream[para_i].z;

			// 由上下两线之间点的XYZ位置得到两线之间的俯仰角
			// 如果俯仰角在10度以内，则判定(scan_i,current_process_column_buffer_id)为地面点,groundMat[scan_i][current_process_column_buffer_id]=1
			// 否则，则不是地面点，进行后续操作
			type_point_hw diffX = nextline_point.x - this_point.x;
			type_point_hw diffY = nextline_point.y - this_point.y;
			type_point_hw diffZ = nextline_point.z - this_point.z;

			// todo:
			type_temp_hw temp_dist = diffX*diffX + diffY*diffY;
			// type_angle_hw angle_tan_value = diffZ / my_sqrt(temp_dist);

			type_temp_hw temp_value;

		#ifdef USE_FLOAT
			float angle_tan_value;
		#else
			type_range_hw angle_tan_value;
		#endif

			if(temp_dist > 0)
			{
				temp_value = my_sqrt(temp_dist);
				angle_tan_value = diffZ / temp_value;
			}
			else
				angle_tan_value = 10000000;

#ifdef LINUX_RUN	// 如果是csim或者gcc编译，就跑float版。否则就上hls版
			float angle_tan_value_float = angle_tan_value;
			float atan_angle = atan(angle_tan_value_float);
			type_angle_hw back_atan_angle = atan_angle;
			type_angle_hw angle = back_atan_angle * 180 / M_PI_HW; //得到竖直角，也可以说是俯仰角
#else
			type_angle_hw angle = hls::atan(angle_tan_value) * 180 / M_PI_HW;	// 计算前后两个角度的倾角
#endif

			if( (scan_i > 0 || scan_i < groundScanInd) && last_range[para_i]  >= 0 )
			{
				if (my_fabs(angle - sensorMountAngle) <= ground_min_angle_hw)
				{
					if(current_column_id < 3 || scan_i <= 4)	// 前三条线或者地面点小于 Horizon_SCAN 时的处理
					{
						if( my_fabs(this_point.z + lidar_height_hw) < ground_global_height_threshold_hw)
							current_point_ground = 1;
					}
					else
					{
						if(last_range[para_i] == -2)
						{
							if( my_fabs(this_point.z + lidar_height_hw) < ground_global_height_threshold_hw)
								current_point_ground = 1;
						}
						else
						{
							if( my_fabs(this_point.z - ground_average_height[scan_i].height) < ground_global_height_threshold_hw )
							{
								current_point_ground = 1;
							}
						}
					}
				}
			}

			// 加入垂直平面点	// 去掉这一步 平面点从 752 降到 559
			if(scan_i < (N_SCANS-1))
			{
				// 如果垂直角度大于 60 度，而且曲率很小，那么就认为是垂直平面点。
				if(my_fabs(angle - sensorMountAngle) >= vertical_plane_min_angle_hw && last_curvature[para_i] <= 5)
				{
					current_point_ground = 2;
				}
			}

			if(current_point_ground == 1)
			{
				// 这一步结果貌似跟之前的不一样。
				type_curvature_hw temp_ground_height = (ground_average_height[scan_i].height*ground_average_height[scan_i].count + this_point.z) / (ground_average_height[scan_i].count +1);
				// std::cout << " ground_point_count: " << ground_point_count << " temp_ground_height: " << temp_ground_height << " this_point.z: " << this_point.z  << std::endl;
				ground_average_height[scan_i].height = temp_ground_height;
				ground_average_height[scan_i].count = ground_average_height[scan_i].count + 1;
			}


			rangeimage_ground_buffer[scan_i][current_line_buffer_id] = current_point_ground;

			if(current_column_id > 0 || scan_i > 0)
			{
				// output the data
				ground_out_point_stream[para_i].write(last_point[para_i]);
				ground_out_range_stream[para_i].write(last_range[para_i]);
				ground_out_curvature_stream[para_i].write(last_curvature[para_i]);
				ground_out_picked_stream[para_i].write(last_picked[para_i]);
				ground_out_sortind_stream[para_i].write(last_sortind[para_i]);
				ground_out_ground_stream[para_i].write(current_point_ground);
				write_count++;
			}

			// 当前值赋给last
			last_point[para_i] = point_read_from_stream[para_i];
			last_range[para_i] = range_read_from_stream[para_i];
			last_curvature[para_i] = curvature_read_from_stream[para_i];
			last_picked[para_i] = picked_read_from_stream[para_i];
			last_sortind[para_i] = this_sortind[para_i];
		}	

		if((read_i + PARAL_NUM) >= N_SCANS)
		{
			read_i = 0;

			current_column_id ++;

			if( (current_line_buffer_id + 1) == N_ground_buffer)
				current_line_buffer_id = 0;
			else
				current_line_buffer_id ++;
		}
		else
		{
			read_i = read_i + PARAL_NUM;
		}
	}

	// output the data
	for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
	{
		ground_out_point_stream[para_i].write(last_point[para_i]);
		ground_out_range_stream[para_i].write(last_range[para_i]);
		ground_out_curvature_stream[para_i].write(last_curvature[para_i]);
		ground_out_picked_stream[para_i].write(last_picked[para_i]);
		ground_out_sortind_stream[para_i].write(last_sortind[para_i]);
		ground_out_ground_stream[para_i].write(ground_zero);
		write_count++;
	}
	

	DEBUG_LOG( "compute_ground read_count = " << read_count << " write_count = " << write_count);
	DEBUG_GETCHAR;
}

static void integrate_features(
					hls::stream<My_PointXYZI_HW> ground_out_point_stream[PARAL_NUM],
					hls::stream<type_range_hw> ground_out_range_stream[PARAL_NUM],
					hls::stream<type_curvature_hw> ground_out_curvature_stream[PARAL_NUM],
					hls::stream<type_picked_hw> ground_out_picked_stream[PARAL_NUM],
					hls::stream<type_sortind_hw> ground_out_sortind_stream[PARAL_NUM],
					hls::stream<type_ground_hw> ground_out_ground_stream[PARAL_NUM],

					hls::stream<My_PointXYZI_HW> registrate_point_cloud_point_stream[PARAL_NUM],
					hls::stream<type_range_hw> registrate_point_cloud_point_range_stream[PARAL_NUM],
					hls::stream<My_PointFeature_hw> registrate_point_cloud_feature_stream[PARAL_NUM])
{
	int write_count = 0;
	int read_count = 0;

	My_PointXYZI_HW point_read_from_stream[PARAL_NUM];
	type_range_hw range_read_from_stream[PARAL_NUM];
	type_curvature_hw curvature_read_from_stream[PARAL_NUM];
	type_picked_hw picked_read_from_stream[PARAL_NUM];
	type_sortind_hw sortind_read_from_stream[PARAL_NUM];
	type_ground_hw ground_read_from_stream[PARAL_NUM];
#pragma HLS array_partition variable=point_read_from_stream complete dim=0
#pragma HLS array_partition variable=range_read_from_stream complete dim=0
#pragma HLS array_partition variable=curvature_read_from_stream complete dim=0
#pragma HLS array_partition variable=picked_read_from_stream complete dim=0
#pragma HLS array_partition variable=sortind_read_from_stream complete dim=0
#pragma HLS array_partition variable=ground_read_from_stream complete dim=0

	My_PointFeature_hw integrate_features[PARAL_NUM];
#pragma HLS array_partition variable=integrate_features complete dim=0

loop_integrate_features_main:
	for(int loop_i = 0; loop_i < Horizon_SCAN*N_SCANS/PARAL_NUM; loop_i ++)
	{
#pragma HLS PIPELINE
#pragma HLS loop_tripcount min=Horizon_SCAN*N_SCANS/PARAL_NUM max=Horizon_SCAN*N_SCANS/PARAL_NUM
		for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
		{
#pragma HLS UNROLL
			point_read_from_stream[para_i] = ground_out_point_stream[para_i].read();
			range_read_from_stream[para_i] = ground_out_range_stream[para_i].read();
			curvature_read_from_stream[para_i] = ground_out_curvature_stream[para_i].read();
			picked_read_from_stream[para_i] = ground_out_picked_stream[para_i].read();
			sortind_read_from_stream[para_i] = ground_out_sortind_stream[para_i].read();
			ground_read_from_stream[para_i] = ground_out_ground_stream[para_i].read();

			integrate_features[para_i].curvature = curvature_read_from_stream[para_i];
			integrate_features[para_i].picked = picked_read_from_stream[para_i];
			integrate_features[para_i].ground = ground_read_from_stream[para_i];
			integrate_features[para_i].SortInd = sortind_read_from_stream[para_i];

			registrate_point_cloud_point_stream[para_i].write(point_read_from_stream[para_i]);
			registrate_point_cloud_point_range_stream[para_i].write(range_read_from_stream[para_i]);
			registrate_point_cloud_feature_stream[para_i].write(integrate_features[para_i]);
		}
		read_count++;
		write_count++;
	}
}


// 1. 输入每条线上单个点，输出每条线上每个点的属性到片外。。输出需要300个周期。
// 输出这块改成 less_flat 点，flat 点， less_sharp 和 sharp. 他们的 scanid， columnid, .
static void extract_features(
					hls::stream<My_PointXYZI_HW> registrate_point_cloud_point_stream[PARAL_NUM],
					hls::stream<type_range_hw> registrate_point_cloud_point_range_stream[PARAL_NUM],
					hls::stream<My_PointFeature_hw> registrate_point_cloud_feature_stream[PARAL_NUM],

					hls::stream<My_PointOutFeatureLocation> out_sharp_stream[PARAL_NUM*SHARP_SIZE],
					hls::stream<My_PointOutFeatureLocation> out_lesssharp_stream[PARAL_NUM*LESS_SHARP_SIZE],
					hls::stream<My_PointOutFeatureLocation> out_flat_stream[PARAL_NUM*FLAT_SIZE],
					hls::stream<type_picked_hw> out_lessflat_stream[PARAL_NUM],
					hls::stream<type_ground_hw> out_ground_stream[PARAL_NUM],
					hls::stream<My_PointXYZI_HW> registrate_point_cloud_outpoint_stream[PARAL_NUM],
					hls::stream<type_range_hw> registrate_point_cloud_outrange_stream[PARAL_NUM]
)
{
	int read_count = 0;
	int point_write_count = 0;
	int feature_write_count = 0;

	My_PointXYZI_HW feature_point_cloud_point_buffer[N_SCANS][N_feature_buffer];
	type_range_hw feature_point_cloud_range_buffer[N_SCANS][N_feature_buffer];
	My_PointFeature_hw feature_point_cloud_feature_buffer[N_SCANS][N_feature_buffer];
#pragma HLS array_partition variable=feature_point_cloud_point_buffer cyclic factor=PARAL_NUM dim=1
#pragma HLS array_partition variable=feature_point_cloud_point_buffer complete dim=2
#pragma HLS array_partition variable=feature_point_cloud_range_buffer cyclic factor=PARAL_NUM dim=1
#pragma HLS array_partition variable=feature_point_cloud_range_buffer complete dim=2
#pragma HLS array_partition variable=feature_point_cloud_feature_buffer cyclic factor=PARAL_NUM dim=1
#pragma HLS array_partition variable=feature_point_cloud_feature_buffer complete dim=2

	My_PointXYZI_HW registrate_point_cloud_point_read_from_stream[PARAL_NUM];
	type_range_hw range_read_from_stream[PARAL_NUM];
	My_PointFeature_hw registrate_point_cloud_feature_read_from_stream[PARAL_NUM];
#pragma HLS array_partition variable=registrate_point_cloud_point_read_from_stream complete dim=0
#pragma HLS array_partition variable=range_read_from_stream complete dim=0
#pragma HLS array_partition variable=registrate_point_cloud_feature_read_from_stream complete dim=0

	// less_flat 点直接输出。 flat 点，sharp 点，以及less_sharp 点 每300 输出一次。
	My_PointOutFeatureLocation sharp_outfeature_buffer[N_SCANS][LESS_SHARP_SIZE];
	My_PointOutFeatureLocation flat_outfeature_buffer[N_SCANS][FLAT_SIZE];
#pragma HLS array_partition variable=sharp_outfeature_buffer cyclic factor=PARAL_NUM dim=1
#pragma HLS array_partition variable=sharp_outfeature_buffer complete dim=2
#pragma HLS array_partition variable=flat_outfeature_buffer cyclic factor=PARAL_NUM dim=1
#pragma HLS array_partition variable=flat_outfeature_buffer complete dim=2

	// 用于判断比较 curvature 和 sortind
	My_PointSortFeature_hw less_sharp_points[N_SCANS][LESS_SHARP_SIZE];
	My_PointSortFeature_hw less_flat_points[N_SCANS][FLAT_SIZE];
#pragma HLS array_partition variable=less_sharp_points cyclic factor=PARAL_NUM dim=1
#pragma HLS array_partition variable=less_sharp_points complete dim=2
#pragma HLS array_partition variable=less_flat_points cyclic factor=PARAL_NUM dim=1
#pragma HLS array_partition variable=less_flat_points complete dim=2

	for(int i = 0; i < N_SCANS; i+= PARAL_NUM)
	{
#pragma HLS PIPELINE
		for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
		{
			registrate_point_cloud_point_read_from_stream[para_i] = registrate_point_cloud_point_stream[para_i].read();
			feature_point_cloud_point_buffer[i + para_i][0] = registrate_point_cloud_point_read_from_stream[para_i];
			registrate_point_cloud_feature_read_from_stream[para_i] = registrate_point_cloud_feature_stream[para_i].read();
			feature_point_cloud_feature_buffer[i + para_i][0] = registrate_point_cloud_feature_read_from_stream[para_i];
			range_read_from_stream[para_i] = registrate_point_cloud_point_range_stream[para_i].read();
			feature_point_cloud_range_buffer[i + para_i][0] = range_read_from_stream[para_i];

			registrate_point_cloud_outpoint_stream[para_i].write(registrate_point_cloud_point_read_from_stream[para_i]);
			registrate_point_cloud_outrange_stream[para_i].write(range_read_from_stream[para_i]);

			// 输出 less_flat 点
			type_picked_hw temp_out_lessflat;
			if (registrate_point_cloud_feature_read_from_stream[para_i].picked == 0 && range_read_from_stream[para_i] > 2 && registrate_point_cloud_feature_read_from_stream[para_i].curvature < 0.1
			&& ( (registrate_point_cloud_feature_read_from_stream[para_i].ground == 1) || (registrate_point_cloud_feature_read_from_stream[para_i].ground == 2) ) )
			{
				temp_out_lessflat = picked_one;
			}
			else
			{
				temp_out_lessflat = picked_zero;
			}
			out_lessflat_stream[para_i].write(temp_out_lessflat);
			// 输出地面点
			type_ground_hw temp_out_ground;
			if ( registrate_point_cloud_feature_read_from_stream[para_i].ground == 1 )
			{
				temp_out_ground = ground_one;
			}
			else
			{
				temp_out_ground = ground_zero;
			}
			out_ground_stream[para_i].write(temp_out_ground);
		}
		read_count++;
		point_write_count++;
	}

	int current_column_id = 1;	// 第 0 行已经被读取了
	int current_line_buffer_id = current_column_id;
	int current_process_column_id = current_column_id-1;
	int current_process_colum_buffer_id = current_process_column_id % N_feature_buffer;
	int read_i = 0;

	for(int i = 0; i < N_SCANS; i++)
	{
#pragma HLS PIPELINE
		for(int m = 0; m < LESS_SHARP_SIZE; m++)
		{
#pragma HLS UNROLL
			less_sharp_points[i][m].curvature = -1;
			less_sharp_points[i][m].SortInd = -5;
			sharp_outfeature_buffer[i][m].scanId = 0;
			sharp_outfeature_buffer[i][m].columnId = 0;
		}
	}
	for(int i = 0; i < N_SCANS; i++)
	{
#pragma HLS PIPELINE
		for(int m = 0; m < FLAT_SIZE; m++)
		{
#pragma HLS UNROLL
			less_flat_points[i][m].curvature = 1000;
			less_flat_points[i][m].SortInd = -5;
			flat_outfeature_buffer[i][m].scanId = 0;
			flat_outfeature_buffer[i][m].columnId = 0;
		}
	}

	int sp = 5; 
	int ep = sp+Horizon_SCAN/EXTRACT_SEGMENT;
	
loop_extract_features_main:
	for(int loop_i = 0; loop_i < N_SCANS*(Horizon_SCAN-1)/PARAL_NUM; loop_i++)
	{
#pragma HLS loop_tripcount min=N_SCANS*(Horizon_SCAN-1)/PARAL_NUM max=N_SCANS*(Horizon_SCAN-1)/PARAL_NUM
#pragma HLS PIPELINE
#pragma HLS dependence variable=less_flat_points intra false
#pragma HLS dependence variable=flat_outfeature_buffer intra false
#pragma HLS dependence variable=less_sharp_points intra false
#pragma HLS dependence variable=sharp_outfeature_buffer intra false
#pragma HLS dependence variable=less_flat_points inter RAW distance=N_SCANS true
#pragma HLS dependence variable=flat_outfeature_buffer inter RAW distance=N_SCANS true
#pragma HLS dependence variable=less_sharp_points inter RAW distance=N_SCANS true
#pragma HLS dependence variable=sharp_outfeature_buffer inter RAW distance=N_SCANS true

		// 读取新数据，但同时处理上一列数据
		for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
		{
			registrate_point_cloud_point_read_from_stream[para_i] = registrate_point_cloud_point_stream[para_i].read();
			feature_point_cloud_point_buffer[read_i + para_i][current_line_buffer_id] = registrate_point_cloud_point_read_from_stream[para_i];
			registrate_point_cloud_feature_read_from_stream[para_i] = registrate_point_cloud_feature_stream[para_i].read();
			feature_point_cloud_feature_buffer[read_i + para_i][current_line_buffer_id] = registrate_point_cloud_feature_read_from_stream[para_i];
			range_read_from_stream[para_i] = registrate_point_cloud_point_range_stream[para_i].read();
			feature_point_cloud_range_buffer[read_i + para_i][current_line_buffer_id] = range_read_from_stream[para_i];

			registrate_point_cloud_outpoint_stream[para_i].write(registrate_point_cloud_point_read_from_stream[para_i]);
			registrate_point_cloud_outrange_stream[para_i].write(range_read_from_stream[para_i]);

			// 输出 less_flat 点
			type_picked_hw temp_out_lessflat;
			if (registrate_point_cloud_feature_read_from_stream[para_i].picked == 0 && range_read_from_stream[para_i] > 2 && registrate_point_cloud_feature_read_from_stream[para_i].curvature < 0.1
			&& ( (registrate_point_cloud_feature_read_from_stream[para_i].ground == 1) || (registrate_point_cloud_feature_read_from_stream[para_i].ground == 2) ) )
			{
				temp_out_lessflat = picked_one;
			}
			else
			{
				temp_out_lessflat = picked_zero;
			}
			out_lessflat_stream[para_i].write(temp_out_lessflat);
			// 输出地面点
			type_ground_hw temp_out_ground;
			if ( registrate_point_cloud_feature_read_from_stream[para_i].ground == 1 )
			{
				temp_out_ground = ground_one;
			}
			else
			{
				temp_out_ground = ground_zero;
			}
			out_ground_stream[para_i].write(temp_out_ground);

			// DEBUG_LOG("i,j,x,y,z,range.cur,pick,ground,sortid: " << i << " "
			// << current_line_buffer_id << " " << feature_point_cloud_point_buffer[i][current_line_buffer_id].x << " "
			// << feature_point_cloud_point_buffer[i][current_line_buffer_id].y << " " << feature_point_cloud_point_buffer[i][current_line_buffer_id].z << " "
			// << feature_point_cloud_range_buffer[i][current_line_buffer_id] << " "
			// << feature_point_cloud_feature_buffer[i][current_line_buffer_id].curvature << " " << feature_point_cloud_feature_buffer[i][current_line_buffer_id].picked << " "
			// << feature_point_cloud_feature_buffer[i][current_line_buffer_id].ground << " " << feature_point_cloud_feature_buffer[i][current_line_buffer_id].SortInd << " ");
		}
		read_count++;
		point_write_count++;

		// 读完数据开始处理
		// 对于每条线，处理每个column
		// 1. 判断点是否是待处理范围内的，sp到ep, 是的话则进入处理，存入buffer
		// 2. 处理sp,ep的变化,如果发生变化，则读取buffer中的数据，流式输出300个结果，并清空buffer.
		
		if(current_process_column_id >= 5 &&  current_process_column_id < Horizon_SCAN-6)
		{

			// 如果处理的点到了ep或者到了 Horizon_SCAN-6-1，那就输出到下一级
			if(current_process_column_id == (4 + 1*Horizon_SCAN/EXTRACT_SEGMENT) 
			|| current_process_column_id == (4 + 2*Horizon_SCAN/EXTRACT_SEGMENT) 
			|| current_process_column_id == (4 + 3*Horizon_SCAN/EXTRACT_SEGMENT)
			|| current_process_column_id == (4 + 4*Horizon_SCAN/EXTRACT_SEGMENT)
			|| current_process_column_id == (4 + 5*Horizon_SCAN/EXTRACT_SEGMENT)
			|| current_process_column_id == (Horizon_SCAN-6-1) )
			{
				// 1. 输出点云
				for(int para_i = 0; para_i < PARAL_NUM; para_i++)
				{
					for(int m = 0; m < LESS_SHARP_SIZE; m++)
					{
#pragma HLS UNROLL
						if(m < SHARP_SIZE)
						{
							out_sharp_stream[para_i*SHARP_SIZE+m].write(sharp_outfeature_buffer[read_i+para_i][m]);
							out_lesssharp_stream[para_i*LESS_SHARP_SIZE+m].write(sharp_outfeature_buffer[read_i+para_i][m]);
						}
						else
						{
							out_lesssharp_stream[para_i*LESS_SHARP_SIZE+m].write(sharp_outfeature_buffer[read_i+para_i][m]);
						}
					}

					for(int m = 0; m < FLAT_SIZE; m++)
					{
#pragma HLS UNROLL
						out_flat_stream[para_i*FLAT_SIZE+m].write(flat_outfeature_buffer[read_i+para_i][m]);
					}
				}
				feature_write_count++;

				// 2. 重置buffer
				for(int para_i = 0; para_i < PARAL_NUM; para_i++)
				{
#pragma HLS UNROLL
					for(int m = 0; m < LESS_SHARP_SIZE; m++)
					{
						less_sharp_points[read_i+para_i][m].curvature = -1;
						less_sharp_points[read_i+para_i][m].SortInd = -5;
						sharp_outfeature_buffer[read_i+para_i][m].scanId = 0;
						sharp_outfeature_buffer[read_i+para_i][m].columnId = 0;
					}
				}
				for(int para_i = 0; para_i < PARAL_NUM; para_i++)
				{
#pragma HLS UNROLL
					for(int m = 0; m < FLAT_SIZE; m++)
					{
						less_flat_points[read_i+para_i][m].curvature = 1000;
						less_flat_points[read_i+para_i][m].SortInd = -5;
						flat_outfeature_buffer[read_i+para_i][m].scanId = 0;
						flat_outfeature_buffer[read_i+para_i][m].columnId = 0;
					}
				}
			}
			else // 否则处理。。
			{
				for(int para_i = 0; para_i < PARAL_NUM; para_i++)
				{
					type_range_hw in_point_range = feature_point_cloud_range_buffer[read_i + para_i][current_process_colum_buffer_id];
					My_PointFeature_hw in_point_Feature = feature_point_cloud_feature_buffer[read_i + para_i][current_process_colum_buffer_id];
					ap_uint<1> continue_next_iter = 1;

					type_curvature_hw curvature_threshold = 1;
					type_curvature_hw max_threshold = 1;
					if(in_point_range >= 2 && in_point_range < 20) curvature_threshold = max_threshold;
					else if(in_point_range >= 20 && in_point_range < 50) curvature_threshold = max_threshold/2;
					else if(in_point_range >= 50 && in_point_range < 80) curvature_threshold = max_threshold/4;
					else if(in_point_range >= 80 && in_point_range < 120) curvature_threshold = max_threshold/8;

					// 检测特征角点
					if (in_point_Feature.picked == 0 && in_point_range > 1 && in_point_Feature.curvature > curvature_threshold && in_point_Feature.ground != 1 && in_point_Feature.ground != 2)
					{
						int smaller_in_range_point_index = LESS_SHARP_SIZE;
						ap_uint<1> cmp_flag[LESS_SHARP_SIZE];	// 0 表示不需要被替换，true 表示需要被替换
						ap_uint<1> insert_flag[LESS_SHARP_SIZE];
						ap_uint<1> insert_flag_sum = 0;
						ap_uint<1> is_insert = 1;
						// 1. 得到 smaller_in_range_point_index 以及 与buffer比较结果
						for(int m = 0; m < LESS_SHARP_SIZE; m++)
						{
	#pragma HLS UNROLL
							// std::cout << less_sharp_points[m].curvature << " ";
							if(in_point_Feature.curvature > less_sharp_points[read_i + para_i][m].curvature)
								cmp_flag[m] = 1;
							else
								cmp_flag[m] = 0;

							if(my_abs(in_point_Feature.SortInd - less_sharp_points[read_i + para_i][m].SortInd) > 5)
								insert_flag[m] = 0;
							else
							{
								insert_flag[m] = 1;
								insert_flag_sum = 1;
							}
						}

						if(insert_flag_sum == 1)
						{
							for(int m = 0; m < LESS_SHARP_SIZE; m++)
							{
								if(insert_flag[m] == 1 && cmp_flag[m] == 1)
								{
									less_sharp_points[read_i + para_i][m].curvature = in_point_Feature.curvature;	// 无论是插入，还是替换，最后都需要把新数据插入。
									less_sharp_points[read_i + para_i][m].SortInd = in_point_Feature.SortInd;
									sharp_outfeature_buffer[read_i + para_i][m].scanId = read_i + para_i;
									sharp_outfeature_buffer[read_i + para_i][m].columnId = current_process_column_id;
								}
							}
						}
						else
						{
							for(int m = LESS_SHARP_SIZE-1; m > 0; m--)
							{
								if(cmp_flag[m] == 1 && cmp_flag[m-1] == 1)
								{
									less_sharp_points[read_i + para_i][m] = less_sharp_points[read_i + para_i][m-1];
									sharp_outfeature_buffer[read_i + para_i][m].scanId = sharp_outfeature_buffer[read_i + para_i][m-1].scanId;
									sharp_outfeature_buffer[read_i + para_i][m].columnId = sharp_outfeature_buffer[read_i + para_i][m-1].columnId;
								}
								else if(cmp_flag[m] == 1 && cmp_flag[m-1] == 0)
								{
									less_sharp_points[read_i+para_i][m].curvature = in_point_Feature.curvature;
									less_sharp_points[read_i + para_i][m].SortInd = in_point_Feature.SortInd;
									sharp_outfeature_buffer[read_i + para_i][m].scanId = read_i + para_i;
									sharp_outfeature_buffer[read_i + para_i][m].columnId = current_process_column_id;
								}
							}
							if(cmp_flag[0] == 1)
							{
								less_sharp_points[read_i+para_i][0].curvature = in_point_Feature.curvature;
								less_sharp_points[read_i + para_i][0].SortInd = in_point_Feature.SortInd;
								sharp_outfeature_buffer[read_i + para_i][0].scanId = read_i + para_i;
								sharp_outfeature_buffer[read_i + para_i][0].columnId = current_process_column_id;
							}
						}
					} // 检测特征角点

					// 找最小点 flat
					//如果曲率的确比较小，并且未被筛选出
					// 全部使用地面点 这是最稳定的平面点。
					// 降到0.001,地面点的个数几乎还是不变。。。
					if (in_point_Feature.picked == 0 && in_point_range > 1 && in_point_Feature.curvature < 0.1 && ( in_point_Feature.ground == 1 ||  in_point_Feature.ground == 2 ))
					{
						int smaller_in_range_point_index = FLAT_SIZE;
						ap_uint<1> cmp_flag[FLAT_SIZE];	// 0 表示不需要被替换，true 表示需要被替换
						ap_uint<1> insert_flag[FLAT_SIZE];
						ap_uint<1> insert_flag_sum = 0;
						ap_uint<1> is_insert = 1;
						// 1. 得到 smaller_in_range_point_index 以及 与buffer比较结果
						for(int m = 0; m < FLAT_SIZE; m++)
						{
	#pragma HLS UNROLL
							// std::cout << less_flat_points[m].curvature << " ";
							if(in_point_Feature.curvature < less_flat_points[read_i + para_i][m].curvature)
								cmp_flag[m] = 1;
							else
								cmp_flag[m] = 0;

							if(my_abs(in_point_Feature.SortInd - less_flat_points[read_i + para_i][m].SortInd) > 5)
								insert_flag[m] = 0;
							else
							{
								insert_flag[m] = 1;
								insert_flag_sum = 1;
							}
						}

						if(insert_flag_sum == 1)
						{
							for(int m = 0; m < FLAT_SIZE; m++)
							{
								if(insert_flag[m] == 1 && cmp_flag[m] == 1)
								{
									less_flat_points[read_i + para_i][m].curvature = in_point_Feature.curvature;	// 无论是插入，还是替换，最后都需要把新数据插入。
									less_flat_points[read_i + para_i][m].SortInd = in_point_Feature.SortInd;
									flat_outfeature_buffer[read_i + para_i][m].scanId = read_i + para_i;
									flat_outfeature_buffer[read_i + para_i][m].columnId = current_process_column_id;
								}
							}
						}
						else
						{
							for(int m = FLAT_SIZE-1; m > 0; m--)
							{
								if(cmp_flag[m] == 1 && cmp_flag[m-1] == 1)
								{
									less_flat_points[read_i + para_i][m] = less_flat_points[read_i + para_i][m-1];
									flat_outfeature_buffer[read_i + para_i][m].scanId = flat_outfeature_buffer[read_i + para_i][m-1].scanId;
									flat_outfeature_buffer[read_i + para_i][m].columnId = flat_outfeature_buffer[read_i + para_i][m-1].columnId;
								}
								else if(cmp_flag[m] == 1 && cmp_flag[m-1] == 0)
								{
									less_flat_points[read_i+para_i][m].curvature = in_point_Feature.curvature;
									less_flat_points[read_i + para_i][m].SortInd = in_point_Feature.SortInd;
									flat_outfeature_buffer[read_i + para_i][m].scanId = read_i + para_i;
									flat_outfeature_buffer[read_i + para_i][m].columnId = current_process_column_id;
								}
							}
							if(cmp_flag[0] == 1)
							{
								less_flat_points[read_i+para_i][0].curvature = in_point_Feature.curvature;
								less_flat_points[read_i + para_i][0].SortInd = in_point_Feature.SortInd;
								flat_outfeature_buffer[read_i + para_i][0].scanId = read_i + para_i;
								flat_outfeature_buffer[read_i + para_i][0].columnId = current_process_column_id;
							}
						}
					} // 找 flat 点

				}	// for para_i
			}// 如果不是保存和刷新，那就计算查找 else
		} // if(current_column_id >= 5 &&  current_column_id < Horizon_SCAN-6)

		if( read_i + PARAL_NUM >= (N_SCANS) )
		{
			read_i = 0;
			current_column_id ++;	// bug就来源于这句话。。这句话之前放在了前面。导致第一轮读取到的数据总是空的。
			current_process_column_id++;
			current_line_buffer_id ++ ;
			current_process_colum_buffer_id++;

			if(current_line_buffer_id == N_feature_buffer)
				current_line_buffer_id = 0;
			if(current_process_colum_buffer_id == N_feature_buffer)
				current_process_colum_buffer_id = 0;
		}
		else
			read_i = read_i + PARAL_NUM;
		
	}


	DEBUG_LOG( "extract_features read_count = " << read_count << " point_write_count = " << point_write_count << " feature_write_count = " << feature_write_count);
	DEBUG_LOG( "current_column_id = " << current_column_id << " current_process_column_id = " << current_process_column_id << " sp = " << sp << " ep = " << ep);
	DEBUG_GETCHAR;
}

static void out_point_range_lessflat_result(
				hls::stream<My_PointXYZI_HW> registrate_point_cloud_outpoint_stream[PARAL_NUM],
				hls::stream<type_range_hw> registrate_point_cloud_outrange_stream[PARAL_NUM],
				hls::stream<type_picked_hw> out_lessflat_stream[PARAL_NUM],
				hls::stream<type_ground_hw> out_ground_stream[PARAL_NUM],
				My_PointXYZI_Port* rangeimage_point, type_point_port_hw* rangeimage_range, type_picked_hw *rangeimage_lessflat, type_ground_hw *rangeimage_ground)
{
	int read_count = 0;
	type_range_hw range_read_from_stream[PARAL_NUM];
	My_PointXYZI_HW registrate_point_cloud_point_read_from_stream[PARAL_NUM];
	type_picked_hw lessflat_read_from_stream[PARAL_NUM];
	type_ground_hw ground_read_from_stream[PARAL_NUM];
#pragma HLS array_partition variable=range_read_from_stream complete dim=0
#pragma HLS array_partition variable=registrate_point_cloud_point_read_from_stream complete dim=0
#pragma HLS array_partition variable=lessflat_read_from_stream complete dim=0
#pragma HLS array_partition variable=ground_read_from_stream complete dim=0

	loop_out_point_range_lessflat_result_main:
	for(int j = 0; j < Horizon_SCAN; j++)
	{
		for(int i = 0; i < N_SCANS; i+=PARAL_NUM)
		{
			for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
			{
#pragma HLS PIPELINE II=1
				registrate_point_cloud_point_read_from_stream[para_i] = registrate_point_cloud_outpoint_stream[para_i].read();
				range_read_from_stream[para_i] = registrate_point_cloud_outrange_stream[para_i].read();
				lessflat_read_from_stream[para_i] = out_lessflat_stream[para_i].read();
				ground_read_from_stream[para_i] = out_ground_stream[para_i].read();

				My_PointXYZI_Port point_tobe_out;
				point_tobe_out.x = registrate_point_cloud_point_read_from_stream[para_i].x;
				point_tobe_out.y = registrate_point_cloud_point_read_from_stream[para_i].y;
				point_tobe_out.z = registrate_point_cloud_point_read_from_stream[para_i].z;
				point_tobe_out.intensity = registrate_point_cloud_point_read_from_stream[para_i].intensity;

				rangeimage_point[j*N_SCANS + i+para_i] = point_tobe_out;
				rangeimage_range[j*N_SCANS + i+para_i] = range_read_from_stream[para_i];
				rangeimage_lessflat[j*N_SCANS + i+para_i] = lessflat_read_from_stream[para_i];
				rangeimage_ground[j*N_SCANS + i+para_i] = ground_read_from_stream[para_i];
			}
		}
		read_count++;
	}
	DEBUG_LOG( "out_point_range_result read_count = " << read_count);
}

static void out_flat_sharp_result(
				hls::stream<My_PointOutFeatureLocation> out_flat_stream[PARAL_NUM*FLAT_SIZE],
				hls::stream<My_PointOutFeatureLocation> out_sharp_stream[PARAL_NUM*SHARP_SIZE],
				hls::stream<My_PointOutFeatureLocation> out_lesssharp_stream[PARAL_NUM*LESS_SHARP_SIZE],
				My_PointOutFeatureLocation *rangeimage_flat,
				My_PointOutFeatureLocation *rangeimage_sharp,
				My_PointOutFeatureLocation *rangeimage_lesssharp)
{
	int read_count = 0;
	My_PointOutFeatureLocation flat_read_from_stream[PARAL_NUM*FLAT_SIZE];
	My_PointOutFeatureLocation sharp_read_from_stream[PARAL_NUM*SHARP_SIZE];
	My_PointOutFeatureLocation lesssharp_read_from_stream[PARAL_NUM*LESS_SHARP_SIZE];

	loop_out_flat_sharp_result_result_main:
	for(int j = 0; j < EXTRACT_SEGMENT; j++)	// 分成了6段。
	{
		for(int i = 0; i < N_SCANS/PARAL_NUM; i++)
		{
			for(int para_i = 0; para_i < PARAL_NUM; para_i ++)
			{
#pragma HLS PIPELINE II=1
				for(int m = 0; m < FLAT_SIZE; m++)
				{
					flat_read_from_stream[para_i*FLAT_SIZE+m] = out_flat_stream[para_i*FLAT_SIZE+m].read();
					rangeimage_flat[j*N_SCANS*FLAT_SIZE+i*PARAL_NUM*FLAT_SIZE+para_i*FLAT_SIZE+m] = flat_read_from_stream[para_i*FLAT_SIZE+m];
				}
				for(int m = 0; m < SHARP_SIZE; m++)
				{
					sharp_read_from_stream[para_i*SHARP_SIZE+m] = out_sharp_stream[para_i*SHARP_SIZE+m].read();
					rangeimage_sharp[j*N_SCANS*SHARP_SIZE+i*PARAL_NUM*SHARP_SIZE+para_i*SHARP_SIZE+m] = sharp_read_from_stream[para_i*SHARP_SIZE+m];
				}
				for(int m = 0; m < LESS_SHARP_SIZE; m++)
				{
					lesssharp_read_from_stream[para_i*LESS_SHARP_SIZE+m] = out_lesssharp_stream[para_i*LESS_SHARP_SIZE+m].read();
					rangeimage_lesssharp[j*N_SCANS*LESS_SHARP_SIZE+i*PARAL_NUM*LESS_SHARP_SIZE+para_i*LESS_SHARP_SIZE+m] = lesssharp_read_from_stream[para_i*LESS_SHARP_SIZE+m];
				}
			}
		}
		read_count++;
	}
	DEBUG_LOG( "out_flat_sharp_result read_count = " << read_count);
}


static void fpga_feature_extract_dataflow(My_PointXYZI_Port* laserCloudInArray, int cloudSize, type_point_port_hw start_point_ori, type_point_port_hw end_point_ori,
	My_PointXYZI_Port *rangeimage_point, type_point_port_hw *rangeimage_range, type_picked_hw *rangeimage_lessflat, type_ground_hw *rangeimage_ground,
	My_PointOutFeatureLocation *rangeimage_flat, My_PointOutFeatureLocation *rangeimage_lesssharp, My_PointOutFeatureLocation *rangeimage_sharp)
{
#pragma HLS DATAFLOW

	// array_to_stream()
	hls::stream<My_PointXYZI_HW> input_point_stream;
#pragma HLS STREAM variable=input_point_stream depth=N_SCANS							// K_DEPTH_MAX

	// allocate_rangeimage()
	hls::stream<My_PointXYZI_HW> allocate_rangeimage_point_stream;
	hls::stream<type_range_hw> allocate_rangeimage_range_stream;
	hls::stream<type_scanid> allocate_rangeimage_scanid_stream;
	hls::stream<type_scanid> allocate_rangeimage_column_id_stream;
	hls::stream<type_picked_hw> allocate_rangeimage_end_stream;
#pragma HLS STREAM variable=allocate_rangeimage_point_stream depth=N_SCANS		// N_SCANS*Horizon_SCAN
#pragma HLS STREAM variable=allocate_rangeimage_range_stream depth=N_SCANS // N_SCANS*Horizon_SCAN
#pragma HLS STREAM variable=allocate_rangeimage_scanid_stream depth=N_SCANS		// N_SCANS*Horizon_SCAN
#pragma HLS STREAM variable=allocate_rangeimage_column_id_stream depth=N_SCANS // N_SCANS*Horizon_SCAN
#pragma HLS STREAM variable=allocate_rangeimage_end_stream depth=N_SCANS // N_SCANS*Horizon_SCAN

	// organize point cloud
	hls::stream<My_PointXYZI_HW> registrate_point_cloud_stream[ALLOCATE_PARAL_NUM];
	#pragma HLS array_partition variable=registrate_point_cloud_stream complete dim=0
	hls::stream<type_range_hw> registrate_point_cloud_range_stream[ALLOCATE_PARAL_NUM];
	#pragma HLS array_partition variable=registrate_point_cloud_range_stream complete dim=0
#pragma HLS STREAM variable=registrate_point_cloud_stream depth=N_SCANS		// N_SCANS*Horizon_SCAN
#pragma HLS STREAM variable=registrate_point_cloud_range_stream depth=N_SCANS // N_SCANS*Horizon_SCAN

	hls::stream<My_PointXYZI_HW> comple_point_cloud_stream[ALLOCATE_PARAL_NUM];
	#pragma HLS array_partition variable=comple_point_cloud_stream complete dim=0
	hls::stream<type_range_hw> comple_point_cloud_range_stream[ALLOCATE_PARAL_NUM];
	#pragma HLS array_partition variable=comple_point_cloud_range_stream complete dim=0
#pragma HLS STREAM variable=comple_point_cloud_stream depth=N_SCANS		// N_SCANS*Horizon_SCAN
#pragma HLS STREAM variable=comple_point_cloud_range_stream depth=N_SCANS // N_SCANS*Horizon_SCAN

	// update_range()
	hls::stream<My_PointXYZI_HW> update_registrate_point_cloud_point_stream[ALLOCATE_PARAL_NUM];
	#pragma HLS array_partition variable=update_registrate_point_cloud_point_stream complete dim=0
	hls::stream<type_range_hw> update_registrate_point_cloud_range_stream[ALLOCATE_PARAL_NUM];
	#pragma HLS array_partition variable=update_registrate_point_cloud_range_stream complete dim=0
#pragma HLS STREAM variable=update_registrate_point_cloud_point_stream depth=N_SCANS	// N_SCANS*Horizon_SCAN
#pragma HLS STREAM variable=update_registrate_point_cloud_range_stream depth=N_SCANS	// N_SCANS*Horizon_SCAN

	// cache_rangeimage()
	hls::stream<My_PointXYZI_HW> cache_image_point_stream[PARAL_NUM];
	#pragma HLS array_partition variable=cache_image_point_stream complete dim=0
	hls::stream<type_range_hw> cache_image_range_stream[PARAL_NUM];
	#pragma HLS array_partition variable=cache_image_range_stream complete dim=0
#pragma HLS STREAM variable=cache_image_point_stream depth=N_SCANS			// N_SCANS*Horizon_SCAN
#pragma HLS STREAM variable=cache_image_range_stream depth=N_SCANS			// N_SCANS*Horizon_SCAN

	// compute_curvature()
	hls::stream<My_PointXYZI_HW> curvature_out_point_stream[PARAL_NUM];
	#pragma HLS array_partition variable=curvature_out_point_stream complete dim=0
	hls::stream<type_range_hw> curvature_out_range_stream[PARAL_NUM];
	#pragma HLS array_partition variable=curvature_out_range_stream complete dim=0
	hls::stream<type_curvature_hw> curvature_out_curvature_stream[PARAL_NUM];
	#pragma HLS array_partition variable=curvature_out_curvature_stream complete dim=0
	hls::stream<type_picked_hw> curvature_out_picked_stream[PARAL_NUM];
#pragma HLS STREAM variable=curvature_out_point_stream depth=N_SCANS		// N_SCANS*Horizon_SCAN
#pragma HLS STREAM variable=curvature_out_range_stream depth=N_SCANS		// N_SCANS*Horizon_SCAN
#pragma HLS STREAM variable=curvature_out_curvature_stream depth=N_SCANS	// N_SCANS*Horizon_SCAN
#pragma HLS STREAM variable=curvature_out_picked_stream depth=N_SCANS	// N_SCANS*Horizon_SCAN

	// compute_ground()
	hls::stream<My_PointXYZI_HW> ground_out_point_stream[PARAL_NUM];
	#pragma HLS array_partition variable=ground_out_point_stream complete dim=0
	hls::stream<type_range_hw> ground_out_range_stream[PARAL_NUM];
	#pragma HLS array_partition variable=ground_out_range_stream complete dim=0
	hls::stream<type_curvature_hw> ground_out_curvature_stream[PARAL_NUM];
	#pragma HLS array_partition variable=ground_out_curvature_stream complete dim=0
	hls::stream<type_picked_hw> ground_out_picked_stream[PARAL_NUM];
	#pragma HLS array_partition variable=ground_out_picked_stream complete dim=0
	hls::stream<type_sortind_hw> ground_out_sortind_stream[PARAL_NUM];
	#pragma HLS array_partition variable=ground_out_sortind_stream complete dim=0
	hls::stream<type_ground_hw> ground_out_ground_stream[PARAL_NUM];
	#pragma HLS array_partition variable=ground_out_ground_stream complete dim=0
#pragma HLS STREAM variable=ground_out_point_stream depth=N_SCANS		// N_SCANS*Horizon_SCAN			40*64*N=2560N/1024/18=10N/4/18=5N/36  *400的话大概50个
#pragma HLS STREAM variable=ground_out_range_stream depth=N_SCANS		// N_SCANS*Horizon_SCAN			上面个数除以4， 大概十几个
#pragma HLS STREAM variable=ground_out_curvature_stream depth=N_SCANS	// N_SCANS*Horizon_SCAN			上面那个成1.5， 大概二十多个，
#pragma HLS STREAM variable=ground_out_picked_stream depth=N_SCANS		// N_SCANS*Horizon_SCAN			一个
#pragma HLS STREAM variable=ground_out_sortind_stream depth=N_SCANS		// N_SCANS*Horizon_SCAN			8个
#pragma HLS STREAM variable=ground_out_ground_stream depth=N_SCANS		// N_SCANS*Horizon_SCAN			2个、

	// integrate_features()
	hls::stream<My_PointXYZI_HW> registrate_point_cloud_point_stream[PARAL_NUM];
	#pragma HLS array_partition variable=registrate_point_cloud_point_stream complete dim=0
	hls::stream<type_range_hw> registrate_point_cloud_point_range_stream[PARAL_NUM];
	#pragma HLS array_partition variable=registrate_point_cloud_point_range_stream complete dim=0
	hls::stream<My_PointFeature_hw> registrate_point_cloud_feature_stream[PARAL_NUM];
	#pragma HLS array_partition variable=registrate_point_cloud_feature_stream complete dim=0
#pragma HLS STREAM variable=registrate_point_cloud_point_stream depth=N_SCANS		// N_SCANS*Horizon_SCAN
#pragma HLS STREAM variable=registrate_point_cloud_point_range_stream depth=N_SCANS	// N_SCANS*Horizon_SCAN
#pragma HLS STREAM variable=registrate_point_cloud_feature_stream depth=N_SCANS		// N_SCANS*Horizon_SCAN

	// extract_features()
	hls::stream<type_range_hw> registrate_point_cloud_outrange_stream[PARAL_NUM];
	hls::stream<My_PointXYZI_HW> registrate_point_cloud_outpoint_stream[PARAL_NUM];
	hls::stream<type_picked_hw> out_lessflat_stream[PARAL_NUM];
	hls::stream<type_ground_hw> out_ground_stream[PARAL_NUM];
	hls::stream<My_PointOutFeatureLocation> out_sharp_stream[PARAL_NUM*SHARP_SIZE];
	hls::stream<My_PointOutFeatureLocation> out_lesssharp_stream[PARAL_NUM*LESS_SHARP_SIZE];
	hls::stream<My_PointOutFeatureLocation> out_flat_stream[PARAL_NUM*FLAT_SIZE];

	#pragma HLS array_partition variable=registrate_point_cloud_outrange_stream complete dim=0
	#pragma HLS array_partition variable=registrate_point_cloud_outpoint_stream complete dim=0
	#pragma HLS array_partition variable=out_lessflat_stream complete dim=0
	#pragma HLS array_partition variable=out_ground_stream complete dim=0
	#pragma HLS array_partition variable=out_sharp_stream complete dim=0
	#pragma HLS array_partition variable=out_lesssharp_stream complete dim=0
	#pragma HLS array_partition variable=out_flat_stream complete dim=0
#pragma HLS STREAM variable=registrate_point_cloud_outrange_stream depth=N_SCANS	// N_SCANS*Horizon_SCAN
#pragma HLS STREAM variable=registrate_point_cloud_outpoint_stream depth=N_SCANS	// N_SCANS*Horizon_SCAN
#pragma HLS STREAM variable=out_lessflat_stream depth=N_SCANS					// N_SCANS*Horizon_SCAN
#pragma HLS STREAM variable=out_ground_stream depth=N_SCANS						// N_SCANS*Horizon_SCAN
#pragma HLS STREAM variable=out_sharp_stream depth=N_SCANS						// N_SCANS*EXTRACT_SEGMENT*SHARP_SIZE
#pragma HLS STREAM variable=out_lesssharp_stream depth=N_SCANS					// N_SCANS*EXTRACT_SEGMENT*LESS_SHARP_SIZE
#pragma HLS STREAM variable=out_flat_stream depth=N_SCANS							// N_SCANS*EXTRACT_SEGMENT*FLAT_SIZE 32*64


    /********************************************** 1. 初始化 rangeImage ******************************************/
	array_to_stream(laserCloudInArray, cloudSize, input_point_stream);
	DEBUG_LOG("to in allocate_rangeimage ");

	/********************************************** 2. 根据垂直角与水平角建立 rangeimage ******************************************/
	allocate_rangeimage(input_point_stream, cloudSize, start_point_ori, end_point_ori,
						allocate_rangeimage_point_stream, allocate_rangeimage_range_stream, 
						allocate_rangeimage_scanid_stream, allocate_rangeimage_column_id_stream, allocate_rangeimage_end_stream);

	DEBUG_LOG("to in organize_point_cloud ");
	organize_point_cloud(allocate_rangeimage_point_stream, allocate_rangeimage_range_stream, 
						allocate_rangeimage_scanid_stream, allocate_rangeimage_column_id_stream, allocate_rangeimage_end_stream,
						registrate_point_cloud_stream, registrate_point_cloud_range_stream);

	// comple_point(registrate_point_cloud_stream, registrate_point_cloud_range_stream,
	// 			comple_point_cloud_stream, comple_point_cloud_range_stream);

	DEBUG_LOG("to in update_range ");
	/********************************************** 4. 计算 rangeimage 的曲率和地面点, 还有pickedflag,   ******************************************/
	update_range(registrate_point_cloud_stream, registrate_point_cloud_range_stream, update_registrate_point_cloud_point_stream, update_registrate_point_cloud_range_stream);

	cache_rangeimage(update_registrate_point_cloud_point_stream, update_registrate_point_cloud_range_stream, cache_image_point_stream, cache_image_range_stream);

	compute_curvature(cache_image_point_stream, cache_image_range_stream,
					curvature_out_point_stream, curvature_out_range_stream, curvature_out_curvature_stream, curvature_out_picked_stream);
	
	compute_ground(curvature_out_point_stream, curvature_out_range_stream, curvature_out_curvature_stream, curvature_out_picked_stream,
					ground_out_point_stream, ground_out_range_stream, ground_out_curvature_stream, ground_out_picked_stream, ground_out_sortind_stream, ground_out_ground_stream);

	// debug to_out
	// compute_ground_to_output(ground_out_point_stream, ground_out_range_stream, ground_out_curvature_stream, ground_out_picked_stream, ground_out_sortind_stream, ground_out_ground_stream,
	// rangeimage_point, rangeimage_range, rangeimage_lessflat, rangeimage_ground, rangeimage_flat, rangeimage_sharp, rangeimage_lesssharp);

	integrate_features(ground_out_point_stream, ground_out_range_stream, ground_out_curvature_stream, ground_out_picked_stream, ground_out_sortind_stream, ground_out_ground_stream,
	registrate_point_cloud_point_stream, registrate_point_cloud_point_range_stream, registrate_point_cloud_feature_stream);

	// // debug to_out
	// // integrate_features_to_output(registrate_point_cloud_point_stream, registrate_point_cloud_point_range_stream, registrate_point_cloud_feature_stream, 
	// // rangeimage_point, rangeimage_range, rangeimage_lessflat, rangeimage_ground, rangeimage_flat, rangeimage_sharp, rangeimage_lesssharp);

	/********************************************** 5. 根据前面的信息，提取特征平面点和特征角点。   *********************************************/
	DEBUG_LOG("to in extract_features ");
	extract_features(registrate_point_cloud_point_stream, registrate_point_cloud_point_range_stream, registrate_point_cloud_feature_stream,
						out_sharp_stream, out_lesssharp_stream, out_flat_stream, out_lessflat_stream, out_ground_stream, registrate_point_cloud_outpoint_stream, registrate_point_cloud_outrange_stream);

	/********************************************** 6. 将结果输出   ***********************************************************************/
	DEBUG_LOG("to in out_result ");
	out_point_range_lessflat_result(registrate_point_cloud_outpoint_stream, registrate_point_cloud_outrange_stream, out_lessflat_stream, out_ground_stream,
				rangeimage_point, rangeimage_range, rangeimage_lessflat, rangeimage_ground);
	out_flat_sharp_result(out_flat_stream, out_sharp_stream, out_lesssharp_stream, rangeimage_flat, rangeimage_sharp, rangeimage_lesssharp);

	DEBUG_GETCHAR;
}

extern "C"
void fpga_feature_extract_hw(My_PointXYZI_Port* laserCloudInArray, int cloudSize, type_point_port_hw start_point_ori, type_point_port_hw end_point_ori,
	My_PointXYZI_Port *rangeimage_point, type_point_port_hw *rangeimage_range, type_picked_hw *rangeimage_lessflat, type_ground_hw *rangeimage_ground,
	My_PointOutFeatureLocation *rangeimage_flat, My_PointOutFeatureLocation *rangeimage_lesssharp, My_PointOutFeatureLocation *rangeimage_sharp)
{
#pragma HLS INTERFACE m_axi bundle = gmem0 port=laserCloudInArray    depth=117142
#pragma HLS INTERFACE m_axi bundle = gmem1 port=rangeimage_point 	depth=115200    // depth 信息不加 co-sim 就过不去。。
#pragma HLS INTERFACE m_axi bundle = gmem2 port=rangeimage_range 	depth=115200
#pragma HLS INTERFACE m_axi bundle = gmem3 port=rangeimage_lessflat 	depth=115200
#pragma HLS INTERFACE m_axi bundle = gmem4 port=rangeimage_ground 	depth=115200
#pragma HLS INTERFACE m_axi bundle = gmem5 port=rangeimage_flat 		depth=1536
#pragma HLS INTERFACE m_axi bundle = gmem6 port=rangeimage_lesssharp depth=7680
#pragma HLS INTERFACE m_axi bundle = gmem7 port=rangeimage_sharp 	depth=768
#pragma HLS aggregate variable=laserCloudInArray
#pragma HLS aggregate variable=rangeimage_point
#pragma HLS aggregate variable=rangeimage_flat
#pragma HLS aggregate variable=rangeimage_lesssharp
#pragma HLS aggregate variable=rangeimage_sharp

#pragma HLS INTERFACE s_axilite port=cloudSize bundle=control_bus
#pragma HLS INTERFACE s_axilite port=start_point_ori bundle=control_bus
#pragma HLS INTERFACE s_axilite port=end_point_ori bundle=control_bus
#pragma HLS INTERFACE s_axilite port=return bundle=control_bus

	fpga_feature_extract_dataflow(laserCloudInArray, cloudSize, start_point_ori, end_point_ori,
		rangeimage_point, rangeimage_range, rangeimage_lessflat, rangeimage_ground, rangeimage_flat, rangeimage_lesssharp, rangeimage_sharp);    // hw 版本

}
