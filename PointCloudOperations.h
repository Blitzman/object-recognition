#ifndef POINTCLOUDOPERATIONS_H_
#define POINTCLOUDOPERATIONS_H_

//#define FLANN_USE_CUDA

#include <iostream>
#include <string>
#include <boost/timer/timer.hpp>
#include <pcl/search/flann_search.h>
#include <pcl/point_types.h>

#include <flann/flann.hpp>

// PCL headers
#include <pcl/point_cloud.h>
#include <pcl/impl/instantiate.hpp>
#include <pcl/common/random.h>
// PCL common headers
#include <pcl/common/transforms.h>
#include <pcl/common/angles.h>
// PCL kdtree headers
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
// PCL features headers
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/spin_image.h>
#include <pcl/features/3dsc.h>
#include <pcl/features/usc.h>
#include <pcl/features/rops_estimation.h>
#include <pcl/features/board.h>
// PCL filters headers
#include <pcl/filters/filter.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/filters/passthrough.h>
// PCL keypoints headers
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/iss_3d.h>
// PCL recognition headers
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/hv/hv_go.h>
// PCL registration headers
#include <pcl/registration/icp.h>
// PCL segmentation headers
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
// PCL surface headers
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
// PCL visualization headers
#include <pcl/visualization/pcl_visualizer.h>

// Custom includes
#include "Types.h"

#define DEBUG 1

class PointCloudOperations
{
	public:
		static void bounding_box
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
				const float & pLimitMinX,
				const float & pLimitMaxX,
				const float & pLimitMinY,
				const float & pLimitMaxY,
				const float & pLimitMinZ,
				const float & pLimitMaxZ,
				pcl::PointCloud<TPoint>::Ptr & pDstCloud
			);

		static void filter_bilateral
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
				const float & pSigmaR,
				const float & pSigmaS,
				pcl::PointCloud<TPoint>::Ptr & pDstCloud
			);

		static void compute_normals_organized
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pCloud,
				pcl::PointCloud<TNormal>::Ptr & pNormals,
				const int & pNeighbors
			);

		static void compute_normals
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pCloud,
				pcl::PointCloud<TNormal>::Ptr & pNormals,
				const int & pNeighbors
			);

		static double compute_resolution
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pCloud
			);

		static void segment_objects
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pCloud,
				const pcl::PointCloud<TNormal>::ConstPtr & pNormals,
				const int & pMinInliers,
				const double & pAngularThreshold,
				const double & pDistanceThreshold,
				pcl::PointCloud<TPoint>::Ptr & pDstCloud
			);

		static void detect_keypoints_iss
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pCloud,
				const float & pSalientRadius,
				const float & pNonMaxRadius,
				const int & pMinNeighbors,
				const float & pThreshold21,
				const float & pThreshold32,
				pcl::PointCloud<TPoint>::Ptr & pDstCloud
			);

		static void detect_keypoints_sift
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pCloud,
				const float & pMinScale,
				const float & pNrOctaves,
				const float & pNrScalesPerOctave,
				const float & pMinConstrast,
				pcl::PointCloud<TPoint>::Ptr & pDstCloud
			);

		static void downsample_uniform_sampling
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
				pcl::PointCloud<TPoint>::Ptr & pDstCloud,
				const float & pRadiusSearch
			);

		static void downsample_voxel_grid
			(
				const pcl::PointCloud<TPoint>::Ptr & pSrcCloud,
				pcl::PointCloud<TPoint>::Ptr & pDstCloud,
				const float & pLeafSize
			);

		static void compute_shot_descriptors
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
				const pcl::PointCloud<TNormal>::ConstPtr & pNormals,
				const pcl::PointCloud<TPoint>::ConstPtr & pKeypoints,
				const float & pSearchRadius,
				pcl::PointCloud<pcl::SHOT352>::Ptr & pDstCloud
			);

		static void compute_cshot_descriptors
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
				const pcl::PointCloud<TNormal>::ConstPtr & pNormals,
				const pcl::PointCloud<TPoint>::ConstPtr & pKeypoints,
				const float & pSearchRadius,
				pcl::PointCloud<pcl::SHOT1344>::Ptr & pDstCloud
			);

		static void compute_fpfh_descriptors
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
				const pcl::PointCloud<TNormal>::ConstPtr & pNormals,
				const pcl::PointCloud<TPoint>::ConstPtr & pKeypoints,
				const float & pSearchRadius,
				pcl::PointCloud<pcl::FPFHSignature33>::Ptr & pDstCloud
			);

		static void compute_spinimage_descriptors
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
				const pcl::PointCloud<TNormal>::ConstPtr & pNormals,
				const pcl::PointCloud<TPoint>::ConstPtr & pKeypoints,
				const float & pSearchRadius,
				pcl::PointCloud<SpinImage>::Ptr & pDstCloud	
			);

		static void compute_3dsc_descriptors
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
				const pcl::PointCloud<TNormal>::ConstPtr & pNormals,
				const pcl::PointCloud<TPoint>::ConstPtr & pKeypoints,
				const float & pSearchRadius,
				pcl::PointCloud<pcl::ShapeContext1980>::Ptr & pDstCloud
			);

		/*static void compute_usc_descriptors
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
				const pcl::PointCloud<TNormal>::ConstPtr & pNormals,
				const pcl::PointCloud<TPoint>::ConstPtr & pKeypoints,
				const float & pSearchRadius,
				pcl::PointCloud<pcl::UniqueShapeContext1960>::Ptr & pDstCloud
			);*/

		static void compute_rops_descriptors
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
				const pcl::PointCloud<TNormal>::ConstPtr & pNormals,
				const pcl::PointCloud<TPoint>::ConstPtr & pKeypoints,
				const float & pSearchRadius,
				pcl::PointCloud<ROPS135>::Ptr & pDstCloud
			);

		static void find_correspondences_gpu
			(
				const pcl::PointCloud<TDescriptor>::ConstPtr & pModelDescriptors,
				const pcl::PointCloud<TDescriptor>::ConstPtr & pSceneDescriptors,
				const float & pThreshold,
				pcl::CorrespondencesPtr & pCorrespondences
			);

		static void find_correspondences_multicore
			(
				const pcl::PointCloud<TDescriptor>::ConstPtr & pModelDescriptors,
				const pcl::PointCloud<TDescriptor>::ConstPtr & pSceneDescriptors,
				const float & pThreshold,
				pcl::CorrespondencesPtr & pCorrespondences
			);

		static void find_correspondences
			(
				const pcl::PointCloud<TDescriptor>::ConstPtr & pModelDescriptors,
				const pcl::PointCloud<TDescriptor>::ConstPtr & pSceneDescriptors,
				const float & pThreshold,
				pcl::CorrespondencesPtr & pCorrespondences
			);

		static void cluster_correspondences
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pModel,
				const pcl::PointCloud<TNormal>::ConstPtr & pModelNormals,
				const pcl::PointCloud<TPoint>::ConstPtr & pModelKeypoints,
				const pcl::PointCloud<TPoint>::ConstPtr & pScene,
				const pcl::PointCloud<TNormal>::ConstPtr & pSceneNormals,
				const pcl::PointCloud<TPoint>::ConstPtr & pSceneKeypoints,
				const pcl::CorrespondencesPtr & pModelSceneCorrespondences,
				const bool & pUseHough3D,
				const float & pClusterSize,
				const float & pClusterThreshold,
				const float & pReferenceFrameRadius,
				std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & pRotoTranslations,
				std::vector<pcl::Correspondences> & pClusteredCorrespondences
			);

		static void register_cloud
			(
				const pcl::PointCloud<TPoint>::ConstPtr & pModel,
				const pcl::PointCloud<TPoint>::ConstPtr & pScene,
				const int & pMaxIterations,
				const float & pMaxCorrespondenceDistance,
				pcl::PointCloud<TPoint>::Ptr & pDstCloud
			);

		static void verify_hypotheses
			(
				const pcl::PointCloud<TPoint>::Ptr & pScene,
				std::vector<pcl::PointCloud<TPoint>::ConstPtr> & pModels,
				const float & pInlierThreshold,
				const float & pOcclusionThreshold,
				const float & pRadiusClutter,
				const float & pRegularizer,
				const float & pClutterRegularizer,
				const bool & pDetectClutter,
				const float & pRadiusNormals,
				std::vector<bool> & pHypothesesMask
			);
};

#endif
