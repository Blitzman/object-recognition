#ifndef RECOGNIZER_H_
#define RECOGNIZER_H_

#include <iostream>
#include <string>
// Boost headers
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>
// PCL input/output headers
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/point_cloud.h>
// PCL visualization headers
#include <pcl/visualization/pcl_visualizer.h>
// Custom headers
#include "Utils.h"
#include "Types.h"
#include "PointCloudOperations.h"
#include "Settings.h"

#define PARAM_HELP					"h"
#define PARAM_MODEL_DIRECTORY				"model_dir"
#define PARAM_SCENE_FILENAME				"scene"

#define PARAM_VISUALIZER_SHOW_NORMALS			"v_n"
#define PARAM_VISUALIZER_SHOW_KEYPOINTS			"v_k"
#define PARAM_VISUALIZER_SHOW_CORRESPONDENCES		"v_c"
#define PARAM_VISUALIZER_SHOW_DESCRIPTORS		"v_d"
#define PARAM_VISUALIZER_SHOW_HYPOTHESES		"v_h"

#define PARAM_USE_CLOUD_RESOLUTION			"r"

#define PARAM_BILATERAL_FILTER				"bf"
#define PARAM_BILATERAL_FILTER_SIGMA_R			"bf_sr"
#define PARAM_BILATERAL_FILTER_SIGMA_S			"bf_ss"

#define PARAM_NORMAL_ESTIMATION_K			"ne_k"

#define PARAM_SEGMENTATION				"mps"
#define PARAM_SEGMENTATION_MININLIERS			"mps_mi"
#define PARAM_SEGMENTATION_ANGULARTHRESHOLD		"mps_at"
#define PARAM_SEGMENTATION_DISTANCETHRESHOLD		"mps_dt"
#define PARAM_DOWNSAMPLE_MODEL 				"vg_m" 
#define PARAM_DOWNSAMPLE_MODEL_LEAF_SIZE 		"vg_m_ls" 
#define PARAM_DOWNSAMPLE_SCENE 				"vg_s" 
#define PARAM_DOWNSAMPLE_SCENE_LEAF_SIZE 		"vg_s_ls"
#define PARAM_KEYPOINTS_ISS				"iss"
#define PARAM_KEYPOINTS_ISS_SALIENTRADIUS		"iss_sr"
#define PARAM_KEYPOINTS_ISS_NONMAXRADIUS		"iss_nmr"
#define PARAM_KEYPOINTS_ISS_MINEIGHBORS			"iss_mn"
#define PARAM_KEYPOINTS_ISS_THRESHOLD21			"iss_t21"
#define PARAM_KEYPOINTS_ISS_THRESHOLD32			"iss_t32"

#define PARAM_KEYPOINTS_SIFT				"sift"
#define PARAM_KEYPOINTS_SIFT_MINSCALE			"sift_ms"
#define PARAM_KEYPOINTS_SIFT_NROCTAVES			"sift_no"
#define PARAM_KEYPOINTS_SIFT_SCALESPEROCT		"sift_spo"
#define PARAM_KEYPOINTS_SIFT_MINCONTRAST		"sift_mc"

#define PARAM_MODEL_SEARCH_RADIUS			"m_ss"
#define PARAM_SCENE_SEARCH_RADIUS			"s_ss"

#define PARAM_DESCRIPTOR_RADIUS				"d_r"
#define PARAM_DESCRIPTOR_LRF_RADIUS			"d_lrf_r"

#define PARAM_MATCHING_THRESHOLD			"m_t"

#define PARAM_GROUPING_USE_HOUGH			"cg_h"
#define PARAM_GROUPING_CLUSTER_SIZE			"cg_cs"
#define PARAM_GROUPING_CLUSTER_THRESHOLD		"cg_ct"

#define PARAM_ICP					"icp"
#define PARAM_ICP_MAX_ITERATIONS			"icp_i"
#define PARAM_ICP_MAX_CORRESPONDENCE_DISTANCE		"icp_d"

#define PARAM_HV_INLIER_THRESHOLD			"hv_it"
#define PARAM_HV_OCCLUSION_THRESHOLD			"hv_ot"
#define PARAM_HV_RADIUS_CLUTTER				"hv_rc"
#define PARAM_HV_CLUTTER_REGULARIZER			"hv_cr"
#define PARAM_HV_REGULARIZER				"hv_r"
#define PARAM_HV_DETECT_CLUTTER				"hv_dc"
#define PARAM_HV_RADIUS_NORMALS				"hv_rn"

class Recognizer
{
	public:
		Recognizer(const boost::program_options::variables_map & variablesMap);
		~Recognizer();

		void load_models();
		void recognize(const pcl::PointCloud<TPoint>::Ptr & sceneCloud);

	private:
		// Parameters
		std::string modelDirectory;
		std::string sceneFilename;

		bool online;

		bool showNormals;
		bool showKeypoints;
		bool showCorrespondences;
		bool showHypotheses;
		bool showDescriptors;
		bool useCloudResolution;

		bool applyBilateralFilter;
		float bilateralFilterSigmaR;
		float bilateralFilterSigmaS;

		int normalEstimationK;

		bool useMPS;
		int mpsMinInliers;
		double mpsAngThres;
		double mpsDstThres;

		bool downsampleModel;
		float downsampleModelLeaf;
		bool downsampleScene;
		float downsampleSceneLeaf;

		bool useISS;
		float issSalientRadius;
		float issNonMaxRadius;
		int issMinNeighbors;
		float issThreshold21;
		float issThreshold32;

		bool useSIFT;
		float siftMinimumScale;
		int siftNumberOfOctaves;
		int siftScalesPerOctave;
		float siftMinimumContrast;

		float modelSearchRadius;
		float sceneSearchRadius;

		float descriptorRadius;
		float descriptorLRFRadius;

		float matchingThreshold;

		bool useHough3D;
		float clusterSize;
		float clusterThreshold;

		bool useICP;
		int icpMaxIterations;
		float icpMaxCorrDistance;

		float hvClutterRegularizer;
		float hvInlierThreshold;
		float hvOcclusionThreshold;
		float hvRadiusClutter;
		float hvRegularizer;
		float hvRadiusNormals;
		bool hvDetectClutter;

		// Model database
		std::vector<std::string> modelFilenames;
		std::vector<pcl::PointCloud<TPoint>::Ptr> originalModels;
		std::vector<pcl::PointCloud<TPoint>::Ptr> models;
		std::vector<pcl::PointCloud<TNormal>::Ptr> modelsNormals;

		//pcl::visualization::PCLVisualizer viewer;
};

#endif
