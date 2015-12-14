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
// OpenNI headers
#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
// Custom headers
#include "Recognizer.h"
#include "Grabber.h"
#include "Types.h"

#define DEBUG 1

#define PARAM_ONLINE					"online"
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

bool parse_command_line_options(boost::program_options::variables_map & pVariablesMap, const int & pArgc, char ** pArgv)
{
	try
	{
		boost::program_options::options_description desc("Allowed options");
		desc.add_options()
			(PARAM_HELP, "produce help message")
			(PARAM_ONLINE, boost::program_options::value<bool>()->default_value(false), "Online")
			(PARAM_MODEL_DIRECTORY, boost::program_options::value<std::string>()->default_value("models"), "Model directory")
			(PARAM_SCENE_FILENAME, boost::program_options::value<std::string>()->required(), "Scene filename")
			(PARAM_VISUALIZER_SHOW_NORMALS, boost::program_options::value<bool>()->default_value(false), "Show normals on visualizer")
			(PARAM_VISUALIZER_SHOW_KEYPOINTS, boost::program_options::value<bool>()->default_value(true), "Show keypoints in visualizer")
			(PARAM_VISUALIZER_SHOW_CORRESPONDENCES, boost::program_options::value<bool>()->default_value(true), "Show correspondences in visualizer")
			(PARAM_VISUALIZER_SHOW_DESCRIPTORS, boost::program_options::value<bool>()->default_value(false), "Show descriptors on visualizer")
			(PARAM_VISUALIZER_SHOW_HYPOTHESES, boost::program_options::value<bool>()->default_value(false), "Show hypotheses on visualizer")
			(PARAM_USE_CLOUD_RESOLUTION, boost::program_options::value<bool>()->default_value(false), "Use cloud resolution")
			(PARAM_BILATERAL_FILTER, boost::program_options::value<bool>()->default_value(false), "Apply bilateral filter to scene")
			(PARAM_BILATERAL_FILTER_SIGMA_R, boost::program_options::value<float>()->default_value(0.05f), "Bilateral filter sigma R")
			(PARAM_BILATERAL_FILTER_SIGMA_S, boost::program_options::value<float>()->default_value(15.0f), "Bilateral filter sigma S")
			(PARAM_NORMAL_ESTIMATION_K, boost::program_options::value<int>()->default_value(25), "Normal estimation radius")
			(PARAM_SEGMENTATION, boost::program_options::value<bool>()->default_value(true), "Activate Multi Plane Segmentation")
			(PARAM_SEGMENTATION_MININLIERS, boost::program_options::value<int>()->default_value(100), "Minimum inliers for plane segmentation")
			(PARAM_SEGMENTATION_ANGULARTHRESHOLD, boost::program_options::value<double>()->default_value(2.0), "Angular threshold for plane segmentation")
			(PARAM_SEGMENTATION_DISTANCETHRESHOLD, boost::program_options::value<double>()->default_value(2.0), "Distance threshold for plane segmentation")
			(PARAM_DOWNSAMPLE_MODEL, boost::program_options::value<bool>()->default_value(false), "Downsample model using Voxel Grid")
			(PARAM_DOWNSAMPLE_MODEL_LEAF_SIZE, boost::program_options::value<float>()->default_value(0.01f), "Model downsampling leaf size")
			(PARAM_DOWNSAMPLE_SCENE, boost::program_options::value<bool>()->default_value(false), "Downsample scene using Voxel Grid")
			(PARAM_DOWNSAMPLE_SCENE_LEAF_SIZE, boost::program_options::value<float>()->default_value(0.01f), "Scene downsampling leaf size")
			(PARAM_KEYPOINTS_ISS, boost::program_options::value<bool>()->default_value(false), "Use ISS for keypoint extraction")
			(PARAM_KEYPOINTS_ISS_SALIENTRADIUS, boost::program_options::value<float>()->default_value(6), "ISS salient radius")
			(PARAM_KEYPOINTS_ISS_NONMAXRADIUS, boost::program_options::value<float>()->default_value(4), "ISS non max radius")
			(PARAM_KEYPOINTS_ISS_MINEIGHBORS, boost::program_options::value<int>()->default_value(5), "ISS minimum number of neighbors")
			(PARAM_KEYPOINTS_ISS_THRESHOLD21, boost::program_options::value<float>()->default_value(0.975), "ISS threshold 21")
			(PARAM_KEYPOINTS_ISS_THRESHOLD32, boost::program_options::value<float>()->default_value(0.975), "ISS threshold 32")
			(PARAM_KEYPOINTS_SIFT, boost::program_options::value<bool>()->default_value(false), "Use SIFT for keypoint extraction")
			(PARAM_KEYPOINTS_SIFT_MINSCALE, boost::program_options::value<float>()->default_value(0.0005f), "SIFT minimum scale")
			(PARAM_KEYPOINTS_SIFT_NROCTAVES, boost::program_options::value<int>()->default_value(8), "SIFT number of scales")
			(PARAM_KEYPOINTS_SIFT_SCALESPEROCT, boost::program_options::value<int>()->default_value(8), "SIFT scales per octave")
			(PARAM_KEYPOINTS_SIFT_MINCONTRAST, boost::program_options::value<float>()->default_value(0.005f), "SIFT minimum contrast")
			(PARAM_MODEL_SEARCH_RADIUS, boost::program_options::value<float>()->default_value(4.0f), "Model search radius")
			(PARAM_SCENE_SEARCH_RADIUS, boost::program_options::value<float>()->default_value(4.0f), "Scene search radius")
			(PARAM_DESCRIPTOR_RADIUS, boost::program_options::value<float>()->default_value(20.0f), "Descriptor radius")
			(PARAM_DESCRIPTOR_LRF_RADIUS, boost::program_options::value<float>()->default_value(20.0f), "Descriptor Reference Frame radius")
			(PARAM_MATCHING_THRESHOLD, boost::program_options::value<float>()->default_value(0.25f), "Correspondence matching threshold")
			(PARAM_GROUPING_USE_HOUGH, boost::program_options::value<bool>()->default_value(false), "Use Hough3D for correspondence grouping")
			(PARAM_GROUPING_CLUSTER_SIZE, boost::program_options::value<float>()->default_value(0.01f), "Correspondence grouping cluster size")
			(PARAM_GROUPING_CLUSTER_THRESHOLD, boost::program_options::value<float>()->default_value(5.0f), "Correspondence grouping cluster threshold")
			(PARAM_ICP, boost::program_options::value<bool>()->default_value(false), "Align clouds with ICP")
			(PARAM_ICP_MAX_ITERATIONS, boost::program_options::value<int>()->default_value(5), "Maximum iterations for ICP")
			(PARAM_ICP_MAX_CORRESPONDENCE_DISTANCE, boost::program_options::value<float>()->default_value(0.005f), "Maximum correspondence distance for ICP")
			(PARAM_HV_INLIER_THRESHOLD, boost::program_options::value<float>()->default_value(0.005f), "Hypothesis verification inlier threshold")
			(PARAM_HV_OCCLUSION_THRESHOLD, boost::program_options::value<float>()->default_value(0.01f), "Hypothesis verification occlusion threshold")
			(PARAM_HV_RADIUS_CLUTTER, boost::program_options::value<float>()->default_value(0.03f), "Hypothesis verification radius clutter")
			(PARAM_HV_CLUTTER_REGULARIZER, boost::program_options::value<float>()->default_value(5.0f), "Hypothesis verification clutter regularizer")
			(PARAM_HV_REGULARIZER, boost::program_options::value<float>()->default_value(3.0f), "Hypothesis verification regularizer")
			(PARAM_HV_RADIUS_NORMALS, boost::program_options::value<float>()->default_value(0.05f), "Hypothesis verification radius normals")
			(PARAM_HV_DETECT_CLUTTER, boost::program_options::value<bool>()->default_value(true), "Hypothesis verification clutter detection");

		boost::program_options::store(boost::program_options::parse_command_line(pArgc, pArgv, desc), pVariablesMap);

		if (pVariablesMap.count(PARAM_HELP))
		{
			std::cout << desc << "\n";
			return true;
		}

		boost::program_options::notify(pVariablesMap);
	}
	catch (std::exception & e)
	{
		std::cerr << "Error: " << e.what() << "\n";
		return true;
	}

	return false;
}

int main (int argc, char** argv)
{
	// Process parameters
	boost::program_options::variables_map variablesMap;
	if (parse_command_line_options(variablesMap, argc, argv))
		return 1;

	bool online = variablesMap[PARAM_ONLINE].as<bool>();

	Recognizer *recognizer = new Recognizer(variablesMap);
	recognizer->load_models();

	if (!online)
	{
		// Get parameter values
		std::string sceneFilename = variablesMap[PARAM_SCENE_FILENAME].as<std::string>();

		pcl::PointCloud<TPoint>::Ptr originalScene(new pcl::PointCloud<TPoint>());

		// Load scene cloud
		std::cout << "Loading scene cloud...\n";

		if (pcl::io::loadPCDFile(sceneFilename, *originalScene) < 0)
		{
			std::cerr << "Error loading scene cloud " << sceneFilename << "\n";
			return -1;
		}

		recognizer->recognize(originalScene);
	}
	else
	{
		Grabber grabber(recognizer);
		grabber.run();
	}

	delete recognizer;
	return 0;
}
