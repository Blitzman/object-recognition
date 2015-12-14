#include "Recognizer.h"

Recognizer::Recognizer(const boost::program_options::variables_map & variablesMap)
{
	online 			= false;
	modelDirectory		= variablesMap[PARAM_MODEL_DIRECTORY].as<std::string>();
	sceneFilename		= variablesMap[PARAM_SCENE_FILENAME].as<std::string>();

	showNormals		= variablesMap[PARAM_VISUALIZER_SHOW_NORMALS].as<bool>();
	showKeypoints		= variablesMap[PARAM_VISUALIZER_SHOW_KEYPOINTS].as<bool>();
	showCorrespondences	= variablesMap[PARAM_VISUALIZER_SHOW_CORRESPONDENCES].as<bool>();
	showHypotheses		= variablesMap[PARAM_VISUALIZER_SHOW_HYPOTHESES].as<bool>();
	showDescriptors		= variablesMap[PARAM_VISUALIZER_SHOW_DESCRIPTORS].as<bool>();

	useCloudResolution	= variablesMap[PARAM_USE_CLOUD_RESOLUTION].as<bool>();

	applyBilateralFilter	= variablesMap[PARAM_BILATERAL_FILTER].as<bool>();
	bilateralFilterSigmaR 	= variablesMap[PARAM_BILATERAL_FILTER_SIGMA_R].as<float>();
	bilateralFilterSigmaS 	= variablesMap[PARAM_BILATERAL_FILTER_SIGMA_S].as<float>();

	normalEstimationK	= variablesMap[PARAM_NORMAL_ESTIMATION_K].as<int>();

	useMPS			= variablesMap[PARAM_SEGMENTATION].as<bool>();
	mpsMinInliers		= variablesMap[PARAM_SEGMENTATION_MININLIERS].as<int>();
	mpsAngThres		= variablesMap[PARAM_SEGMENTATION_ANGULARTHRESHOLD].as<double>();
	mpsDstThres		= variablesMap[PARAM_SEGMENTATION_DISTANCETHRESHOLD].as<double>();

	downsampleModel		= variablesMap[PARAM_DOWNSAMPLE_MODEL].as<bool>();
	downsampleModelLeaf	= variablesMap[PARAM_DOWNSAMPLE_MODEL_LEAF_SIZE].as<float>();
	downsampleScene		= variablesMap[PARAM_DOWNSAMPLE_SCENE].as<bool>();
	downsampleSceneLeaf	= variablesMap[PARAM_DOWNSAMPLE_SCENE_LEAF_SIZE].as<float>();

	useISS			= variablesMap[PARAM_KEYPOINTS_ISS].as<bool>();
	issSalientRadius	= variablesMap[PARAM_KEYPOINTS_ISS_SALIENTRADIUS].as<float>();
	issNonMaxRadius		= variablesMap[PARAM_KEYPOINTS_ISS_NONMAXRADIUS].as<float>();
	issMinNeighbors		= variablesMap[PARAM_KEYPOINTS_ISS_MINEIGHBORS].as<int>();
	issThreshold21		= variablesMap[PARAM_KEYPOINTS_ISS_THRESHOLD21].as<float>();
	issThreshold32		= variablesMap[PARAM_KEYPOINTS_ISS_THRESHOLD32].as<float>();

	useSIFT			= variablesMap[PARAM_KEYPOINTS_SIFT].as<bool>();
	siftMinimumScale	= variablesMap[PARAM_KEYPOINTS_SIFT_MINSCALE].as<float>();
	siftNumberOfOctaves	= variablesMap[PARAM_KEYPOINTS_SIFT_NROCTAVES].as<int>();
	siftScalesPerOctave	= variablesMap[PARAM_KEYPOINTS_SIFT_SCALESPEROCT].as<int>();
	siftMinimumContrast	= variablesMap[PARAM_KEYPOINTS_SIFT_MINCONTRAST].as<float>();

	modelSearchRadius	= variablesMap[PARAM_MODEL_SEARCH_RADIUS].as<float>();
	sceneSearchRadius	= variablesMap[PARAM_SCENE_SEARCH_RADIUS].as<float>();

	descriptorRadius	= variablesMap[PARAM_DESCRIPTOR_RADIUS].as<float>();
	descriptorLRFRadius	= variablesMap[PARAM_DESCRIPTOR_LRF_RADIUS].as<float>();

	matchingThreshold	= variablesMap[PARAM_MATCHING_THRESHOLD].as<float>();

	useHough3D		= variablesMap[PARAM_GROUPING_USE_HOUGH].as<bool>();
	clusterSize		= variablesMap[PARAM_GROUPING_CLUSTER_SIZE].as<float>();
	clusterThreshold	= variablesMap[PARAM_GROUPING_CLUSTER_THRESHOLD].as<float>();

	useICP			= variablesMap[PARAM_ICP].as<bool>();
	icpMaxIterations	= variablesMap[PARAM_ICP_MAX_ITERATIONS].as<int>();
	icpMaxCorrDistance	= variablesMap[PARAM_ICP_MAX_CORRESPONDENCE_DISTANCE].as<float>();

	hvClutterRegularizer	= variablesMap[PARAM_HV_CLUTTER_REGULARIZER].as<float>();
	hvInlierThreshold	= variablesMap[PARAM_HV_INLIER_THRESHOLD].as<float>();
	hvOcclusionThreshold	= variablesMap[PARAM_HV_OCCLUSION_THRESHOLD].as<float>();
	hvRadiusClutter		= variablesMap[PARAM_HV_RADIUS_CLUTTER].as<float>();
	hvRegularizer		= variablesMap[PARAM_HV_REGULARIZER].as<float>();
	hvRadiusNormals		= variablesMap[PARAM_HV_RADIUS_NORMALS].as<float>();
	hvDetectClutter		= variablesMap[PARAM_HV_DETECT_CLUTTER].as<bool>();
}

Recognizer::~Recognizer()
{

}

void Recognizer::load_models()
{
	// Get model PCD files from the specified directory
	Utils::get_clouds_filenames(modelDirectory, modelFilenames);
	int numModels = modelFilenames.size();

	std::cout << "Loading model clouds...\n";

	for (std::vector<std::string>::iterator it = modelFilenames.begin(); it != modelFilenames.end(); ++it)
	{
		std::cout << "Loading model: " << (*it) << "\n";

		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr modelWithNormals(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
		pcl::PointCloud<TPoint>::Ptr originalModel(new pcl::PointCloud<TPoint>());
		pcl::PointCloud<TNormal>::Ptr modelNormals(new pcl::PointCloud<TNormal>());

		if (pcl::io::loadPCDFile(*it, *modelWithNormals) < 0)
		{
			std::cerr << "Error loading model cloud " << (*it) << "\n";
			exit(-1);
		}

		pcl::copyPointCloud(*modelWithNormals, *originalModel);
		pcl::copyPointCloud(*modelWithNormals, *modelNormals);

		originalModels.push_back(originalModel);
		modelsNormals.push_back(modelNormals);
	}
}

void Recognizer::recognize(const pcl::PointCloud<TPoint>::Ptr & sceneCloud)
{
	//viewer.removeAllPointClouds();
	//pcl::visualization::PCLVisualizer correspondencesViewer("Correspondences Viewer");

	pcl::visualization::PCLVisualizer viewer("Object recognition");
	viewer.addPointCloud(sceneCloud, "scene");

	//pcl::io::savePCDFile("recognized_cloud.pcd", *sceneCloud);
	pcl::PointCloud<TPoint>::Ptr originalScene(new pcl::PointCloud<TPoint>());
	pcl::PointCloud<TPoint>::Ptr scene(new pcl::PointCloud<TPoint>());
	pcl::PointCloud<TNormal>::Ptr sceneNormals(new pcl::PointCloud<TNormal>());
	pcl::PointCloud<TPoint>::Ptr sceneKeypoints(new pcl::PointCloud<TPoint>());
	pcl::PointCloud<TDescriptor>::Ptr sceneDescriptors(new pcl::PointCloud<TDescriptor>());

	// Scene operations ----------------------------------------------
	pcl::copyPointCloud(*sceneCloud, *originalScene);

	// Apply bilateral filter to scene
	if (applyBilateralFilter)
		PointCloudOperations::filter_bilateral(
			originalScene,
			bilateralFilterSigmaR,
			bilateralFilterSigmaS,
			originalScene);

	// Downsample scene
	if (downsampleScene)
		PointCloudOperations::downsample_voxel_grid(
			originalScene,
			scene,
			downsampleSceneLeaf);
	else
		pcl::copyPointCloud(*originalScene, *scene);

	// Compute scene normals
	PointCloudOperations::compute_normals_organized(
		scene,
		sceneNormals,
		normalEstimationK);

	// Segment objects
	if (useMPS)
	{
		// Apply OMPS and OCCS segmentation
		PointCloudOperations::segment_objects(
			scene,
			sceneNormals,
			mpsMinInliers,
			mpsAngThres,
			mpsDstThres,
			scene);

		// Apply bounding box segmentation
		float xCenter = 0.087109f;
		float yCenter = 0.065120f;
		float zCenter = 0.888000f;
		float xDeviation = 0.4f;
		float yDeviation = 0.4f;
		float zDeviation = 0.4f;

		PointCloudOperations::bounding_box(
			scene,
			xCenter - xDeviation,
			xCenter + xDeviation,
			yCenter - yDeviation,
			yCenter + yDeviation,
			zCenter - zDeviation,
			zCenter + zDeviation,
			scene);

		std::vector<int> mapping;
		pcl::removeNaNFromPointCloud(*scene, *scene, mapping);

		// Recompute normals
		sceneNormals->clear();
		PointCloudOperations::compute_normals(
			scene,
			sceneNormals,
			normalEstimationK);
	}

	// Compute scene resolution
	float sceneResolution = static_cast<float>(PointCloudOperations::compute_resolution(scene));

	float currentSceneSearchRadius = sceneSearchRadius;
	float currentSceneDescriptorRadius = descriptorRadius;
	float currentSceneDescriptorLRFRadius = descriptorLRFRadius;

	// Set up resolution invariance
	if (useCloudResolution)
	{
		// Set up resolution invariance
		if (sceneResolution > 0.0f)
		{
			//normalEstimationRad *= modelResolution;
			currentSceneSearchRadius *= sceneResolution;
			currentSceneDescriptorRadius *= sceneResolution;
			currentSceneDescriptorLRFRadius *= sceneResolution;
		}
	}

	currentSceneSearchRadius = 0.00153581f * 4.0f;
	currentSceneDescriptorRadius = 0.00153581f * 20.0f;

	/*PointCloudOperations::detect_keypoints_iss(
		scene,
		issSalientRadius,
		issNonMaxRadius,
		issMinNeighbors,
		issThreshold21,
		issThreshold32,
		sceneKeypoints);*/
	PointCloudOperations::downsample_uniform_sampling(
		scene,
		sceneKeypoints,
		currentSceneSearchRadius);

	PointCloudOperations::compute_shot_descriptors(
		scene,
		sceneNormals,
		sceneKeypoints,
		currentSceneDescriptorRadius,
		sceneDescriptors);

	// Set up scene viewer
	std::cout << "Setting up scene viewer...\n";

	//pcl::visualization::PCLVisualizer sceneViewer("Scene viewer");
	//sceneViewer.addPointCloud(scene, "scene");

	if (showNormals)
		//sceneViewer.addPointCloudNormals<TPoint, TNormal>(scene, sceneNormals, 50, 0.025, "sceneNormals");

	if (showKeypoints)
	{
		pcl::visualization::PointCloudColorHandlerCustom<TPoint> sceneKeypointsColorhandler(sceneKeypoints, 0, 0, 255);
		//sceneViewer.addPointCloud(sceneKeypoints, sceneKeypointsColorhandler, "sceneKeypoints");
	}

	// Set up object recognition viewer
	std::cout << "Setting up object recognition viewer...\n";


// Recognize models one by one
	for (int i = 0; i < modelFilenames.size(); ++i)
	{
		std::cout << "*** Recognizing model " << i << " " << modelFilenames[i] << "\n";

		float currentModelSearchRadius		= modelSearchRadius;
		float currentSceneSearchRadius		= sceneSearchRadius;
		float currentDescriptorRadius		= descriptorRadius;
		float currentDescriptorLRFRadius	= descriptorLRFRadius;
		float currentClusterSize		= clusterSize;
		float currentClusterThreshold		= clusterThreshold;
		float currentHvInlierThreshold		= hvInlierThreshold;

		Settings objectSettings(modelFilenames[i] + ".cfg");
		objectSettings.add_option(PARAM_MODEL_SEARCH_RADIUS, currentModelSearchRadius, "");
		objectSettings.add_option(PARAM_SCENE_SEARCH_RADIUS, currentSceneSearchRadius, "");
		objectSettings.add_option(PARAM_DESCRIPTOR_RADIUS, currentDescriptorRadius, "");
		objectSettings.add_option(PARAM_DESCRIPTOR_LRF_RADIUS, currentDescriptorLRFRadius, "");
		objectSettings.add_option(PARAM_GROUPING_CLUSTER_SIZE, currentClusterSize, "");
		objectSettings.add_option(PARAM_GROUPING_CLUSTER_THRESHOLD, currentClusterThreshold, "");
		objectSettings.add_option(PARAM_HV_INLIER_THRESHOLD, currentHvInlierThreshold, "");

		if (!objectSettings.read_settings())
		{
			std::cerr << "Error reading model settings...\n";
			exit(-1);
		}

		pcl::PointCloud<TPoint>::Ptr model(new pcl::PointCloud<TPoint>());
		pcl::PointCloud<TPoint>::Ptr modelKeypoints(new pcl::PointCloud<TPoint>());
		pcl::PointCloud<TDescriptor>::Ptr modelDescriptors(new pcl::PointCloud<TDescriptor>());

		if (!online)
		{
			std::cout << "Model search radius: " << currentModelSearchRadius << "\n";
			std::cout << "Scene search radius: " << currentSceneSearchRadius << "\n";
			std::cout << "Descriptor radius: " << currentDescriptorRadius << "\n";
			std::cout << "Descriptor LRF radius: " << currentDescriptorLRFRadius << "\n";
			std::cout << "Cluster size: " << currentClusterSize << "\n";
			std::cout << "Cluster threshold: " << currentClusterThreshold << "\n";
			std::cout << "Inlier threshold: " << currentHvInlierThreshold << "\n";


			// Downsample model cloud
			if (downsampleModel)
				PointCloudOperations::downsample_voxel_grid(
					originalModels[i],
					model,
					downsampleModelLeaf);
			else
				pcl::copyPointCloud(*originalModels[i], *model);

			models.push_back(model);

			// Compute cloud resolution
			float modelResolution = static_cast<float>(PointCloudOperations::compute_resolution(models[i]));

			// Set up resolution invariance
			if (useCloudResolution)
			{
				// Set up resolution invariance
				if (modelResolution != 0.0f)
				{
					//normalEstimationRad *= modelResolution;
					currentModelSearchRadius *= modelResolution;
					currentSceneSearchRadius *= modelResolution;
					currentDescriptorRadius *= modelResolution;
					currentDescriptorLRFRadius *= modelResolution;
					currentClusterSize *= modelResolution;
				}
			}

			// Detect keypoints
			/*PointCloudOperations::detect_keypoints_iss(
				models[i],
				issSalientRadius,
				issNonMaxRadius,
				issMinNeighbors,
				issThreshold21,
				issThreshold32,
				modelKeypoints);*/
			PointCloudOperations::downsample_uniform_sampling(
				models[i],
				modelKeypoints,
				currentModelSearchRadius);

			//pcl::io::savePCDFile(modelFilenames[i] + "_keypoints.pcd", *modelKeypoints);

			// Compute descriptors
			PointCloudOperations::compute_shot_descriptors(
				models[i],
				modelsNormals[i],
				modelKeypoints,
				currentDescriptorRadius,
				modelDescriptors);

			//pcl::io::savePCDFile(modelFilenames[i] + "_descriptors.pcd", *modelDescriptors);
		}

		pcl::CorrespondencesPtr modelSceneCorrespondences(new pcl::Correspondences());
		std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rotoTranslations;
		std::vector<pcl::Correspondences> clusteredCorrespondences;
		std::vector<pcl::PointCloud<TPoint>::ConstPtr> instances;
		std::vector<bool> hypothesesMask;
		std::vector<pcl::PointCloud<TPoint>::ConstPtr> registeredInstances;

		// Find model-scene correspondences
		PointCloudOperations::find_correspondences_multicore(
			modelDescriptors,
			sceneDescriptors,
			matchingThreshold,
			modelSceneCorrespondences);

		// Cluster correspondences
		PointCloudOperations::cluster_correspondences(
			models[i],
			modelsNormals[i],
			modelKeypoints,
			scene,
			sceneNormals,
			sceneKeypoints,
			modelSceneCorrespondences,
			useHough3D,
			currentClusterSize,
			currentClusterThreshold,
			currentDescriptorLRFRadius,
			rotoTranslations,
			clusteredCorrespondences);

		if (rotoTranslations.size() > 0)
		{
			std::cout << "Aligning detected instances...\n";

			// Store rotated detected instances
			for (size_t j = 0; j < rotoTranslations.size(); ++j)
			{
				std::cout << "Aligning instance " << j << "...\n";

				pcl::PointCloud<TPoint>::Ptr rotatedModel(new pcl::PointCloud<TPoint>());
				pcl::transformPointCloud(*models[i], *rotatedModel, rotoTranslations[j]);
				instances.push_back(rotatedModel);
			}

			std::cout << "Registering instances...\n";

			// Register instances
			if (useICP)
			{
				for (size_t j = 0; j < rotoTranslations.size(); ++j)
				{
					pcl::PointCloud<TPoint>::Ptr registeredCloud(new pcl::PointCloud<TPoint>);

					PointCloudOperations::register_cloud(
						instances[j],
						scene,
						icpMaxIterations,
						icpMaxCorrDistance,
						registeredCloud);

					registeredInstances.push_back(registeredCloud);
				}
			}
			else
			{
				registeredInstances = instances;
			}

			// Hypothesis verification
			std::cout << "Verifying instances...\n";
			PointCloudOperations::verify_hypotheses(
				scene,
				registeredInstances,
				currentHvInlierThreshold,
				hvOcclusionThreshold,
				hvRadiusClutter,
				hvRegularizer,
				hvClutterRegularizer,
				hvDetectClutter,
				hvRadiusNormals,
				hypothesesMask);

			if (showCorrespondences)
			{
				//correspondencesViewer.addPointCloud(scene, "scene");
				pcl::PointCloud<TPoint>::Ptr offSceneModel(new pcl::PointCloud<TPoint>);
				pcl::PointCloud<TPoint>::Ptr offSceneModelKeypoints(new pcl::PointCloud<TPoint>);

				pcl::transformPointCloud(*models[i], *offSceneModel, Eigen::Vector3f(-1,0,0), Eigen::Quaternionf(1, 0, 0, 0));
				pcl::transformPointCloud(*modelKeypoints, *offSceneModelKeypoints, Eigen::Vector3f(-1,0,0), Eigen::Quaternionf(1, 0, 0, 0));

				pcl::visualization::PointCloudColorHandlerCustom<TPoint> offSceneModelKeypointsColorHandler(offSceneModelKeypoints, 0, 0, 255);
				//correspondencesViewer.addPointCloud(offSceneModelKeypoints, offSceneModelKeypointsColorHandler, "offscenemodelKeypoints");
				//correspondencesViewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "offscenemodelKeypoints");

				for (size_t k = 0; k < rotoTranslations.size(); ++k)
				{
					for (size_t m = 0; m < clusteredCorrespondences[k].size(); ++m)
					{
						std::stringstream ss_line;
						ss_line << "correspondence_line" << k << "_" << m;
						TPoint &model_point = offSceneModelKeypoints->at(clusteredCorrespondences[k][m].index_query);
						TPoint &scene_point = sceneKeypoints->at(clusteredCorrespondences[k][m].index_match);
						//correspondencesViewer.addLine<TPoint, TPoint>(model_point, scene_point, 0, 255, 0, ss_line.str());
					}
				}

				//correspondencesViewer.addCoordinateSystem(1.0);

				//while (!correspondencesViewer.wasStopped())
					//correspondencesViewer.spinOnce();
			}

		}

		std::cout << "Adding model to viewer...\n";

		// Visualization
		//pcl::visualization::PCLVisualizer modelViewer(modelFilenames[i]);
		//modelViewer.addPointCloud(models[i], modelFilenames[i]);

		std::cout << "Adding normals to viewer...\n";

		if (showNormals)
			//modelViewer.addPointCloudNormals<TPoint, TNormal>(models[i], modelsNormals[i], 100, 0.05, modelFilenames[i] + " normals");

		std::cout << "Adding keypoints to viewer...\n";

		if (showKeypoints)
		{
			pcl::visualization::PointCloudColorHandlerCustom<TPoint> modelKeypointsColorhandler(modelKeypoints, 0, 0, 255);
			//modelViewer.addPointCloud(modelKeypoints, modelKeypointsColorhandler, modelFilenames[i] + " keypoints");
		}

		std::cout << "Adding descriptors to viewer...\n";

		if (showDescriptors)
		{
			int sphPerKp = modelKeypoints->size() / 100;
			for (int j = 0; j < sphPerKp; ++j)
			{
				std::stringstream ssLine;
				ssLine << (modelFilenames[i] + " modelLRF") << j;
				//modelViewer.addSphere(modelKeypoints->at(j*sphPerKp), currentDescriptorRadius, ssLine.str(), 0);
				//modelViewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, ssLine.str());
			}
		}

		//modelViewer.spin();

		std::cout << "Adding instances to viewer...\n";

		std::cout << "Instances: " << registeredInstances.size() << "\n";
		std::cout << "Hypotheses: " << hypothesesMask.size() << "\n";

		for (size_t j = 0; j < registeredInstances.size(); ++j)
		{
			//std::cout << "Processing instance " << i << "...\n";

			std::stringstream ssCloud;
			ssCloud << (modelFilenames[i] + " instance") << j;

			//std::cout << "Create color handler...\n";

			int color = hypothesesMask[j] ? 255 : 32;

			//std::cout << "Color assigned...\n";
			pcl::visualization::PointCloudColorHandlerCustom<TPoint> rotatedModelColorHandler(registeredInstances[j], color, 0, 0);

			//std::cout << "Color handler created...\n";

			if (showHypotheses || hypothesesMask[j])
			{
				std::cout << "Adding instance " << j << " to viewer...\n";
				viewer.addPointCloud(registeredInstances[j], rotatedModelColorHandler, ssCloud.str());
				std::cout << "Instance added...\n";
			}
		}
	}

	std::cout << "Running viewer...\n";

	//sceneViewer.spinOnce();
	while (!viewer.wasStopped())
		viewer.spinOnce();

}
