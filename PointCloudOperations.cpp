#include "PointCloudOperations.h"

void PointCloudOperations::bounding_box
	(
		const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
		const float & pLimitMinX,
		const float & pLimitMaxX,
		const float & pLimitMinY,
		const float & pLimitMaxY,
		const float & pLimitMinZ,
		const float & pLimitMaxZ,
		pcl::PointCloud<TPoint>::Ptr & pDstCloud
	)
{
	pcl::PointCloud<TPoint>::Ptr filtered_cloud_z(new pcl::PointCloud<TPoint>());
	pcl::PointCloud<TPoint>::Ptr filtered_cloud_x(new pcl::PointCloud<TPoint>());

	pcl::PassThrough<TPoint> pass;
	pass.setInputCloud(pSrcCloud);
	pass.setKeepOrganized(true);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(pLimitMinZ, pLimitMaxZ);
	pass.filter(*filtered_cloud_z);

	pass.setInputCloud(filtered_cloud_z);
	pass.setFilterFieldName("x");
	pass.setFilterLimits(pLimitMinX, pLimitMaxX);
	pass.filter(*filtered_cloud_x);

	pass.setInputCloud(filtered_cloud_x);
	pass.setFilterFieldName("y");
	pass.setFilterLimits(pLimitMinY, pLimitMaxY);
	pass.filter(*pDstCloud);
}

void PointCloudOperations::filter_bilateral
	(
		const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
		const float & pSigmaR,
		const float & pSigmaS,
		pcl::PointCloud<TPoint>::Ptr &pDstCloud
	)
{
#ifdef DEBUG
	std::cout << "Filtering cloud with bilateral filter...\n";
#endif

	boost::timer::cpu_timer t;

	pcl::FastBilateralFilterOMP<TPoint> bf;
	bf.setInputCloud(pSrcCloud);
	bf.setSigmaR(pSigmaR);
	bf.setSigmaS(pSigmaS);
	bf.applyFilter(*pDstCloud);

	boost::timer::cpu_times elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Cloud filtered...\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " s\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " s\n";
#endif
}

void PointCloudOperations::compute_normals_organized
	(
		const pcl::PointCloud<TPoint>::ConstPtr & pCloud,
		pcl::PointCloud<TNormal>::Ptr & pNormals,
		const int & pNeighbors
	)
{
#ifdef DEBUG
	std::cout << "Estimating normals for organized point cloud...\n";
#endif

	boost::timer::cpu_timer t;

	pcl::IntegralImageNormalEstimation<TPoint, TNormal> ne;
	ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
	ne.setMaxDepthChangeFactor(0.02f);
	ne.setNormalSmoothingSize(10.0f);
	ne.setInputCloud(pCloud);
	ne.compute(*pNormals);

	boost::timer::cpu_times elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Normals estimated (" << pNormals->size() << ")...\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " s\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " s\n";
#endif
}

void PointCloudOperations::compute_normals
	(
		const pcl::PointCloud<TPoint>::ConstPtr & pCloud,
		pcl::PointCloud<TNormal>::Ptr & pNormals,
		const int & pNeighbors
	)
{
#ifdef DEBUG
	std::cout << "Estimating normals...\n";
#endif

	boost::timer::cpu_timer t;

	pcl::NormalEstimation<TPoint, TNormal> ne;
	pcl::search::KdTree<TPoint>::Ptr kdtree(new pcl::search::KdTree<TPoint>);
	ne.setSearchMethod(kdtree);
	//ne.setRadiusSearch(pRadius);
	ne.setKSearch(pNeighbors);
	ne.setInputCloud(pCloud);
	ne.compute(*pNormals);

	boost::timer::cpu_times elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Normals estimated (" << pNormals->size() << ") ...\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " s\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " s\n";
#endif
}

double PointCloudOperations::compute_resolution
	(
		const pcl::PointCloud<TPoint>::ConstPtr & pCloud
	)
{
#ifdef DEBUG
	std::cout << "Computing cloud resolution...\n";
#endif

	boost::timer::cpu_timer t;

	double resolution = 0.0;
	int points = 0;
	int nres;

	std::vector<int> indices(2);
	std::vector<float> sqrDistances(2);
	pcl::search::KdTree<TPoint> kdtree;
	kdtree.setInputCloud(pCloud);

	for (size_t i = 0; i < pCloud->size(); ++i)
	{
		if (!pcl_isfinite((*pCloud)[i].x))
			continue;

		nres = kdtree.nearestKSearch(i, 2, indices, sqrDistances);

		if (nres == 2)
		{
			resolution += sqrt(sqrDistances[1]);
			++points;
		}
	}

	if (points != 0)
		resolution /= points;

	boost::timer::cpu_times elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Cloud resolution is " << resolution << "\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";
#endif

	return resolution;
}

void PointCloudOperations::segment_objects
	(
		const pcl::PointCloud<TPoint>::ConstPtr & pCloud,
		const pcl::PointCloud<TNormal>::ConstPtr & pNormals,
		const int & pMinInliers,
		const double & pAngularThreshold,
		const double & pDistanceThreshold,
		pcl::PointCloud<TPoint>::Ptr & pDstCloud
	)
{
	std::cout << "Segmenting regions with OMPS...\n";

	boost::timer::cpu_timer t;

	pcl::IntegralImageNormalEstimation<TPoint, pcl::Normal> ne;
	ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
	ne.setMaxDepthChangeFactor(0.02f);
	ne.setNormalSmoothingSize(20.0f);
	pcl::PointCloud<pcl::Normal>::Ptr normalCloud(new pcl::PointCloud<pcl::Normal>);
	ne.setInputCloud(pCloud);
	ne.compute(*normalCloud);
	float* distance_map = ne.getDistanceMap();

	// Segment planes
	pcl::OrganizedMultiPlaneSegmentation<TPoint, TNormal, pcl::Label> mps;
	mps.setMinInliers(pMinInliers);
	mps.setAngularThreshold(pAngularThreshold*0.017453);
	mps.setDistanceThreshold(pDistanceThreshold);
	mps.setInputNormals(normalCloud);
	mps.setInputCloud(pCloud);

	std::vector<pcl::PlanarRegion<TPoint>, Eigen::aligned_allocator<pcl::PlanarRegion<TPoint> > > regions;
	std::vector<pcl::ModelCoefficients> modelCoefficients;
	std::vector<pcl::PointIndices> inlierIndices;
	pcl::PointCloud<pcl::Label>::Ptr labels(new pcl::PointCloud<pcl::Label>);
	std::vector<pcl::PointIndices> labelIndices;
	std::vector<pcl::PointIndices> boundaryIndices;

	mps.segmentAndRefine(regions, modelCoefficients, inlierIndices, labels, labelIndices, boundaryIndices);

	cout << regions.size() << " regions found...\n";

	boost::timer::cpu_times elapsedTime = t.elapsed();
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << "seconds \n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << "seconds \n";

	t.start();

	// Segment objects
	if (regions.size() > 0)
	{
		cout << "Segmenting components with OCCS...\n";

		pcl::PointCloud<TPoint>::CloudVectorType clusters;

		std::vector<bool> planeLabels;
		planeLabels.resize(labelIndices.size(), false);
		for (size_t i = 0; i < labelIndices.size(); i++)
		{
			if (labelIndices[i].indices.size() > pMinInliers *10)
			{
				planeLabels[i] = true;
			}
		}

		pcl::PointCloud<pcl::Label> euclideanLabels;
		std::vector<pcl::PointIndices> euclideanLabelIndices;
		pcl::EuclideanClusterComparator<TPoint, TNormal, pcl::Label>::Ptr ecc(new pcl::EuclideanClusterComparator<TPoint, TNormal, pcl::Label>());
		ecc->setInputCloud(pCloud);
		ecc->setLabels(labels);
		ecc->setExcludeLabels(planeLabels);
		ecc->setDistanceThreshold(pDistanceThreshold, false);
		pcl::OrganizedConnectedComponentSegmentation<TPoint, pcl::Label> occs(ecc);
		occs.setInputCloud(pCloud);
		occs.segment(euclideanLabels, euclideanLabelIndices);

		//pcl::visualization::PCLVisualizer clusterViewer("Cluster viewer");

		cout << "Discarding clusters...\n";
		pcl::common::UniformGenerator<int> rng(0, 255, 1);
		for (size_t i = 0; i < euclideanLabelIndices.size(); ++i)
		{
			if (euclideanLabelIndices[i].indices.size() > pMinInliers)
			{
				pcl::PointCloud<TPoint>::Ptr cluster(new pcl::PointCloud<TPoint>());
				pcl::copyPointCloud(*pCloud, euclideanLabelIndices[i].indices, *cluster);
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color(cluster, rng.run(), rng.run(), rng.run());
				std::stringstream ssLine;
				ssLine << "cluster" << i;
				//clusterViewer.addPointCloud(cluster, single_color, ssLine.str());
				clusters.push_back(*cluster);
			}
		}

		// Copy clusters
		pDstCloud->clear();
		for (size_t i = 0; i < clusters.size(); ++i)
			*pDstCloud += clusters[i];
	}

	elapsedTime = t.elapsed();
	std::cout << pDstCloud->size() << " points in segmented cloud...\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << "seconds \n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " seconds \n";
}

void PointCloudOperations::downsample_uniform_sampling
	(
		const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
		pcl::PointCloud<TPoint>::Ptr &pDstCloud,
		const float & pRadiusSearch
	)
{
#ifdef DEBUG
	std::cout << "Downsampling point cloud (" << pSrcCloud->points.size() << " points) with uniform sampling...\n";
#endif

	boost::timer::cpu_timer t;

	pcl::PointCloud<int> sampledIndices;

	pcl::UniformSampling<TPoint> us;
	us.setInputCloud(pSrcCloud);
	us.setRadiusSearch(pRadiusSearch);
	us.compute(sampledIndices);

	pcl::copyPointCloud(*pSrcCloud, sampledIndices.points, *pDstCloud);

	boost::timer::cpu_times elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Cloud downsampled (" << pDstCloud->points.size() << " points)...\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";
#endif
}

void PointCloudOperations::detect_keypoints_iss
	(
		const pcl::PointCloud<TPoint>::ConstPtr & pCloud,
		const float & pSalientRadius,
		const float & pNonMaxRadius,
		const int & pMinNeighbors,
		const float & pThreshold21,
		const float & pThreshold32,
		pcl::PointCloud<TPoint>::Ptr & pDstCloud
	)
{
#ifdef DEBUG
	std::cout << "Detecting keypoints from point cloud (" << pCloud->points.size() << " points) with ISS...\n";
#endif

	boost::timer::cpu_timer t;

	pcl::ISSKeypoint3D<pcl::PointXYZRGBA, pcl::PointXYZRGBA> iss;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZRGBA>);

	pcl::copyPointCloud(*pCloud, *cloud);
	iss.setInputCloud(cloud);

	pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGBA>);
	iss.setSearchMethod(kdtree);

	double resolution = compute_resolution(pCloud);

	iss.setSalientRadius(pSalientRadius * resolution);
	iss.setNonMaxRadius(pNonMaxRadius * resolution);
	iss.setMinNeighbors(pMinNeighbors);
	iss.setThreshold21(pThreshold21);
	iss.setThreshold32(pThreshold32);
	iss.setNumberOfThreads(4);
	iss.compute(*keypoints);

	pcl::copyPointCloud(*keypoints, *pDstCloud);

	boost::timer::cpu_times elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Keypoints detected (" << pDstCloud->points.size() << " points)...\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";
#endif
}

void PointCloudOperations::detect_keypoints_sift
	(
		const pcl::PointCloud<TPoint>::ConstPtr & pCloud,
		const float & pMinScale,
		const float & pNrOctaves,
		const float & pNrScalesPerOctave,
		const float & pMinConstrast,
		pcl::PointCloud<TPoint>::Ptr & pDstCloud
	)
{

#ifdef DEBUG
	std::cout << "Downsampling point cloud (" << pCloud->points.size() << " points) with SIFT...\n";
#endif

	boost::timer::cpu_timer t;

	pcl::SIFTKeypoint<TPoint, pcl::PointWithScale> sift;
	sift.setSearchMethod(pcl::search::KdTree<TPoint>::Ptr (new pcl::search::KdTree<TPoint>));
	sift.setScales(pMinScale, pNrOctaves, pNrScalesPerOctave);
	sift.setMinimumContrast(pMinConstrast);
	pcl::PointCloud<pcl::PointWithScale>::Ptr keypointsWithScale(new pcl::PointCloud<pcl::PointWithScale>);
	sift.setInputCloud(pCloud);
	sift.compute(*keypointsWithScale);
	pcl::copyPointCloud(*keypointsWithScale, *pDstCloud);

	boost::timer::cpu_times elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Cloud downsampled (" << pDstCloud->points.size() << " points)...\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";
#endif
}

void PointCloudOperations::downsample_voxel_grid
	(
		const pcl::PointCloud<TPoint>::Ptr & pSrcCloud,
		pcl::PointCloud<TPoint>::Ptr & pDstCloud,
		const float & pLeafSize
	)
{
#ifdef DEBUG
	std::cout << "Voxelizing cloud(" << pSrcCloud->points.size() << ")...\n";
#endif

	boost::timer::cpu_timer t;

	pcl::VoxelGrid<pcl::PointXYZRGB> vg;
	vg.setInputCloud(pSrcCloud);
	vg.setLeafSize(pLeafSize, pLeafSize, pLeafSize);
	vg.filter(*pDstCloud);

	boost::timer::cpu_times elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Cloud voxelized(" << pDstCloud->points.size() << ")...\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";
#endif
}

void PointCloudOperations::compute_shot_descriptors
	(
		const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
		const pcl::PointCloud<TNormal>::ConstPtr & pNormals,
		const pcl::PointCloud<TPoint>::ConstPtr & pKeypoints,
		const float & pSearchRadius,
		pcl::PointCloud<pcl::SHOT352>::Ptr & pDstCloud
	)
{
#ifdef DEBUG
	std::cout << "Computing SHOT descriptors...\n";
#endif

	boost::timer::cpu_timer t;

	pcl::SHOTEstimationOMP<TPoint, TNormal, pcl::SHOT352> shot;
	shot.setSearchMethod(pcl::search::KdTree<TPoint>::Ptr(new pcl::search::KdTree<TPoint>));
	shot.setInputCloud(pKeypoints);
	shot.setInputNormals(pNormals);
	shot.setRadiusSearch(pSearchRadius);
	shot.setSearchSurface(pSrcCloud);
//	shot.setNumberOfThreads(4);
	shot.compute(*pDstCloud);

	boost::timer::cpu_times elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Computed " << pDstCloud->points.size() << " descriptors...\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";
#endif
}

void PointCloudOperations::compute_cshot_descriptors
	(
		const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
		const pcl::PointCloud<TNormal>::ConstPtr & pNormals,
		const pcl::PointCloud<TPoint>::ConstPtr & pKeypoints,
		const float & pSearchRadius,
		pcl::PointCloud<pcl::SHOT1344>::Ptr & pDstCloud
	)
{
#ifdef DEBUG
	std::cout << "Computing CSHOT descriptors...\n";
#endif

	boost::timer::cpu_timer t;

	pcl::SHOTColorEstimation<TPoint, TNormal, pcl::SHOT1344> shot;
	shot.setSearchMethod(pcl::search::KdTree<TPoint>::Ptr(new pcl::search::KdTree<TPoint>));
	shot.setInputCloud(pKeypoints);
	shot.setInputNormals(pNormals);
	shot.setRadiusSearch(pSearchRadius);
	shot.setSearchSurface(pSrcCloud);
	//shot.setNumberOfThreads(4);
	shot.compute(*pDstCloud);

	boost::timer::cpu_times elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Computed " << pDstCloud->points.size() << " descriptors...\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";
#endif
}


void PointCloudOperations::compute_fpfh_descriptors
	(
		const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
		const pcl::PointCloud<TNormal>::ConstPtr & pNormals,
		const pcl::PointCloud<TPoint>::ConstPtr & pKeypoints,
		const float & pSearchRadius,
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr & pDstCloud
	)
{
#ifdef DEBUG
	std::cout << "Computing FPFH descriptors...\n";
#endif

	boost::timer::cpu_timer t;

	pcl::FPFHEstimation<TPoint, TNormal, pcl::FPFHSignature33> fpfh;
	fpfh.setSearchMethod(pcl::search::KdTree<TPoint>::Ptr(new pcl::search::KdTree<TPoint>));
	fpfh.setInputCloud(pKeypoints);
	fpfh.setInputNormals(pNormals);
	fpfh.setRadiusSearch(pSearchRadius);
	fpfh.setSearchSurface(pSrcCloud);
	fpfh.compute(*pDstCloud);

	boost::timer::cpu_times elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Computed " << pDstCloud->points.size() << " descriptors...\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";
#endif
}

void PointCloudOperations::compute_spinimage_descriptors
	(
		const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
		const pcl::PointCloud<TNormal>::ConstPtr & pNormals,
		const pcl::PointCloud<TPoint>::ConstPtr & pKeypoints,
		const float & pSearchRadius,
		pcl::PointCloud<SpinImage>::Ptr & pDstCloud
	)
{
#ifdef DEBUG
	std::cout << "Computing Spin Image descriptors...\n";
#endif

	boost::timer::cpu_timer t;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::copyPointCloud(*pSrcCloud, *cloud);
	pcl::copyPointCloud(*pKeypoints, *keypoints);

	std::cout << "Number of points in cloud: " << cloud->points.size() << "\n";
	std::cout << "Numer of points in normal: " << pNormals->points.size() << "\n";

	pcl::SpinImageEstimation<pcl::PointXYZ, TNormal, SpinImage> si;
	si.setInputCloud(keypoints);
	si.setInputNormals(pNormals);
	si.setRadiusSearch(pSearchRadius);
	si.setImageWidth(8);
	si.setSearchSurface(cloud);
	si.compute(*pDstCloud);

	boost::timer::cpu_times elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Computed " << pDstCloud->points.size() << " descriptors...\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";
#endif
}

void PointCloudOperations::compute_3dsc_descriptors
	(
		const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
		const pcl::PointCloud<TNormal>::ConstPtr & pNormals,
		const pcl::PointCloud<TPoint>::ConstPtr & pKeypoints,
		const float & pSearchRadius,
		pcl::PointCloud<pcl::ShapeContext1980>::Ptr & pDstCloud
	)
{
#ifdef DEBUG
	std::cout << "Computing 3DSC descriptors...\n";
#endif

	boost::timer::cpu_timer t;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::copyPointCloud(*pSrcCloud, *cloud);
	pcl::copyPointCloud(*pKeypoints, *keypoints);

	pcl::ShapeContext3DEstimation<pcl::PointXYZ, TNormal, pcl::ShapeContext1980> sc3d;
	sc3d.setInputCloud(keypoints);
	sc3d.setInputNormals(pNormals);
	sc3d.setRadiusSearch(pSearchRadius);
	sc3d.setMinimalRadius(pSearchRadius / 10.0);
	sc3d.setPointDensityRadius(pSearchRadius / 5.0);
	sc3d.setSearchSurface(cloud);
	sc3d.compute(*pDstCloud);

	boost::timer::cpu_times elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Computed " << pDstCloud->points.size() << " descriptors...\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";
#endif
}

/*void PointCloudOperations::compute_usc_descriptors
	(
		const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
		const pcl::PointCloud<TNormal>::ConstPtr & pNormals,
		const pcl::PointCloud<TPoint>::ConstPtr & pKeypoints,
		const float & pSearchRadius,
		pcl::PointCloud<pcl::UniqueShapeContext1960>::Ptr & pDstCloud
	)
{
#ifdef DEBUG
	std::cout << "Computing USC descriptors...\n";
#endif

	boost::timer t;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::copyPointCloud(*pSrcCloud, *cloud);
	pcl::copyPointCloud(*pKeypoints, *keypoints);

	pcl::UniqueShapeContext<pcl::PointXYZ, pcl::UniqueShapeContext1960, pcl::ReferenceFrame> usc;
	usc.setInputCloud(keypoints);
	usc.setRadiusSearch(pSearchRadius);
	usc.setMinimalRadius(pSearchRadius / 10.0);
	usc.setPointDensityRadius(pSearchRadius / 5.0);
	usc.setLocalRadius(pSearchRadius);
	usc.setSearchSurface(cloud);
	usc.compute(*pDstCloud);

	double elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Computed " << pDstCloud->points.size() << " descriptors...\n";
	std::cout << "Elapsed time: " << elapsedTime << " ms\n";
#endif
}*/

void PointCloudOperations::compute_rops_descriptors
	(
		const pcl::PointCloud<TPoint>::ConstPtr & pSrcCloud,
		const pcl::PointCloud<TNormal>::ConstPtr & pNormals,
		const pcl::PointCloud<TPoint>::ConstPtr & pKeypoints,
		const float & pSearchRadius,
		pcl::PointCloud<ROPS135>::Ptr & pDstCloud
	)
{
#ifdef DEBUG
	std::cout << "Computing RoPS descriptors...\n";
#endif

	//boost::timer t;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::copyPointCloud(*pSrcCloud, *cloud);
	pcl::copyPointCloud(*pKeypoints, *keypoints);

	pcl::PointCloud<pcl::PointNormal>::Ptr cloudWithNormals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields(*cloud, *pNormals, *cloudWithNormals);
	pcl::search::KdTree<pcl::PointNormal>::Ptr kdtreeNormals(new pcl::search::KdTree<pcl::PointNormal>);
	kdtreeNormals->setInputCloud(cloudWithNormals);

	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	pcl::PolygonMesh triangles;
	gp3.setSearchRadius(0.025f);
	gp3.setMu(2.5);
	gp3.setMaximumNearestNeighbors(100);
	gp3.setMaximumSurfaceAngle(M_PI / 4);
	gp3.setMinimumAngle(M_PI / 18);
	gp3.setMaximumAngle(2 * M_PI / 3);
	gp3.setNormalConsistency(false);

	gp3.setInputCloud(cloudWithNormals);
	gp3.setSearchMethod(kdtreeNormals);
	gp3.reconstruct(triangles);

	pcl::ROPSEstimation<pcl::PointXYZ, ROPS135> rops;
	rops.setSearchMethod(pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>));
	rops.setInputCloud(keypoints);
	rops.setSearchSurface(cloud);
	rops.setRadiusSearch(pSearchRadius);
	rops.setNumberOfPartitionBins(5);
	rops.setNumberOfRotations(3);
	rops.setSupportRadius(pSearchRadius);
	rops.setTriangles(triangles.polygons);
	rops.compute(*pDstCloud);

	//double elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Computed " << pDstCloud->points.size() << " descriptors...\n";
	//std::cout << "Elapsed time: " << elapsedTime << " ms\n";
#endif
}

/*void PointCloudOperations::find_correspondences_gpu
	(
		const pcl::PointCloud<TDescriptor>::ConstPtr & pModelDescriptors,
		const pcl::PointCloud<TDescriptor>::ConstPtr & pSceneDescriptors,
		const float & pThreshold,
		pcl::CorrespondencesPtr & pCorrespondences
	)
{
#ifdef DEBUG
	std::cout << "Finding correspondences...\n";
#endif

	boost::timer::cpu_timer t;

	int numInput = pModelDescriptors->points.size();
	int numQueries = pSceneDescriptors->points.size();
	int inputsize = 352;//TDescriptor::descriptorSize();

	flann::Matrix<float> data(new float[numInput * inputsize], numInput, inputsize);
	for (size_t i = 0; i < data.rows; ++i)
		for (size_t j = 0; j < data.cols; ++j)
			data[i][j] = pModelDescriptors->at(i).descriptor[j];

	//flann::Index<flann::L2<float> > index(data, flann::KDTreeCuda3dIndexParams(4));
	//flann::KDTreeCuda3dIndex< flann::L2<float> > index(data);
	flann::KDTreeCuda3dIndexParams params;
	params["dim"] = inputsize;
	flann::Index<flann::L2<float> > index(data, params);
	index.buildIndex();

	boost::timer::cpu_times elapsedTime = t.elapsed();
	cout << "Time for building kdtree-index (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	cout << "Time for building kdtree-index (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";

	t.start();

	flann::Matrix<float> queries(new float[numQueries * inputsize], numQueries, inputsize);
	for (size_t i = 0; i < queries.rows; ++i)
		for (size_t j = 0; j < queries.cols; ++j)
			queries[i][j] = pSceneDescriptors->at(i).descriptor[j];

	int kValue = 1;

	flann::Matrix<int> indices = flann::Matrix<int>(new int[numQueries * kValue], numQueries, kValue);
	flann::Matrix<float> distances = flann::Matrix<float>(new float[numQueries * kValue], numQueries, kValue);

	index.knnSearch(queries, indices, distances, kValue, flann::SearchParams(512));

	for (size_t i = 0; i < numQueries; ++i)
	{
		std::cout << "Query " << i << " distance: " << distances[i][0] << " index: " << indices[i][0] << " \n";
		if (distances[i][0] < pThreshold)
		{
			pcl::Correspondence correspondence(indices[i][0], static_cast<int>(i), distances[i][0]);
			pCorrespondences->push_back(correspondence);
		}
	}

	elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Found " << pCorrespondences->size() << " correspondences...\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";
#endif
}*/

void PointCloudOperations::find_correspondences_multicore
	(
		const pcl::PointCloud<TDescriptor>::ConstPtr & pModelDescriptors,
		const pcl::PointCloud<TDescriptor>::ConstPtr & pSceneDescriptors,
		const float & pThreshold,
		pcl::CorrespondencesPtr & pCorrespondences
	)
{
#ifdef DEBUG
	std::cout << "Finding correspondences...\n";
#endif

	boost::timer::cpu_timer t;

	int numInput = pModelDescriptors->points.size();
	int inputsize = 352;//TDescriptor::descriptorSize();

	flann::Matrix<float> data(new float[numInput * inputsize], numInput, inputsize);
	for (size_t i = 0; i < data.rows; ++i)
		for (size_t j = 0; j < data.cols; ++j)
			data[i][j] = pModelDescriptors->at(i).descriptor[j];

	flann::Index<flann::L2<float> > index(data, flann::KDTreeIndexParams(4));
	index.buildIndex();

	boost::timer::cpu_times elapsedTime = t.elapsed();
	cout << "Time for building kdtree-index (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	cout << "Time for building kdtree-index (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";

	t.start();

	double distance = 0.0;
	int neighbors = 0;
	int kValue = 1;

	#pragma omp parallel for firstprivate(index) num_threads(4)
	for (size_t i = 0; i < pSceneDescriptors->size(); ++i)
	{
		flann::Matrix<float> p = flann::Matrix<float>(new float[inputsize], 1, inputsize);
		memcpy(&p.ptr()[0], &pSceneDescriptors->at(i).descriptor[0], p.cols * p.rows * sizeof(float));

		flann::Matrix<int> indices;
		flann::Matrix<float> distances;
		indices = flann::Matrix<int>(new int[kValue], 1, kValue);
		distances = flann::Matrix<float>(new float[kValue], 1, kValue);

		int neighborsFound = 0;
		neighborsFound = index.knnSearch(p, indices, distances, kValue, flann::SearchParams(512));

		#pragma omp critical
		{
			if (neighborsFound == 1)
			{
				distance += distances[0][0];
				++neighbors;
			}

			if (neighborsFound == 1 && distances[0][0] < pThreshold)
			{
				pcl::Correspondence correspondence(indices[0][0], static_cast<int>(i), distances[0][0]);
				pCorrespondences->push_back(correspondence);
			}
		}
	}

	if (neighbors > 0)
		distance /= neighbors;
	else
		distance = std::numeric_limits<double>::infinity();

	elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Found " << pCorrespondences->size() << " correspondences...\n";
	//std::cout << "Average distance: " << distance << "\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";
#endif
}

void PointCloudOperations::find_correspondences
	(
		const pcl::PointCloud<TDescriptor>::ConstPtr & pModelDescriptors,
		const pcl::PointCloud<TDescriptor>::ConstPtr & pSceneDescriptors,
		const float & pThreshold,
		pcl::CorrespondencesPtr & pCorrespondences
	)
{
#ifdef DEBUG
	std::cout << "Finding correspondences...\n";
#endif

	boost::timer::cpu_timer t;

	pcl::KdTreeFLANN<TDescriptor> kdtree;
	kdtree.setInputCloud(pModelDescriptors);

	boost::timer::cpu_times elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Time elapsed for building kdtree (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	std::cout << "Time elapsed for building kdtree (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";
#endif

	t.start();

	double distance = 0.0;
	int neighbors = 0;

	for (size_t i = 0; i < pSceneDescriptors->size(); ++i)
	{
		std::vector<int> neighborsIndices(1);
		std::vector<float> neighborsSqrDistances(1);


		//if (!pcl_isfinite(pSceneDescriptors->at(i).descriptor[0]))
			//continue;

		int neighborsFound = kdtree.nearestKSearch(pSceneDescriptors->at(i), 1, neighborsIndices, neighborsSqrDistances);

		if (neighborsFound == 1)
		{
			distance += neighborsSqrDistances[0];
			++neighbors;
		}

		if (neighborsFound == 1 && neighborsSqrDistances[0] < pThreshold)
		{
			pcl::Correspondence correspondence(neighborsIndices[0], static_cast<int>(i), neighborsSqrDistances[0]);
			pCorrespondences->push_back(correspondence);
		}
	}

	if (neighbors > 0)
		distance /= neighbors;
	else
		distance = std::numeric_limits<double>::infinity();

	elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Found " << pCorrespondences->size() << " correspondences...\n";
	std::cout << "Average distance: " << distance << "\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";
#endif
}

void PointCloudOperations::cluster_correspondences
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
	)
{
	boost::timer::cpu_timer t;

	if (pUseHough3D)
	{
#ifdef DEBUG
		std::cout << "Computing reference frames...\n";
#endif

		pcl::PointCloud<TReferenceFrame>::Ptr modelReferenceFrames(new pcl::PointCloud<TReferenceFrame>());
		pcl::PointCloud<TReferenceFrame>::Ptr sceneReferenceFarmes(new pcl::PointCloud<TReferenceFrame>());

		// Compute reference frames
		pcl::BOARDLocalReferenceFrameEstimation<TPoint, TNormal, TReferenceFrame> rfe;
		rfe.setFindHoles(true);
		rfe.setRadiusSearch(pReferenceFrameRadius);

		rfe.setInputCloud(pModelKeypoints);
		rfe.setInputNormals(pModelNormals);
		rfe.setSearchSurface(pModel);
		rfe.compute(*modelReferenceFrames);

		rfe.setInputCloud(pSceneKeypoints);
		rfe.setInputNormals(pSceneNormals);
		rfe.setSearchSurface(pScene);
		rfe.compute(*sceneReferenceFarmes);

#ifdef DEBUG
		std::cout << "Computing Hough 3D Grouping...\n";
#endif
		pcl::Hough3DGrouping<TPoint, TPoint, TReferenceFrame, TReferenceFrame> h3dg;
		h3dg.setHoughBinSize(pClusterSize);
		h3dg.setHoughThreshold(pClusterThreshold);
		h3dg.setUseInterpolation(true);
		h3dg.setUseDistanceWeight(false);

		h3dg.setInputCloud(pModelKeypoints);
		h3dg.setInputRf(modelReferenceFrames);
		h3dg.setSceneCloud(pSceneKeypoints);
		h3dg.setSceneRf(sceneReferenceFarmes);
		h3dg.setModelSceneCorrespondences(pModelSceneCorrespondences);

		h3dg.recognize(pRotoTranslations, pClusteredCorrespondences);
	}
	else
	{
#ifdef DEBUG
		std::cout << "Computing Geometric Conssitency Grouping...\n";
#endif

		pcl::GeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ> gcg;
		pcl::PointCloud<pcl::PointXYZ>::Ptr modelKeypoints(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::copyPointCloud(*pModelKeypoints, *modelKeypoints);
		pcl::PointCloud<pcl::PointXYZ>::Ptr sceneKeypoints(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::copyPointCloud(*pSceneKeypoints, *sceneKeypoints);
		gcg.setGCSize(pClusterSize);
		gcg.setGCThreshold(pClusterThreshold);
		gcg.setInputCloud(modelKeypoints);
		gcg.setSceneCloud(sceneKeypoints);
		gcg.setModelSceneCorrespondences(pModelSceneCorrespondences);
		gcg.recognize(pRotoTranslations, pClusteredCorrespondences);
	}

	boost::timer::cpu_times elapsedTime = t.elapsed();

#ifdef DEBUG
	std::cout << "Model instances found: " << pRotoTranslations.size() << "...\n";
	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";
#endif
}

void PointCloudOperations::register_cloud
	(
		const pcl::PointCloud<TPoint>::ConstPtr & pModel,
		const pcl::PointCloud<TPoint>::ConstPtr & pScene,
		const int & pMaxIterations,
		const float & pMaxCorrespondenceDistance,
		pcl::PointCloud<TPoint>::Ptr & pDstCloud
	)
{
#ifdef DEBUG
	std::cout << "Registering point clouds with ICP...\n";
#endif

	boost::timer::cpu_timer t;

	pcl::IterativeClosestPoint<TPoint, TPoint> icp;
	icp.setMaximumIterations(pMaxIterations);
	icp.setMaxCorrespondenceDistance(pMaxCorrespondenceDistance);
	icp.setInputSource(pModel);
	icp.setInputTarget(pScene);
	icp.align(*pDstCloud);

	boost::timer::cpu_times elapsedTime = t.elapsed();

#ifdef DEBUG
	if (icp.hasConverged())
	{
		std::cout << "ICP has converged!\n";
		std::cout << "Transformation:\n";
		std::cout << icp.getFinalTransformation() << "\n";
	}
	else
		std::cout << "ICP has not converged...\n";

	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";
#endif
}


void PointCloudOperations::verify_hypotheses
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
	)
{
#ifdef DEBUG
	std::cout << "Verifying hypotheses with Global Hypotheseses Verification...\n";
#endif

	boost::timer::cpu_timer t;

	pcl::GlobalHypothesesVerification<TPoint, TPoint> ghv;
	ghv.setSceneCloud(pScene);
	ghv.addModels(pModels, true);

	ghv.setInlierThreshold(pInlierThreshold);
	ghv.setOcclusionThreshold(pOcclusionThreshold);
	ghv.setRegularizer(pRegularizer);
	ghv.setRadiusClutter(pRadiusClutter);
	ghv.setClutterRegularizer(pClutterRegularizer);
	ghv.setDetectClutter(pDetectClutter);
	ghv.setRadiusNormals(pRadiusNormals);

	ghv.verify();
	ghv.getMask(pHypothesesMask);

	boost::timer::cpu_times elapsedTime = t.elapsed();

#ifdef DEBUG
	for (int i = 0; i < pHypothesesMask.size(); ++i)
		if (pHypothesesMask[i])
			std::cout << "Instance " << i << " accepted...\n";
		else
			std::cout << "Instance " << i << " rejected...\n";

	std::cout << "Elapsed time (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	std::cout << "Elapsed time (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";
#endif
}
