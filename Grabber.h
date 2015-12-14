#ifndef GRABBER_H_
#define GRABBER_H_

#include <boost/timer/timer.hpp>

#include <pcl/io/openni2_grabber.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/openni_camera/openni_image.h>
#include <pcl/io/openni_camera/openni_depth_image.h>
#include <pcl/io/oni_grabber.h>
#include <pcl/io/pcd_grabber.h>
#include <pcl/io/image.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/boost.h>

#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/initialization.h>
#include <pcl/gpu/containers/device_memory.h>

#include "PointCloudOperationsCUDA.h"
#include "Recognizer.h"

typedef pcl::gpu::DeviceArray2D<unsigned short> DepthMap;
typedef pcl::gpu::DeviceArray2D<float> MapArr;

namespace pcl
{
	typedef union
	{
		struct
		{
			unsigned char Blue;
			unsigned char Green;
			unsigned char Red;
			unsigned char Alpha;
		};
		float float_value;
		uint32_t long_value;
	}RGBValue;
}

 class Grabber
 {
 	public:
		Grabber(Recognizer * rec) : viewer ("PCL OpenNI Viewer") 
		{
			recognizer = rec;
			cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
		}

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertToXYZRGBPointCloud (
			const boost::shared_ptr<pcl::io::Image> &image,
			const boost::shared_ptr<pcl::io::DepthImage> &depth_image,
			std::string rgb_frame_id, float constant_);


		void images_cb_(const boost::shared_ptr<pcl::io::Image>& rgbImage,
			const boost::shared_ptr<pcl::io::DepthImage>& depthImage,
			float constant);

		void keyboard_cb (const pcl::visualization::KeyboardEvent & event, void *);

     		void run ();

     		pcl::visualization::CloudViewer viewer;

	private:
		Recognizer *recognizer;
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
 };

#endif
