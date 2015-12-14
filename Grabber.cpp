#include "Grabber.h"

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Grabber::convertToXYZRGBPointCloud (
			const boost::shared_ptr<pcl::io::Image> &image,
			const boost::shared_ptr<pcl::io::DepthImage> &depth_image,
			std::string rgb_frame_id, float constant_)
{
	  static unsigned rgb_array_size = 0;
	  static boost::shared_array<unsigned char> rgb_array;
	  static unsigned char* rgb_buffer = 0;

	int image_width_=image->getWidth();
	int image_height_=image->getHeight();
	int depth_width_=depth_image->getWidth();
	int depth_height_=depth_image->getHeight();

	  boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> > cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

	  cloud->header.frame_id = rgb_frame_id;
	  cloud->height = std::max (image_height_, depth_height_);
	  cloud->width = std::max (image_width_, depth_width_);
	  cloud->is_dense = false;

	  cloud->points.resize (cloud->height * cloud->width);

	  float constant = constant_;//1.0f / -1.f; //device->getImageFocalLength (depth_width_);
	  register int centerX = (depth_width_ >> 1);
	  int centerY = (depth_height_ >> 1);

	  register const XnDepthPixel* depth_map = depth_image->getData();
	  if (depth_image->getWidth () != depth_width_ || depth_image->getHeight() != depth_height_)
	  {
		static unsigned buffer_size = 0;
		static boost::shared_array<unsigned short> depth_buffer;

		if (buffer_size < depth_width_ * depth_height_)
		{
		  buffer_size = depth_width_ * depth_height_;
		  depth_buffer.reset (new unsigned short [buffer_size]);
		}

		depth_image->fillDepthImageRaw (depth_width_, depth_height_, depth_buffer.get ());
		depth_map = depth_buffer.get ();
	  }

	  // here we need exact the size of the point cloud for a one-one correspondence!
	  if (rgb_array_size < image_width_ * image_height_ * 3)
	  {
		rgb_array_size = image_width_ * image_height_ * 3;
		rgb_array.reset (new unsigned char [rgb_array_size]);
		rgb_buffer = rgb_array.get ();
	  }
	  image->fillRGB (image_width_, image_height_, rgb_buffer, image_width_ * 3);
	  float bad_point = std::numeric_limits<float>::quiet_NaN ();

	  // set xyz to Nan and rgb to 0 (black)  
	  if (image_width_ != depth_width_)
	  {
		pcl::PointXYZRGB pt;
		pt.x = pt.y = pt.z = bad_point;
		pt.b = pt.g = pt.r = 0;
		pt.a = 255; // point has no color info -> alpha = max => transparent 
		cloud->points.assign (cloud->points.size (), pt);
	  }
  
	  // fill in XYZ values
	  unsigned step = cloud->width / depth_width_;
	  unsigned skip = cloud->width * step - cloud->width;
  
	  int value_idx = 0;
	  int point_idx = 0;
	  for (int v = -centerY; v < centerY; ++v, point_idx += skip)
	  {
			for (register int u = -centerX; u < centerX; ++u, ++value_idx, point_idx += step)
			{
				pcl::PointXYZRGB& pt = cloud->points[point_idx];
				/// @todo Different values for these cases
				// Check for invalid measurements

				if (depth_map[value_idx] != 0 &&
					depth_map[value_idx] != depth_image->getNoSampleValue () &&
					depth_map[value_idx] != depth_image->getShadowValue ())
				{
					pt.z = depth_map[value_idx] * 0.001f;
					pt.x = static_cast<float> (u) * pt.z * constant;
					pt.y = static_cast<float> (v) * pt.z * constant;
				}
				else
				{
					pt.x = pt.y = pt.z = bad_point;
				}
			}
	  }

	  // fill in the RGB values
	  step = cloud->width / image_width_;
	  skip = cloud->width * step - cloud->width;
  
	  value_idx = 0;
	  point_idx = 0;
	  pcl::RGBValue color;
	  color.Alpha = 0;

	  for (unsigned yIdx = 0; yIdx < image_height_; ++yIdx, point_idx += skip)
	  {
		for (unsigned xIdx = 0; xIdx < image_width_; ++xIdx, point_idx += step, value_idx += 3)
		{
		  pcl::PointXYZRGB& pt = cloud->points[point_idx];
      
		  color.Red   = rgb_buffer[value_idx];
		  color.Green = rgb_buffer[value_idx + 1];
		  color.Blue  = rgb_buffer[value_idx + 2];
      
		  pt.rgba = color.long_value;
		}
	  }

	  /*cloud->sensor_origin_.setZero ();
	  cloud->sensor_orientation_.w () = 0.0;
	  cloud->sensor_orientation_.x () = 1.0;
	  cloud->sensor_orientation_.y () = 0.0;
	  cloud->sensor_orientation_.z () = 0.0;*/

	return (cloud);
}

void Grabber::images_cb_(const boost::shared_ptr<pcl::io::Image>& rgbImage,
			const boost::shared_ptr<pcl::io::DepthImage>& depthImage,
			float constant)
{
	boost::timer::cpu_timer t;

	std::cout << "Callback\n";
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	t.start();

	cloud = convertToXYZRGBPointCloud(rgbImage, depthImage, "pene", constant);

	boost::timer::cpu_times elapsedTime = t.elapsed();
	cout << "Time for creating point cloud (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	cout << "Time for creating point cloud (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";

	viewer.showCloud(cloud);

	/*if (!viewer.wasStopped())
		viewer.showCloud(cloud);*/

	/*std::vector<unsigned short> source_depth_data_;
	pcl::gpu::PtrStepSz<const unsigned short> depth_;

	depth_.cols = depthImage->getWidth();
	depth_.rows = depthImage->getHeight();
	depth_.step = depth_.cols * depth_.elemSize();
	source_depth_data_.resize(depth_.cols * depth_.rows);
	depthImage->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
	depth_.data = &source_depth_data_[0];

	DepthMap depth;
	DepthMap depth_filtered;
	MapArr map;
	MapArr nmap;

	depth.upload(depth_.data, depth_.step, depth_.rows, depth_.cols);

	t.start();

	pcl::device::bilateralFilter(depth, depth_filtered);
	pcl::device::sync();

	elapsedTime = t.elapsed();
	cout << "Time for bilateral filter with CUDA (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	cout << "Time for bilateral filter with CUDA (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";

	t.start();

	pcl::device::createVMap(constant, constant, depth, map);
	pcl::device::sync();

	elapsedTime = t.elapsed();
	cout << "Time for creating point cloud with CUDA (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	cout << "Time for creating point cloud with CUDA (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";

	t.start();

	double cloud_resolution = pcl::device::computeCloudResolution(map, depth_.rows, depth_.cols);
	pcl::device::sync();

	elapsedTime = t.elapsed();
	cout << "Cloud resolution is " << cloud_resolution << "\n";
	cout << "Time for computing cloud resolution with CUDA (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	cout << "Time for computing cloud resolution with CUDA (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";

	t.start();
	pcl::device::computeNormalsEigen(map, nmap);
	pcl::device::sync();

	elapsedTime = t.elapsed();
	cout << "Time for computing normals with CUDA (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	cout << "Time for computing normals with CUDA (CPU): " << (elapsedTime.user * 1E-9) << " seconds\n";

	/*boost::timer::cpu_times accTime;
	accTime.clear();
	for (int i = 0; i < 100; ++i)
	{
		t.start();

		pcl::device::bilateralFilter(depth, depth_filtered);
		pcl::device::sync();

		boost::timer::cpu_times elapsedTime = t.elapsed();

		accTime.wall += elapsedTime.wall;
		accTime.user += elapsedTime.user;

		cout << "[" << i << "] Time elapsed for bilateral filter with CUDA (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	}

	cout << "Mean time for bilateral filter with CUDA (Wall): " << (accTime.wall * 1E-9 / 100.0) << " seconds\n";
	cout << "Mean Time for bilateral filter with CUDA (CPU): " << (accTime.user * 1E-9 / 100.0) << " seconds\n";

	accTime.clear();
	for (int i = 0; i < 100; ++i)
	{
		t.start();

		pcl::device::createVMap(constant, constant, depth, map);
		pcl::device::sync();

		boost::timer::cpu_times elapsedTime = t.elapsed();

		accTime.wall += elapsedTime.wall;
		accTime.user += elapsedTime.user;

		cout << "[" << i << "] Time elapsed for creating point cloud with CUDA (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	}

	cout << "Mean time for creating point cloud with CUDA (Wall): " << (accTime.wall * 1E-9 / 100.0) << " seconds\n";
	cout << "Mean time for creating point cloud with CUDA (CPU): " << (accTime.user * 1E-9 / 100.0) << " seconds\n";

	accTime.clear();
	double cloud_resolution = 0.0;
	for (int i = 0; i < 100; ++i)
	{
		t.start();

		cloud_resolution = pcl::device::computeCloudResolution(map, depth_.rows, depth_.cols);
		pcl::device::sync();

		boost::timer::cpu_times elapsedTime = t.elapsed();

		accTime.wall += elapsedTime.wall;
		accTime.user += elapsedTime.user;

		cout << "[" << i << "] Time elapsed for computing cloud resolution with CUDA (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	}

	cout << "Cloud resolution is " << cloud_resolution << "\n";
	cout << "Mean time for computing cloud resolution with CUDA (Wall): " << (accTime.wall * 1E-9 / 100.0) << " seconds\n";
	cout << "Mean time for computing cloud resolution with CUDA (CPU): " << (accTime.user * 1E-9 / 100.0) << " seconds\n";

	accTime.clear();
	for (int i = 0; i < 100; ++i)
	{
		t.start();

		pcl::device::computeNormalsEigen(map, nmap);
		pcl::device::sync();

		boost::timer::cpu_times elapsedTime = t.elapsed();

		accTime.wall += elapsedTime.wall;
		accTime.user += elapsedTime.user;

		cout << "[" << i << "] Time elapsed for estimating normals with CUDA (Wall): " << (elapsedTime.wall * 1E-9) << " seconds\n";
	}

	cout << "Mean time for computing normals with CUDA (Wall): " << (accTime.wall * 1E-9 / 100.0) << " seconds\n";
	cout << "Mean time for computing normals with CUDA (CPU): " << (accTime.user * 1E-9 / 100.0) << " seconds\n";*/

	//recognizer->recognize(cloud);
}

void Grabber::keyboard_cb (const pcl::visualization::KeyboardEvent & event, void *)
{
	if (event.getKeyCode())
	{
		if (!event.keyDown())
		{
			cout << "The key " << event.getKeyCode() << "key was pressed\n";
			recognizer->recognize(cloud);
		}
	}
}

void Grabber::run ()
{
	viewer.registerKeyboardCallback(&Grabber::keyboard_cb, *this);

	pcl::Grabber* interface = new pcl::io::OpenNI2Grabber();

	boost::function<void (const boost::shared_ptr<pcl::io::Image> &, const boost::shared_ptr<pcl::io::DepthImage> &, float constant)> f =
		boost::bind(&Grabber::images_cb_, this, _1, _2, _3);

       interface->registerCallback (f);

       interface->start ();

	while(1){}

       interface->stop ();
}
