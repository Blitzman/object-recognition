#ifndef TYPES_H_
#define TYPES_H_

#include <pcl/point_cloud.h>
#include <pcl/features/rops_estimation.h>

typedef pcl::Histogram<153> SpinImage;
typedef pcl::Histogram<135> ROPS135;

typedef pcl::PointXYZRGB TPoint;
typedef pcl::Normal TNormal;
typedef pcl::SHOT352 TDescriptor;
//typedef pcl::SHOT1344 TDescriptor;
//typedef pcl::ShapeContext1980 TDescriptor;
//typedef pcl::UniqueShapeContext1960 TDescriptor;
//typedef pcl::FPFHSignature33 TDescriptor;
//typedef SpinImage TDescriptor;
//typedef ROPS135 TDescriptor;
typedef pcl::ReferenceFrame TReferenceFrame;



#endif
