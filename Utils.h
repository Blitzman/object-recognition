#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
#include <string>
// Boost headers
#include <boost/filesystem.hpp>

class Utils
{
	public:

		static void get_clouds_filenames
			(
				const std::string & pDirectoryPath,
				std::vector<std::string> & pFilenames
			);
};

#endif