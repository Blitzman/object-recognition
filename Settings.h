#ifndef SETTINGS_H_
#define SETTINGS_H_

#include <iostream>
#include <fstream>
#include <string>
// Boost headers
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

class Settings
{
	public:
		Settings(const std::string& pSettingsFile);
		~Settings();

		bool read_settings();

		template <class T>
		void add_option(
			const std::string& pOptionParameter,
			T& pParameter,
			const std::string& pHelpDescription)
		{
			mDescription.add_options()
				(pOptionParameter.c_str(), boost::program_options::value<T>(&pParameter), pHelpDescription.c_str());
		}


		template <class T>
		T read_option(
			const std::string& pOptionParameter);

	private:
		std::string mSettingsFile;
		boost::program_options::options_description mDescription;
		boost::program_options::variables_map mVariablesMap;
};

#endif
