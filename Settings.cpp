#include "Settings.h"

Settings::Settings(const std::string& pSettingsFile)
{
	mSettingsFile = pSettingsFile;
}

Settings::~Settings()
{

}

bool Settings::read_settings()
{

	std::ifstream settingsFile(mSettingsFile.c_str());

	if (!settingsFile.is_open())
	{
		std::cerr << "Settings file not found: " << mSettingsFile << "\n";
		return false;
	}

	mVariablesMap = boost::program_options::variables_map();
	boost::program_options::store(boost::program_options::parse_config_file(settingsFile, mDescription), mVariablesMap);
	boost::program_options::notify(mVariablesMap);

	return true;
}

template <class T>
T Settings::read_option(const std::string& pOptionParameter)
{
	return mVariablesMap[pOptionParameter].as<T>();
}
