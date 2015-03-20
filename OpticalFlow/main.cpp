#include <CL/cl.hpp>

#include <boost/gil/image.hpp>
#include <boost/gil/extension/io/jpeg_io.hpp>

#include <iostream>
#include <chrono>
#include <vector>

namespace gil = boost::gil;

class Timer
{
public:
	Timer() : m_start() { }

	void start()
	{
		m_start = clock_t::now();
	}

	void stop(std::string const& event)
	{
		auto duration = clock_t::now() - m_start;
		auto durationInMs = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
		std::cout << "[Timer]: Event '" << event << "' took " << durationInMs.count() << " ms\n";
	}
	
private:
	typedef std::chrono::steady_clock clock_t;
	typedef clock_t::time_point time_point_t;
	time_point_t m_start;
};

cl::Platform choosePlatform()
{
	std::cout << "Querying platforms\n";
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	std::cout << "Platforms found: " << platforms.size() << "\n";

	for (std::size_t i = 0; i < platforms.size(); ++i)
	{
		auto& platform = platforms[i];
		std::cout << "Platform[" << i << "]:\n";
		std::cout << "  Profile: " << platform.getInfo<CL_PLATFORM_PROFILE>() << "\n";
		std::cout << "  Version: " << platform.getInfo<CL_PLATFORM_VERSION>() << "\n";
		std::cout << "  Name:    " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
		std::cout << "  Vendor:  " << platform.getInfo<CL_PLATFORM_VENDOR>() << "\n";
		std::cout << "  Ext.:    " << platform.getInfo<CL_PLATFORM_EXTENSIONS>() << "\n";
		std::cout << std::endl;
	}

	if (platforms.size() == 1)
		return platforms.front();
	
	std::cout << "Choose a platform: ";
	std::size_t platformIndex = 0;
	if (!(std::cin >> platformIndex) || platformIndex >= platforms.size())
		throw std::runtime_error("Not a valid index");
	return platforms[platformIndex];
}

cl::Device chooseDevice(cl::Platform const& platform, cl_device_type deviceType)
{
	std::cout << "Querying devices\n";
	std::vector<cl::Device> devices;
	platform.getDevices(deviceType, &devices);
	std::cout << "Devices found: " << devices.size() << "\n";

	for (std::size_t i = 0; i < devices.size(); ++i)
	{
		auto& device = devices[i];
		std::cout << "Device[" << i << "]:\n";
		
		std::cout << "  Name:      " << device.getInfo<CL_DEVICE_NAME>() << "\n";
		std::cout << "  Type:      " << device.getInfo<CL_DEVICE_TYPE>() << "\n";
		std::cout << "  Profile:   " << device.getInfo<CL_DEVICE_PROFILE>() << "\n";
		std::cout << "  Version:   " << device.getInfo<CL_DEVICE_VERSION>() << "\n";
		std::cout << "  Vendor:    " << device.getInfo<CL_DEVICE_VENDOR>() << "\n";
		std::cout << "  Max. CUs:  " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
		std::cout << "  Freq.:     " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "\n";
		std::cout << "  Image:     " << device.getInfo<CL_DEVICE_IMAGE_SUPPORT>() << "\n";
	}

	if (devices.size() == 1)
		return devices.front();

	std::cout << "Choose a device: ";
	std::size_t deviceIndex = 0;
	if (!(std::cin >> deviceIndex) || deviceIndex >= devices.size())
		throw std::runtime_error("Not a valid index");
	return devices[deviceIndex];
}

const std::string FIRST_IMAGE = "images/first.jpg";
const std::string SECOND_IMAGE = "images/second.jpg";

int main()
{
	Timer timer;

	timer.start();

	gil::gray8_image_t image;
	jpeg_read_image(FIRST_IMAGE, image);
	timer.stop("read_image");

	cl::ImageFormat format(CL_R, CL_UNSIGNED_INT8);

	auto platform = choosePlatform();
	auto device = chooseDevice(platform, CL_DEVICE_TYPE_CPU);

	cl::Context context(device);
	cl::CommandQueue(context, device);

	return 0;
}
