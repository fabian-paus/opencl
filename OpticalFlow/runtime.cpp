#include "runtime.hpp"

#include <boost/gil/extension/io/jpeg_io.hpp>
#include <vector>
#include <iostream>

Timer::Timer() 
	: m_start() 
{ }

void Timer::start()
{
	m_start = clock_t::now();
}

void Timer::stop(std::string const& event)
{
	auto duration = clock_t::now() - m_start;
	auto durationInMs = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
	std::cout << "[Timer]: Event '" << event << "' took " << durationInMs.count() << " ms\n";
}

TimedEvent::TimedEvent(std::string const& event)
	: m_event(event)
{
	m_timer.start();
}

TimedEvent::~TimedEvent()
{
	m_timer.stop(m_event);
}

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
		std::cout << "  Queue:     " << device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>() << "\n";
	}

	if (devices.size() == 1)
		return devices.front();

	std::cout << "Choose a device: ";
	std::size_t deviceIndex = 0;
	if (!(std::cin >> deviceIndex) || deviceIndex >= devices.size())
		throw std::runtime_error("Not a valid index");
	return devices[deviceIndex];
}

MappedImage mapImage(cl::CommandQueue const& queue, cl::Image2D const& image, cl_map_flags flags, std::vector<cl::Event> const* waitEvents)
{
	cl::size_t<3> origin;
	cl::size_t<3> region;
	region[0] = image.getImageInfo<CL_IMAGE_WIDTH>();
	region[1] = image.getImageInfo<CL_IMAGE_HEIGHT>();
	region[2] = 1;

	MappedImage result;
	result.data = queue.enqueueMapImage(image, CL_TRUE, flags, origin, region, &result.rowSize, nullptr, waitEvents);
	return result;
}

void loadImage(std::string const& filename, boost::gil::gray8_image_t& image)
{
	TimedEvent timer("read_image");
	jpeg_read_image(filename, image);
}

cl::Event copyImage(cl::CommandQueue const& queue, boost::gil::gray8_image_t const& source, cl::Image2D const& target)
{
	TimedEvent timer("copy_image");
	auto mappedTargetImage = mapImage(queue, target, CL_MAP_WRITE);
	auto* mappedData = (boost::gil::gray8_pixel_t*)mappedTargetImage.data;
	auto memoryView = boost::gil::interleaved_view(source.width(), source.height(), mappedData, mappedTargetImage.rowSize);
	copy_pixels(boost::gil::const_view(source), memoryView);

	cl::Event event;
	queue.enqueueUnmapMemObject(target, mappedData, nullptr, &event);
	return event;
}

std::string readFileToString(std::string const& filename)
{
	std::ifstream inputFile;
	inputFile.exceptions(std::ios_base::badbit | std::ios_base::failbit);
	inputFile.open(filename);

	std::istreambuf_iterator<char> begin(inputFile), end;
	return std::string(begin, end);
}

cl::Program buildProgram(cl::Context const& context, cl::Device const& device, std::string const& programFile)
{
	TimedEvent timer("build_program");

	auto programSource = readFileToString(programFile);
	cl::Program program(context, programSource);

	try
	{
		program.build({ device });
		return program;
	}
	catch (cl::Error const& error)
	{
		std::cout << "Build error (" << error.err() << "): " << error.what() << std::endl;
		auto buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
		std::cout << buildLog << std::endl;

		throw;
	}
}