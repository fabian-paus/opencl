#pragma once

#define __CL_ENABLE_EXCEPTIONS
#ifdef _WIN32
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#else
#include <CL/cl.h>
#undef CL_VERSION_1_2
#undef CL_VERSION_2_0
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#undef CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED
#undef CL_EXT_PREFIX__VERSION_1_1_DEPRECATED
#define CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED
#define CL_EXT_PREFIX__VERSION_1_1_DEPRECATED
#endif
#include <CL/cl.hpp>

#include <boost/gil/image.hpp>
#include <boost/gil/gray.hpp>
#include <boost/gil/typedefs.hpp>
#include <string>
#include <chrono>

class Timer
{
public:
	Timer();

	void start();

	void stop(std::string const& event);
	
private:
	typedef std::chrono::steady_clock clock_t;
	typedef clock_t::time_point time_point_t;
	time_point_t m_start;
};

class TimedEvent
{
public:
	TimedEvent(std::string const& event);

	~TimedEvent();

private:
	Timer m_timer;
	std::string m_event;
};

cl::Platform choosePlatform();

cl::Device chooseDevice(cl::Platform const& platform, cl_device_type deviceType);

struct MappedImage
{
	void* data;
	std::size_t rowSize;
};

MappedImage mapImage(cl::CommandQueue const& queue, cl::Image2D const& image, cl_map_flags flags, std::vector<cl::Event> const* waitEvents = nullptr);

void loadImage(std::string const& filename, boost::gil::gray8_image_t& image);

cl::Event copyImage(cl::CommandQueue const& queue, boost::gil::gray8_image_t const& source, cl::Image2D const& target);

cl::Program buildProgram(cl::Context const& context, cl::Device const& device, std::string const& programFile);
