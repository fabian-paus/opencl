#pragma once

#include <CL/cl.hpp>
#include <boost/gil/image.hpp>
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
