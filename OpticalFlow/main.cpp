#include "runtime.hpp"

#include <CL/cl.hpp>

#include <boost/gil/image.hpp>

#include <iostream>
#include <fstream>

namespace gil = boost::gil;

const std::string FIRST_IMAGE = "images/first.jpg";
const std::string SECOND_IMAGE = "images/second.jpg";
const std::string PROGRAM_FILE = "optical-flow.cl";

void saveImage(cl::CommandQueue const& queue, cl::Image2D const& source, std::string targetFile, std::vector<cl::Event> const& waitEvents)
{
	TimedEvent event("save_image");
	auto mappedImage = mapImage(queue, source, CL_MAP_READ, &waitEvents);
	auto* mappedImageData = (gil::gray8_pixel_t*)mappedImage.data;
	auto width = source.getImageInfo<CL_IMAGE_WIDTH>();
	auto height = source.getImageInfo<CL_IMAGE_HEIGHT>();
	auto view = gil::interleaved_view(width, height, mappedImageData, mappedImage.rowSize);
	jpeg_write_view(targetFile, view);
	queue.enqueueUnmapMemObject(source, mappedImageData);
}

void writeProfileInfo(std::ostream& out, cl::Event const& event, std::string name, cl_ulong baseCounter)
{
	auto queued = event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() - baseCounter;
	auto submit = event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() - baseCounter;
	auto start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>() - baseCounter;
	auto end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - baseCounter;
	out << name << ";" << queued << ";" << submit - queued << ";" << start - submit << ";" << end - start << "\n";
}

int main()
{
	try
	{
		gil::gray8_image_t firstImage, secondImage;
		loadImage(FIRST_IMAGE, firstImage);
		loadImage(SECOND_IMAGE, secondImage);
		if (firstImage.dimensions() != secondImage.dimensions())
		{
			std::cout << "The images have different dimensions!\n";
			return -1;
		}

		auto platform = choosePlatform();
		auto device = chooseDevice(platform, CL_DEVICE_TYPE_CPU);

		cl::Context context(device);
		cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

		auto program = buildProgram(context, device, PROGRAM_FILE);
		cl::Kernel downFilterX(program, "downfilter_x");
		cl::Kernel downFilterY(program, "downfilter_y");
		cl::Kernel filterG(program, "filter_G");
		cl::Kernel testBuffer(program, "test_buffer");

		cl::ImageFormat format(CL_R, CL_UNSIGNED_INT8);
		std::size_t widthLevel0 = firstImage.width();
		std::size_t heightLevel0 = firstImage.height();

		Timer timer;
		timer.start();
		// Copy images
		cl::Image2D firstImageLevel0(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, format, widthLevel0, heightLevel0);
		auto firstImageCopyEvent = copyImage(queue, firstImage, firstImageLevel0);

		cl::Image2D secondImageLevel0(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, format, widthLevel0, heightLevel0);
		auto secondImageCopyEvent = copyImage(queue, secondImage, secondImageLevel0);

		// Level 0 -> 0 (Downfilter X)
		cl::Image2D firstImageLevel0_X(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, format, widthLevel0, heightLevel0);
		downFilterX.setArg(0, firstImageLevel0);
		downFilterX.setArg(1, firstImageLevel0_X);

		cl::NDRange rangeLevel0(widthLevel0, heightLevel0);
		std::vector<cl::Event> waitEvents{ firstImageCopyEvent };
		cl::Event downFilterX_firstLevel0;
		queue.enqueueNDRangeKernel(downFilterX, cl::NullRange, rangeLevel0, cl::NullRange, &waitEvents, &downFilterX_firstLevel0);

		cl::Image2D secondImageLevel0_X(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, format, widthLevel0, heightLevel0);
		downFilterX.setArg(0, secondImageLevel0);
		downFilterX.setArg(1, secondImageLevel0_X);

		waitEvents[0] = secondImageCopyEvent;
		cl::Event downFilterX_secondLevel0;
		queue.enqueueNDRangeKernel(downFilterX, cl::NullRange, rangeLevel0, cl::NullRange, &waitEvents, &downFilterX_secondLevel0);

		// Level 0 -> 1 (Downfilter Y)
		auto widthLevel1 = widthLevel0 / 2;
		auto heightLevel1 = heightLevel0 / 2;
		cl::NDRange rangeLevel1(widthLevel1, heightLevel1);

		cl::Image2D firstImageLevel1(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, format, widthLevel1, heightLevel1);
		downFilterY.setArg(0, firstImageLevel0_X);
		downFilterY.setArg(1, firstImageLevel1);

		waitEvents[0] = downFilterX_firstLevel0;
		cl::Event downFilterY_firstLevel0;
		queue.enqueueNDRangeKernel(downFilterY, cl::NullRange, rangeLevel1, cl::NullRange, &waitEvents, &downFilterY_firstLevel0);

		cl::Image2D secondImageLevel1(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, format, widthLevel1, heightLevel1);
		downFilterY.setArg(0, secondImageLevel0_X);
		downFilterY.setArg(1, secondImageLevel1);

		waitEvents[0] = downFilterX_secondLevel0;
		cl::Event downFilterY_secondLevel0;
		queue.enqueueNDRangeKernel(downFilterY, cl::NullRange, rangeLevel1, cl::NullRange, &waitEvents, &downFilterY_secondLevel0);

		// Level 1 -> 1 (Downfilter X)
		cl::Image2D firstImageLevel1_X(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, format, widthLevel1, heightLevel1);
		downFilterX.setArg(0, firstImageLevel1);
		downFilterX.setArg(1, firstImageLevel1_X);

		waitEvents[0] = downFilterY_firstLevel0;
		cl::Event downFilterX_firstLevel1;
		queue.enqueueNDRangeKernel(downFilterX, cl::NullRange, rangeLevel1, cl::NullRange, &waitEvents, &downFilterX_firstLevel1);

		cl::Image2D secondImageLevel1_X(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, format, widthLevel1, heightLevel1);
		downFilterX.setArg(0, secondImageLevel1);
		downFilterX.setArg(1, secondImageLevel1_X);

		waitEvents[0] = downFilterY_secondLevel0;
		cl::Event downFilterX_secondLevel1;
		queue.enqueueNDRangeKernel(downFilterX, cl::NullRange, rangeLevel1, cl::NullRange, &waitEvents, &downFilterX_secondLevel1);

		// Level 1 -> 2 (Downfilter Y)
		auto widthLevel2 = widthLevel0 / 4;
		auto heightLevel2 = heightLevel0 / 4;
		cl::NDRange rangeLevel2(widthLevel2, heightLevel2);

		cl::Image2D firstImageLevel2(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, format, widthLevel2, heightLevel2);
		downFilterY.setArg(0, firstImageLevel1_X);
		downFilterY.setArg(1, firstImageLevel2);

		waitEvents[0] = downFilterX_firstLevel1;
		cl::Event downFilterY_firstLevel1;
		queue.enqueueNDRangeKernel(downFilterY, cl::NullRange, rangeLevel2, cl::NullRange, &waitEvents, &downFilterY_firstLevel1);

		cl::Image2D secondImageLevel2(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, format, widthLevel2, heightLevel2);
		downFilterY.setArg(0, secondImageLevel1_X);
		downFilterY.setArg(1, secondImageLevel2);

		waitEvents[0] = downFilterX_secondLevel1;
		cl::Event downFilterY_secondLevel1;
		queue.enqueueNDRangeKernel(downFilterY, cl::NullRange, rangeLevel2, cl::NullRange, &waitEvents, &downFilterY_secondLevel1);

		std::vector<cl::Event> waitEvents2(2);

		// Level 0 G
		std::size_t sizeG0 = widthLevel0 * heightLevel0 * sizeof (cl_int4);
		cl::Buffer bufferG0(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, sizeG0);

		filterG.setArg(0, firstImageLevel0);
		filterG.setArg(1, secondImageLevel0);
		filterG.setArg(2, widthLevel0);
		filterG.setArg(3, bufferG0);
		waitEvents2[0] = firstImageCopyEvent;
		waitEvents2[1] = secondImageCopyEvent;
		cl::Event filterG0;
		queue.enqueueNDRangeKernel(filterG, cl::NullRange, rangeLevel0, cl::NullRange, &waitEvents2, &filterG0);

		// Level 1 G
		std::size_t sizeG1 = widthLevel1 * heightLevel1 * sizeof (cl_int4);
		cl::Buffer bufferG1(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, sizeG1);

		filterG.setArg(0, firstImageLevel1);
		filterG.setArg(1, secondImageLevel1);
		filterG.setArg(2, widthLevel1);
		filterG.setArg(3, bufferG1);
		waitEvents2[0] = downFilterY_firstLevel0;
		waitEvents2[1] = downFilterY_secondLevel0;
		cl::Event filterG1;
		queue.enqueueNDRangeKernel(filterG, cl::NullRange, rangeLevel1, cl::NullRange, &waitEvents2, &filterG1);

		// Level 2 G
		std::size_t sizeG2 = widthLevel2 * heightLevel2 * sizeof (cl_int4);
		cl::Buffer bufferG2(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, sizeG2);

		filterG.setArg(0, firstImageLevel2);
		filterG.setArg(1, secondImageLevel2);
		filterG.setArg(2, widthLevel2);
		filterG.setArg(3, bufferG2);
		waitEvents2[0] = downFilterY_firstLevel1;
		waitEvents2[1] = downFilterY_secondLevel1;
		cl::Event filterG2;
		queue.enqueueNDRangeKernel(filterG, cl::NullRange, rangeLevel2, cl::NullRange, &waitEvents2, &filterG2);

		queue.finish();
		timer.stop("down_filter_all");

		std::ofstream out("profile.csv");

		auto baseCounter = firstImageCopyEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();

		out << ";Not Existing;Queued;Submitted;Running\n";
		writeProfileInfo(out, firstImageCopyEvent, "Copy Image 1", baseCounter);
		writeProfileInfo(out, secondImageCopyEvent, "Copy Image 2", baseCounter);

		writeProfileInfo(out, downFilterX_firstLevel0, "DownFilterX Image 1 Level 0", baseCounter);
		writeProfileInfo(out, downFilterX_secondLevel0, "DownFilterX Image 2 Level 0", baseCounter);
		writeProfileInfo(out, downFilterY_firstLevel0, "DownFilterY Image 1 Level 0", baseCounter);
		writeProfileInfo(out, downFilterY_secondLevel0, "DownFilterY Image 2 Level 0", baseCounter);

		writeProfileInfo(out, downFilterX_firstLevel1, "DownFilterX Image 1 Level 1", baseCounter);
		writeProfileInfo(out, downFilterX_secondLevel1, "DownFilterX Image 2 Level 1", baseCounter);
		writeProfileInfo(out, downFilterY_firstLevel1, "DownFilterY Image 1 Level 1", baseCounter);
		writeProfileInfo(out, downFilterY_secondLevel1, "DownFilterY Image 2 Level 1", baseCounter);

		writeProfileInfo(out, filterG0, "FilterG Level 0", baseCounter);
		writeProfileInfo(out, filterG1, "FilterG Level 1", baseCounter);
		writeProfileInfo(out, filterG2, "FilterG Level 2", baseCounter);

		auto maxCounter = downFilterY_secondLevel1.getProfilingInfo<CL_PROFILING_COMMAND_END>() - baseCounter;
		std::cout << "Max counter: " << maxCounter << std::endl;

		return 0;
	}
	catch (std::exception const& ex)
	{
		std::cout << ex.what() << std::endl;
		return -1;
	}
}
