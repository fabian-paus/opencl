#include "runtime.hpp"

#include <CL/cl.hpp>

#include <boost/gil/image.hpp>

#include <iostream>
#include <fstream>
#include <array>

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

#ifdef NDEBUG
const cl_mem_flags INTERMEDIATE_MEMORY_FLAGS = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;
const cl_mem_flags INPUT_MEMORY_FLAGS = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_HOST_WRITE_ONLY;
#else
// Allow host access to the images in debug mode
const cl_mem_flags INTERMEDIATE_MEMORY_FLAGS = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
const cl_mem_flags INPUT_MEMORY_FLAGS = CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY;
#endif

const std::size_t PYRAMID_HEIGHT = 3;
const cl::ImageFormat IMAGE_FORMAT(CL_R, CL_UNSIGNED_INT8);

class ImagePyramid
{
public:
	ImagePyramid(gil::gray8_image_t const& image, cl::Context const& context, cl::CommandQueue const& queue,
		cl::Kernel& downFilterX, cl::Kernel& downFilterY)
	{
		for (std::size_t i = 0; i < PYRAMID_HEIGHT; ++i)
		{
			// Half the dimensions with every level
			auto width = image.width() >> i;
			auto height = image.height() >> i;
			m_dimensions[i] = cl::NDRange(width, height);

			cl_mem_flags memoryFlags = (i == 0) ? INPUT_MEMORY_FLAGS : INTERMEDIATE_MEMORY_FLAGS;
			m_images[i] = cl::Image2D(context, memoryFlags, IMAGE_FORMAT, width, height);
		}

		m_finished[0] = copyImage(queue, image, m_images[0]);

		// Level 0 -> 1 Downfiltering
		m_intermediate_0_1 = cl::Image2D(context, INTERMEDIATE_MEMORY_FLAGS, IMAGE_FORMAT, m_dimensions[0][0], m_dimensions[0][1]);
		downFilterX.setArg(0, m_images[0]);
		downFilterX.setArg(1, m_intermediate_0_1);

		std::vector<cl::Event> waitEvents{ m_finished[0] };
		cl::Event downFilterX_Level0;
		queue.enqueueNDRangeKernel(downFilterX, cl::NullRange, m_dimensions[0], cl::NullRange, &waitEvents, &downFilterX_Level0);

		downFilterY.setArg(0, m_intermediate_0_1);
		downFilterY.setArg(1, m_images[1]);

		waitEvents[0] = downFilterX_Level0;
		queue.enqueueNDRangeKernel(downFilterY, cl::NullRange, m_dimensions[1], cl::NullRange, &waitEvents, &m_finished[1]);

		// Level 1 -> 2 Downfiltering
		m_intermediate_1_2 = cl::Image2D(context, INTERMEDIATE_MEMORY_FLAGS, IMAGE_FORMAT, m_dimensions[1][0], m_dimensions[1][1]);
		downFilterX.setArg(0, m_images[1]);
		downFilterX.setArg(1, m_intermediate_1_2);

		waitEvents[0] = m_finished[1];
		cl::Event downFilterX_Level1;
		queue.enqueueNDRangeKernel(downFilterX, cl::NullRange, m_dimensions[1], cl::NullRange, &waitEvents, &downFilterX_Level1);

		downFilterY.setArg(0, m_intermediate_1_2);
		downFilterY.setArg(1, m_images[2]);

		waitEvents[0] = downFilterX_Level1;
		queue.enqueueNDRangeKernel(downFilterY, cl::NullRange, m_dimensions[2], cl::NullRange, &waitEvents, &m_finished[2]);
	}

	cl::Image2D const& getImage(std::size_t level) const { return m_images[level]; }

	cl::NDRange const& getDimenstion(std::size_t level) const { return m_dimensions[level]; }

	cl::Event const& getFinished(std::size_t level) const { return m_finished[level]; }

private:
	std::array<cl::Image2D, PYRAMID_HEIGHT> m_images;
	std::array<cl::NDRange, PYRAMID_HEIGHT> m_dimensions;
	std::array<cl::Event, PYRAMID_HEIGHT> m_finished;

	cl::Image2D m_intermediate_0_1;
	cl::Image2D m_intermediate_1_2;
};

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
		cl::Kernel scharrHorX(program, "scharr_x_horizontal");
		cl::Kernel scharrVerX(program, "scharr_x_vertical");
		cl::Kernel scharrHorY(program, "scharr_y_horizontal");
		cl::Kernel scharrVerY(program, "scharr_y_vertical");

		cl::ImageFormat format(CL_R, CL_UNSIGNED_INT8);
		std::size_t widthLevel0 = firstImage.width();
		std::size_t heightLevel0 = firstImage.height();

		Timer timer;
		timer.start();

		// Copy images to level 0
		ImagePyramid firstImagePyramid(firstImage, context, queue, downFilterX, downFilterY);
		ImagePyramid secondImagePyramid(secondImage, context, queue, downFilterX, downFilterY);

		queue.finish();

		std::vector<cl::Event> waitEvents2(2);

		// Level 0 G
		cl::ImageFormat formatG(CL_RGBA, CL_SIGNED_INT32);
		cl::Image2D imageG0(context, INTERMEDIATE_MEMORY_FLAGS, formatG, widthLevel0, heightLevel0);

		filterG.setArg(0, firstImagePyramid.getImage(0));
		filterG.setArg(1, secondImagePyramid.getImage(0));
		filterG.setArg(2, imageG0);
		waitEvents2[0] = firstImagePyramid.getFinished(0);
		waitEvents2[1] = secondImagePyramid.getFinished(0);;
		cl::Event filterG0;
		queue.enqueueNDRangeKernel(filterG, cl::NullRange, firstImagePyramid.getDimenstion(0), cl::NullRange, &waitEvents2, &filterG0);

		// Level 1 G
		auto widthLevel1 = firstImagePyramid.getDimenstion(1)[0];
		auto heightLevel1 = firstImagePyramid.getDimenstion(1)[1];
		cl::Image2D imageG1(context, INTERMEDIATE_MEMORY_FLAGS, formatG, widthLevel1, heightLevel1);

		filterG.setArg(0, firstImagePyramid.getImage(1));
		filterG.setArg(1, secondImagePyramid.getImage(1));
		filterG.setArg(2, imageG1);
		waitEvents2[0] = firstImagePyramid.getFinished(1);
		waitEvents2[1] = secondImagePyramid.getFinished(1);
		cl::Event filterG1;
		queue.enqueueNDRangeKernel(filterG, cl::NullRange, firstImagePyramid.getDimenstion(1), cl::NullRange, &waitEvents2, &filterG1);

		// Level 2 G
		auto widthLevel2 = firstImagePyramid.getDimenstion(2)[0];
		auto heightLevel2 = firstImagePyramid.getDimenstion(2)[1];
		cl::Image2D imageG2(context, INTERMEDIATE_MEMORY_FLAGS, formatG, widthLevel2, heightLevel2);

		filterG.setArg(0, firstImagePyramid.getImage(2));
		filterG.setArg(1, secondImagePyramid.getImage(2));
		filterG.setArg(2, imageG2);
		waitEvents2[0] = firstImagePyramid.getFinished(2);
		waitEvents2[1] = secondImagePyramid.getFinished(2);
		cl::Event filterG2;
		queue.enqueueNDRangeKernel(filterG, cl::NullRange, firstImagePyramid.getDimenstion(2), cl::NullRange, &waitEvents2, &filterG2);

		cl::ImageFormat formatScharr(CL_R, CL_SIGNED_INT16);

		// Level 0 Scharr X Horizontal
		cl::Image2D imageScharrHorX0(context, INTERMEDIATE_MEMORY_FLAGS, formatScharr, widthLevel0, heightLevel0);

		std::vector<cl::Event> waitEvents(1);

		scharrHorX.setArg(0, firstImagePyramid.getImage(0));
		scharrHorX.setArg(1, imageScharrHorX0);
		waitEvents[0] = firstImagePyramid.getFinished(0);
		cl::Event scharrHorX0;
		queue.enqueueNDRangeKernel(scharrHorX, cl::NullRange, firstImagePyramid.getDimenstion(0), cl::NullRange, &waitEvents, &scharrHorX0);

		// Level 0 Scharr X Vertical
		cl::Image2D imageScharrVerX0(context, INTERMEDIATE_MEMORY_FLAGS, formatScharr, widthLevel0, heightLevel0);

		scharrVerX.setArg(0, imageScharrHorX0);
		scharrVerX.setArg(1, imageScharrVerX0);
		waitEvents[0] = scharrHorX0;
		cl::Event scharrVerX0;
		queue.enqueueNDRangeKernel(scharrVerX, cl::NullRange, firstImagePyramid.getDimenstion(0), cl::NullRange, &waitEvents, &scharrVerX0);

		// Level 0 Scharr Y Horizontal
		cl::Image2D imageScharrHorY0(context, INTERMEDIATE_MEMORY_FLAGS, formatScharr, widthLevel0, heightLevel0);

		scharrHorY.setArg(0, firstImagePyramid.getImage(0));
		scharrHorY.setArg(1, imageScharrHorY0);
		waitEvents[0] = firstImagePyramid.getFinished(0);
		cl::Event scharrHorY0;
		queue.enqueueNDRangeKernel(scharrHorY, cl::NullRange, firstImagePyramid.getDimenstion(0), cl::NullRange, &waitEvents, &scharrHorY0);

		// Level 0 Scharr Y Vertical
		cl::Image2D imageScharrVerY0(context, INTERMEDIATE_MEMORY_FLAGS, formatScharr, widthLevel0, heightLevel0);

		scharrVerY.setArg(0, imageScharrHorY0);
		scharrVerY.setArg(1, imageScharrVerY0);
		waitEvents[0] = scharrHorX0;
		cl::Event scharrVerY0;
		queue.enqueueNDRangeKernel(scharrVerY, cl::NullRange, firstImagePyramid.getDimenstion(0), cl::NullRange, &waitEvents, &scharrVerY0);

		queue.finish();
		timer.stop("down_filter_all");

		std::ofstream out("profile.csv");

		auto baseCounter = firstImagePyramid.getFinished(0).getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();

		out << ";Not Existing;Queued;Submitted;Running\n";
		writeProfileInfo(out, firstImagePyramid.getFinished(0), "Copy Image 1", baseCounter);
		writeProfileInfo(out, secondImagePyramid.getFinished(0), "Copy Image 2", baseCounter);

		//writeProfileInfo(out, downFilterX_firstLevel0, "DownFilterX Image 1 Level 0", baseCounter);
		//writeProfileInfo(out, downFilterX_secondLevel0, "DownFilterX Image 2 Level 0", baseCounter);
		//writeProfileInfo(out, downFilterY_firstLevel0, "DownFilterY Image 1 Level 0", baseCounter);
		//writeProfileInfo(out, downFilterY_secondLevel0, "DownFilterY Image 2 Level 0", baseCounter);

		//writeProfileInfo(out, downFilterX_firstLevel1, "DownFilterX Image 1 Level 1", baseCounter);
		//writeProfileInfo(out, downFilterX_secondLevel1, "DownFilterX Image 2 Level 1", baseCounter);
		//writeProfileInfo(out, downFilterY_firstLevel1, "DownFilterY Image 1 Level 1", baseCounter);
		//writeProfileInfo(out, downFilterY_secondLevel1, "DownFilterY Image 2 Level 1", baseCounter);

		writeProfileInfo(out, filterG0, "FilterG Level 0", baseCounter);
		writeProfileInfo(out, filterG1, "FilterG Level 1", baseCounter);
		writeProfileInfo(out, filterG2, "FilterG Level 2", baseCounter);

		writeProfileInfo(out, scharrHorX0, "Scharr X Horizontal Level 0", baseCounter);
		writeProfileInfo(out, scharrVerX0, "Scharr X Vertical Level 0", baseCounter);
		writeProfileInfo(out, scharrHorY0, "Scharr Y Horizontal Level 0", baseCounter);
		writeProfileInfo(out, scharrVerY0, "Scharr Y Vertical Level 0", baseCounter);

		auto maxCounter = filterG0.getProfilingInfo<CL_PROFILING_COMMAND_END>() - baseCounter;
		std::cout << "Max counter: " << maxCounter << std::endl;

		return 0;
	}
	catch (std::exception const& ex)
	{
		std::cout << ex.what() << std::endl;
		return -1;
	}
}
