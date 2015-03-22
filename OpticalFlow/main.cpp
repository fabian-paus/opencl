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

cl::Image2D createImage(cl::Context const& context, cl_mem_flags memFlags, cl::ImageFormat const& format, cl::NDRange const& dimension)
{
	return cl::Image2D(context, memFlags, format, dimension[0], dimension[1]);
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
			m_images[i] = createImage(context, memoryFlags, IMAGE_FORMAT, m_dimensions[i]);
		}

		// Copy level 0
		m_finished[0] = copyImage(queue, image, m_images[0]);

		// Downfiltering for levels 1 and 2
		std::vector<cl::Event> waitEvents(1);

		for (std::size_t i = 0; i < PYRAMID_HEIGHT - 1; ++i)
		{
			m_intermediateImages[i] = createImage(context, INTERMEDIATE_MEMORY_FLAGS, IMAGE_FORMAT, m_dimensions[i]);
			downFilterX.setArg(0, m_images[i]);
			downFilterX.setArg(1, m_intermediateImages[i]);

			waitEvents[0] = m_finished[i];
			queue.enqueueNDRangeKernel(downFilterX, cl::NullRange, m_dimensions[i], cl::NullRange, &waitEvents, &m_intermediateEvents[i]);

			downFilterY.setArg(0, m_intermediateImages[i]);
			downFilterY.setArg(1, m_images[i + 1]);

			waitEvents[0] = m_intermediateEvents[i];
			queue.enqueueNDRangeKernel(downFilterY, cl::NullRange, m_dimensions[i + 1], cl::NullRange, &waitEvents, &m_finished[i + 1]);
		}
	}

	cl::Image2D const& getImage(std::size_t level) const { return m_images[level]; }

	cl::NDRange const& getDimension(std::size_t level) const { return m_dimensions[level]; }

	cl::Event const& getFinished(std::size_t level) const { return m_finished[level]; }

	void writeProfile(std::ostream& out, std::string const& baseName, cl_ulong baseCounter)
	{
		writeProfileInfo(out, getFinished(0), baseName + " copy", baseCounter);

		for (std::size_t i = 0; i < PYRAMID_HEIGHT - 1; ++i)
		{
			writeProfileInfo(out, m_intermediateEvents[i], baseName + " downfilter X level " + std::to_string(i + 1), baseCounter);
			writeProfileInfo(out, getFinished(i + 1), baseName + " downfilter Y level " + std::to_string(i + 1), baseCounter);
		}
	}

private:
	std::array<cl::Image2D, PYRAMID_HEIGHT> m_images;
	std::array<cl::NDRange, PYRAMID_HEIGHT> m_dimensions;
	std::array<cl::Event, PYRAMID_HEIGHT> m_finished;

	std::array<cl::Image2D, PYRAMID_HEIGHT - 1> m_intermediateImages;
	std::array<cl::Event, PYRAMID_HEIGHT - 1> m_intermediateEvents;
};

const cl::ImageFormat SCHARR_FORMAT(CL_R, CL_SIGNED_INT16);

class ScharrPyramid
{
public:
	ScharrPyramid(cl::Context const& context, cl::CommandQueue const& queue, cl::Kernel& filterHorizontal, cl::Kernel& filterVertical,
		ImagePyramid const& basePyramid)
	{
		std::vector<cl::Event> waitEvents(1);

		for (std::size_t i = 0; i < PYRAMID_HEIGHT; ++i)
		{
			auto& dimension = basePyramid.getDimension(i);
			m_dimensions[i] = dimension;

			m_intermediates[i] = createImage(context, INTERMEDIATE_MEMORY_FLAGS, SCHARR_FORMAT, dimension);
			filterHorizontal.setArg(0, basePyramid.getImage(i));
			filterHorizontal.setArg(1, m_intermediates[i]);
			waitEvents[0] = basePyramid.getFinished(i);
			queue.enqueueNDRangeKernel(filterHorizontal, cl::NullRange, dimension, cl::NullRange, &waitEvents, &m_intermediateEvents[i]);

			m_derivatives[i] = createImage(context, INTERMEDIATE_MEMORY_FLAGS, SCHARR_FORMAT, dimension);
			filterVertical.setArg(0, m_intermediates[i]);
			filterVertical.setArg(1, m_derivatives[i]);
			waitEvents[0] = m_intermediateEvents[i];
			queue.enqueueNDRangeKernel(filterVertical, cl::NullRange, dimension, cl::NullRange, &waitEvents, &m_finished[i]);
		}
	}

	cl::Image2D const& getDerivative(std::size_t level) const { return m_derivatives[level]; }

	cl::NDRange const& getDimension(std::size_t level) const { return m_dimensions[level]; }

	cl::Event const& getFinished(std::size_t level) const { return m_finished[level]; }

	void writeProfile(std::ostream& out, std::string const& baseName, cl_ulong baseCounter)
	{
		for (std::size_t i = 0; i < PYRAMID_HEIGHT; ++i)
		{
			writeProfileInfo(out, m_intermediateEvents[i], baseName + " scharr hor level " + std::to_string(i), baseCounter);
			writeProfileInfo(out, getFinished(i), baseName + " scharr ver level " + std::to_string(i), baseCounter);
		}
	}

private:
	std::array<cl::Image2D, PYRAMID_HEIGHT> m_derivatives;
	std::array<cl::Image2D, PYRAMID_HEIGHT> m_intermediates;
	std::array<cl::NDRange, PYRAMID_HEIGHT> m_dimensions;
	std::array<cl::Event, PYRAMID_HEIGHT> m_finished;
	std::array<cl::Event, PYRAMID_HEIGHT> m_intermediateEvents;
};

const cl::ImageFormat G_MATRIX_FORMAT(CL_RGBA, CL_SIGNED_INT32);

class GMatrixPyramid
{
public:
	GMatrixPyramid(cl::Context const& context, cl::CommandQueue const& queue, cl::Kernel& filterG,
		ScharrPyramid const& derivativeX, ScharrPyramid const& derivativeY)
	{
		std::vector<cl::Event> waitEvents(2);

		for (std::size_t i = 0; i < PYRAMID_HEIGHT; ++i)
		{
			auto& dimension = derivativeX.getDimension(i);
			m_matrices[i] = createImage(context, INTERMEDIATE_MEMORY_FLAGS, G_MATRIX_FORMAT, derivativeX.getDimension(i));

			filterG.setArg(0, derivativeX.getDerivative(i));
			filterG.setArg(1, derivativeY.getDerivative(i));
			filterG.setArg(2, m_matrices[i]);

			waitEvents[0] = derivativeX.getFinished(i);
			waitEvents[1] = derivativeY.getFinished(i);
			queue.enqueueNDRangeKernel(filterG, cl::NullRange, dimension, cl::NullRange, &waitEvents, &m_finished[i]);
		}
	}

	cl::Image2D const& getMatrix(std::size_t level) const { return m_matrices[level]; }

	cl::Event const& getFinished(std::size_t level) const { return m_finished[level]; }

	void writeProfile(std::ostream& out, std::string const& baseName, cl_ulong baseCounter)
	{
		for (std::size_t i = 0; i < PYRAMID_HEIGHT; ++i)
		{
			writeProfileInfo(out, getFinished(i), baseName + " filter G level " + std::to_string(i), baseCounter);
		}
	}

private:
	std::array<cl::Image2D, PYRAMID_HEIGHT> m_matrices;
	std::array<cl::Event, PYRAMID_HEIGHT> m_finished;
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

		ImagePyramid firstImagePyramid(firstImage, context, queue, downFilterX, downFilterY);
		ScharrPyramid derivativeX(context, queue, scharrHorX, scharrVerX, firstImagePyramid);
		ScharrPyramid derivativeY(context, queue, scharrHorY, scharrVerY, firstImagePyramid);
		ImagePyramid secondImagePyramid(secondImage, context, queue, downFilterX, downFilterY);

		GMatrixPyramid matrixG(context, queue, filterG, derivativeX, derivativeY);

		queue.finish();
		timer.stop("down_filter_all");

		std::ofstream out("profile.csv");

		auto baseCounter = firstImagePyramid.getFinished(0).getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();

		out << ";Not Existing;Queued;Submitted;Running\n";

		firstImagePyramid.writeProfile(out, "image 1", baseCounter);
		secondImagePyramid.writeProfile(out, "image 2", baseCounter);
		derivativeX.writeProfile(out, "X", baseCounter);
		derivativeY.writeProfile(out, "Y", baseCounter);
		matrixG.writeProfile(out, "matrix", baseCounter);

		return 0;
	}
	catch (std::exception const& ex)
	{
		std::cout << ex.what() << std::endl;
		return -1;
	}
}
