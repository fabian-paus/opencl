#include "runtime.hpp"

#include <boost/gil/image.hpp>
#include <boost/gil/extension/io/jpeg_io.hpp>

#include <iostream>
#include <fstream>
#include <array>
#include <cstdint>
#include <ctime>
#include <random>

namespace gil = boost::gil;

const std::string FIRST_IMAGE = "images/frame10.jpg";
const std::string SECOND_IMAGE = "images/frame11.jpg";
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

struct RangeColorConverter
{
	RangeColorConverter(int minValue, int maxValue)
		: m_min(minValue), m_max(maxValue)
	{ }

	template <typename SrcT>
	void operator () (SrcT const& src, gil::gray8_pixel_t& dst) const
	{
		const float range = (float)(m_max - m_min);
		int value = src[0] - m_min;
		float t = (float)value / range;
		dst[0] = (uint8_t)(t*255);
	}

	int m_min, m_max;
};

struct RangeColorConverter1
{
	RangeColorConverter1(int minValue, int maxValue)
		: m_min(minValue), m_max(maxValue)
	{ }

	template <typename SrcT>
	void operator () (SrcT const& src, gil::gray8_pixel_t& dst) const
	{
		const float range = (float)(m_max - m_min);
		int value = src[1] - m_min;
		float t = (float)value / range;
		dst[0] = (uint8_t)(t * 255);
	}

	int m_min, m_max;
};

struct RangeColorConverterI
{
	RangeColorConverterI(int index, int minValue, int maxValue)
		: m_index(index), m_min(minValue), m_max(maxValue)
	{ }

	template <typename SrcT>
	void operator () (SrcT const& src, gil::gray8_pixel_t& dst) const
	{
		const float range = (float)(m_max - m_min);
		int value = src[m_index] - m_min;
		float t = (float)value / range;
		dst[0] = (uint8_t)(t * 255);
	}

	int m_index, m_min, m_max;
};

struct RangeColorConverterF
{
	RangeColorConverterF(int index, float minValue, float maxValue)
		: m_index(index), m_min(minValue), m_max(maxValue)
	{ }

	template <typename SrcT>
	void operator () (SrcT const& src, gil::gray8_pixel_t& dst) const
	{
		const float range = (float)(m_max - m_min);
		float value = src[m_index] - m_min;
		float t = (float)value / range;
		dst[0] = (uint8_t)(t * 255);
	}

	int m_index;
	float m_min, m_max;
};

void saveScharrImage(cl::CommandQueue const& queue, cl::Image2D const& source, std::string targetFile, std::vector<cl::Event> const& waitEvents)
{
	TimedEvent event("save_image");
	auto mappedImage = mapImage(queue, source, CL_MAP_READ, &waitEvents);
	auto* mappedImageData = (gil::gray16s_pixel_t*)mappedImage.data;
	auto width = source.getImageInfo<CL_IMAGE_WIDTH>();
	auto height = source.getImageInfo<CL_IMAGE_HEIGHT>();
	auto view = gil::interleaved_view(width, height, mappedImageData, mappedImage.rowSize);
	
	auto rawPixel = view(3, 0);
	int minC = INT16_MAX;
	int maxC = INT16_MIN;
	std::function<void(gil::gray16s_pixel_t const& pix)> minMax = [&](gil::gray16s_pixel_t const& pix)
	{
		if (pix[0] < minC)
			minC = pix[0];
		else if (pix[0] > maxC)
			maxC = pix[0];
	};
	boost::gil::for_each_pixel(view, minMax);
	RangeColorConverter converter(minC, maxC);
	auto colorConverted = boost::gil::color_converted_view<boost::gil::gray8_pixel_t>(view, converter);
	auto convertedPixel = colorConverted(3, 0);
	jpeg_write_view(targetFile, colorConverted);
	queue.enqueueUnmapMemObject(source, mappedImageData);
}

void saveGMatrix(cl::CommandQueue const& queue, cl::Image2D const& source, std::string targetFile, std::vector<cl::Event> const& waitEvents, int index)
{
	TimedEvent event("save_image");
	auto mappedImage = mapImage(queue, source, CL_MAP_READ, &waitEvents);
	auto* mappedImageData = (gil::rgba32s_pixel_t*)mappedImage.data;
	auto width = source.getImageInfo<CL_IMAGE_WIDTH>();
	auto height = source.getImageInfo<CL_IMAGE_HEIGHT>();
	auto view = gil::interleaved_view(width, height, mappedImageData, mappedImage.rowSize);

	int32_t minC = INT32_MAX;
	int32_t maxC = INT32_MIN;
	std::function<void(gil::rgba32s_pixel_t const& pix)> minMax = [&](gil::rgba32s_pixel_t const& pix)
	{
		auto value = pix[index];
		if (value < minC)
			minC = value;
		else if (value > maxC)
			maxC = value;
	};
	boost::gil::for_each_pixel(view, minMax);
	RangeColorConverterI converter(index, minC, maxC);
	auto rawPixel = view(1, 0);
	auto colorConverted = boost::gil::color_converted_view<boost::gil::gray8_pixel_t>(view, converter);
	auto convertedPixel = colorConverted(1, 0);
	jpeg_write_view(targetFile, colorConverted);
	queue.enqueueUnmapMemObject(source, mappedImageData);
}

void saveFlow(cl::CommandQueue const& queue, cl::Image2D const& source, std::string targetFile, std::vector<cl::Event> const& waitEvents, int index)
{
	TimedEvent event("save_image");
	auto mappedImage = mapImage(queue, source, CL_MAP_READ, &waitEvents);
	auto* mappedImageData = (gil::gray32f_pixel_t*)mappedImage.data;
	auto width = source.getImageInfo<CL_IMAGE_WIDTH>();
	auto height = source.getImageInfo<CL_IMAGE_HEIGHT>();
	std::vector<gil::gray32f_pixel_t> oneChannel(width * height);
	for (int i = 0; i < width * height; ++i)
	{
		oneChannel[i] = mappedImageData[2 * i + index];
	}

	auto rowSize = mappedImage.rowSize / 2;
	auto view = gil::interleaved_view(width, height, oneChannel.data(), rowSize);

	float minC = 1000.0f;
	float maxC = -1000.0f;
	std::function<void(gil::gray32f_pixel_t const& pix)> minmax = [&](gil::gray32f_pixel_t const& pix)
	{
		auto value = pix[0];
		if (value < minC)
			minC = value;
		else if (value > maxC)
			maxC = value;
	};
	int maxX = 0;
	int maxY = 0;
	int minX = 0;
	int minY = 0;
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x)
		{
			auto value = view(x, y)[0];
			if (value < minC)
			{
				minC = value;
				minX = x;
				minY = y;
			}
			else if (value > maxC)
			{
				maxC = value;
				maxX = x; 
				maxY = y;
			}
		}
	//boost::gil::for_each_pixel(view, minmax);
	std::cout << targetFile << " min: " << minC << " max: " << maxC << std::endl;
	std::cout << targetFile << " maxX: " << maxX << " maxY: " << maxY << std::endl;
	std::cout << targetFile << " minX: " << minX << " minY: " << minY << std::endl;
	RangeColorConverterF converter(0, minC, maxC);
	auto colorConverted = boost::gil::color_converted_view<boost::gil::gray8_pixel_t>(view, converter);
	jpeg_write_view(targetFile, colorConverted);
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

#if 0 // NDEBUG
const cl_mem_flags INTERMEDIATE_MEMORY_FLAGS = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;
const cl_mem_flags INPUT_MEMORY_FLAGS = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_HOST_WRITE_ONLY;
#else
// Allow host access to the images in debug mode
const cl_mem_flags INTERMEDIATE_MEMORY_FLAGS = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
const cl_mem_flags INPUT_MEMORY_FLAGS = CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY;
#endif
const cl_mem_flags OUTPUT_MEMORY_FLAGS = CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE;

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

const cl::ImageFormat FLOW_VECTOR_FORMAT(CL_RG, CL_FLOAT);

// Helper to get next up value for integer division
static inline size_t DivUp(size_t dividend, size_t divisor)
{
	return (dividend % divisor == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}

class FlowPyramid
{
public:
	FlowPyramid(cl::Context const& context, cl::CommandQueue const& queue, cl::Kernel& calcFlow,
		ImagePyramid const& first, ImagePyramid const& second,
		ScharrPyramid const& derivativeX, ScharrPyramid const& derivativeY,
		GMatrixPyramid const& matrixG)
	{
		std::vector<cl::Event> waitEvents(1);

		for (int i = PYRAMID_HEIGHT - 1; i >= 0; --i)
		{
			auto& dimension = first.getDimension(i);
			m_vectors[i] = createImage(context, OUTPUT_MEMORY_FLAGS, FLOW_VECTOR_FORMAT, dimension);

			calcFlow.setArg(0, first.getImage(i));
			calcFlow.setArg(1, derivativeX.getDerivative(i));
			calcFlow.setArg(2, derivativeY.getDerivative(i));
			calcFlow.setArg(3, matrixG.getMatrix(i));
			calcFlow.setArg(4, second.getImage(i));
			calcFlow.setArg(5, (i == PYRAMID_HEIGHT - 1) ? 0 : 1);
			calcFlow.setArg(6, (i == PYRAMID_HEIGHT - 1) ? m_vectors[i] : m_vectors[i + 1]);
			calcFlow.setArg(7, m_vectors[i]);
			calcFlow.setArg(8, (std::int32_t)dimension[0]);
			calcFlow.setArg(9, (std::int32_t)dimension[1]);
			
			waitEvents[0] = matrixG.getFinished(i);
			if (i != PYRAMID_HEIGHT - 1)
			{
				waitEvents.resize(2);
				waitEvents[1] = m_finished[i + 1];
			}

			auto localWorkSize = cl::NDRange(16, 8);
			auto globalWorkSize = cl::NDRange(localWorkSize[0] * DivUp(dimension[0], localWorkSize[0]),
				localWorkSize[1] * DivUp(dimension[1], localWorkSize[1]));

			queue.enqueueNDRangeKernel(calcFlow, cl::NullRange, globalWorkSize, localWorkSize, &waitEvents, &m_finished[i]);
		}
	}

	cl::Image2D const& getVector(std::size_t level) const { return m_vectors[level]; }

	cl::Event const& getFinished(std::size_t level) const { return m_finished[level]; }

	void writeProfile(std::ostream& out, std::string const& baseName, cl_ulong baseCounter)
	{
		for (std::size_t i = 0; i < PYRAMID_HEIGHT; ++i)
		{
			writeProfileInfo(out, getFinished(i), baseName + " calc flow " + std::to_string(i), baseCounter);
		}
	}

private:
	std::array<cl::Image2D, PYRAMID_HEIGHT> m_vectors;
	std::array<cl::Event, PYRAMID_HEIGHT> m_finished;
};

boost::gil::rgba8_pixel_t randColor()
{
	static std::mt19937 generator;
	static std::uniform_int_distribution<int> distribution(0, 255);
	int r = distribution(generator);
	int g = distribution(generator);
	int b = distribution(generator);
	return boost::gil::rgba8_pixel_t(r, g, b, 255);
}

void drawLines(gil::rgb8_image_t& output, gil::gray8_image_t const& base, cl::Image2D const& vector, cl::CommandQueue const& queue, std::vector<cl::Event> const& waitEvents)
{
	boost::gil::copy_pixels(gil::color_converted_view<gil::rgb8_pixel_t>(const_view(base)), view(output));
	//boost::gil::fill_pixels(view(output), gil::rgba8_pixel_t(0, 0, 0, 0));

	// NOTE: vector ist 4x so klein wie Output
	//  alle 32 Pixel in output soll ein Vektor angebracht werden
	auto width = output.width();
	auto height = output.height();
	auto vectorWidth = width / 4;

	std::default_random_engine generator(2);
	auto outputView = view(output);
	auto STEP_SIZE = 8;
	for (int y = 1; y < height; y += STEP_SIZE)
		for (int x = 1; x < width; x += STEP_SIZE)
		{
			auto mappedImage = mapImage(queue, vector, CL_MAP_READ, &waitEvents);
			auto* mappedImageData = (float*)mappedImage.data;
			auto vectorPosX = x / 4;
			auto vectorPosY = y / 4;
			auto vectorX = mappedImageData[(vectorPosY * vectorWidth + vectorPosX) * 2];
			auto vectorY = mappedImageData[(vectorPosY * vectorWidth + vectorPosX) * 2 + 1];
			//std::cout << "vector(" << vectorPosX << ", " << vectorPosY << "): " 
			//	<< "(" << vectorX << ", " << vectorY << ")" << std::endl;
			float length = std::roundf(vectorX * vectorX + vectorY * vectorY);
			float unitX = (vectorX / length);
			float unitY = (vectorY / length);
			//auto color = randColor();
			//outputView((int)std::roundf(x), (int)std::roundf(y)) = color;
			float maxLength = 4 * length;
			for (int i = 0; i <= (int)maxLength; i += 1)
			{
				int xPos = (int)std::roundf(x + i * unitX);
				int yPos = (int)std::roundf(y + i * unitY);
				int8_t colorValue = (int8_t)(i * 128.0f / maxLength);
				if (xPos >= 0 && xPos < width && yPos >= 0 && yPos < height)
				{
					boost::gil::rgba8_pixel_t color(255, colorValue, 16, 255);
					outputView(xPos, yPos) = color;
				}
			}
		}
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
		auto device = chooseDevice(platform, CL_DEVICE_TYPE_ALL);

		cl::Context context(device);
		cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

		auto program = buildProgram(context, device, PROGRAM_FILE);
		cl::Kernel downFilterX(program, "downfilter_x");
		cl::Kernel downFilterY(program, "downfilter_y");
		cl::Kernel filterG(program, "filter_G");
		cl::Kernel scharrHorX(program, "scharr_x_horizontal");
		cl::Kernel scharrVerX(program, "scharr_x_vertical");
		cl::Kernel scharrHorY(program, "scharr_y_horizontal");
		cl::Kernel scharrVerY(program, "scharr_y_vertical");
		cl::Kernel calcFlow(program, "optical_flow_2");

		cl::ImageFormat format(CL_R, CL_UNSIGNED_INT8);
		std::size_t widthLevel0 = firstImage.width();
		std::size_t heightLevel0 = firstImage.height();

		Timer timer;
		timer.start();

		ImagePyramid firstImagePyramid(firstImage, context, queue, downFilterX, downFilterY);
		ImagePyramid secondImagePyramid(secondImage, context, queue, downFilterX, downFilterY);
		ScharrPyramid derivativeX(context, queue, scharrHorX, scharrVerX, firstImagePyramid);
		ScharrPyramid derivativeY(context, queue, scharrHorY, scharrVerY, firstImagePyramid);

		GMatrixPyramid matrixG(context, queue, filterG, derivativeX, derivativeY);
		FlowPyramid flow(context, queue, calcFlow,
			firstImagePyramid, secondImagePyramid, derivativeX, derivativeY, matrixG);

		for (int i = 0; i < 3; ++i)
		{
			auto& image = firstImagePyramid.getImage(i);
			saveImage(queue, image, "output/first-scaled-" + std::to_string(i) + ".jpg", { firstImagePyramid.getFinished(i) });
		}

		for (int i = 0; i < 3; ++i)
		{
			auto& image = secondImagePyramid.getImage(i);
			saveImage(queue, image, "output/second-scaled-" + std::to_string(i) + ".jpg", { firstImagePyramid.getFinished(i) });
		}

		for (int i = 0; i < 3; ++i)
		{
			auto& image = derivativeX.getDerivative(i);
			saveScharrImage(queue, image, "output/scharr-x-" + std::to_string(i) + ".jpg", { derivativeX.getFinished(i) });
		}

		for (int i = 0; i < 3; ++i)
		{
			auto& image = derivativeY.getDerivative(i);
			saveScharrImage(queue, image, "output/scharr-y-" + std::to_string(i) + ".jpg", { derivativeY.getFinished(i) });
		}

		for (int i = 0; i < 3; ++i)
		{
			auto& image = matrixG.getMatrix(i);
			saveGMatrix(queue, image, "output/g-matrix-0-" + std::to_string(i) + ".jpg", { matrixG.getFinished(i) }, 0);
			saveGMatrix(queue, image, "output/g-matrix-1-" + std::to_string(i) + ".jpg", { matrixG.getFinished(i) }, 1);
			saveGMatrix(queue, image, "output/g-matrix-2-" + std::to_string(i) + ".jpg", { matrixG.getFinished(i) }, 2);
			saveGMatrix(queue, image, "output/g-matrix-3-" + std::to_string(i) + ".jpg", { matrixG.getFinished(i) }, 3);
		}

		for (int i = 0; i < 3; ++i)
		{
			auto& image = flow.getVector(i);
			saveFlow(queue, image, "output/flow-x-" + std::to_string(i) + ".jpg", { flow.getFinished(i) }, 0);
			saveFlow(queue, image, "output/flow-y-" + std::to_string(i) + ".jpg", { flow.getFinished(i) }, 1);
		}

		boost::gil::rgb8_image_t withLines(firstImage.width(), firstImage.height());
		drawLines(withLines, firstImage, flow.getVector(2), queue, { flow.getFinished(2) });
		jpeg_write_view("output/lines.jpeg", view(withLines));

		boost::gil::rgb8_image_t withLines2(firstImage.width(), firstImage.height());
		drawLines(withLines2, secondImage, flow.getVector(2), queue, { flow.getFinished(2) });
		jpeg_write_view("output/lines2.jpeg", view(withLines2));


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
		flow.writeProfile(out, "optical", baseCounter);

		return 0;
	}
	catch (std::exception const& ex)
	{
		std::cout << ex.what() << std::endl;
		return -1;
	}
}
