#include <vpi/Event.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/GaussianFilter.h>
#include <vpi/algo/MedianFilter.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>

#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);

int main(int argc, char *argv[])
{
    int retval = 0;

    try
    {
        // ------------------------------
        // 1. Parse command-line arguments
        // ------------------------------
        if (argc != 4)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] +
                                     " <cpu|cuda|pva> <num_streams> <kernel_size>");
        }

        std::string strBackend = argv[1];
        int numStreams = std::stoi(argv[2]);
        int kernelSize = std::stoi(argv[3]);

        if (numStreams <= 0)
            throw std::runtime_error("Number of streams must be greater than 0.");
        if (kernelSize <= 0 || kernelSize % 2 == 0)
            throw std::runtime_error("Kernel size must be a positive odd number.");

        // ------------------------------
        // 2. Backend Selection
        // ------------------------------
        VPIBackend backend;
        if (strBackend == "cpu")
            backend = VPI_BACKEND_CPU;
        else if (strBackend == "cuda")
            backend = VPI_BACKEND_CUDA;
        else if (strBackend == "pva")
            backend = VPI_BACKEND_PVA;
        else
            throw std::runtime_error("Backend '" + strBackend + "' not recognized.");

        // ------------------------------
        // 3. Initialization
        // ------------------------------
        int width = 1920, height = 1080;
        VPIImageFormat imgFormat = VPI_IMAGE_FORMAT_U16;

        std::cout << "Input size: " << width << " x " << height << '\n'
                  << "Backend: " << strBackend << '\n'
                  << "Streams: " << numStreams << '\n'
                  << "Kernel Size: " << kernelSize << "x" << kernelSize << '\n';

        uint64_t memFlags = backend | VPI_EXCLUSIVE_STREAM_ACCESS;

        std::vector<int8_t> kernel(kernelSize * kernelSize, 1);

        std::vector<VPIStream> streams(numStreams);
        std::vector<VPIImage> images(numStreams);
        std::vector<VPIImage> outputs(numStreams);

        for (int i = 0; i < numStreams; ++i)
        {
            CHECK_STATUS(vpiStreamCreate(0, &streams[i]));
            CHECK_STATUS(vpiImageCreate(width, height, imgFormat, memFlags, &images[i]));
            CHECK_STATUS(vpiImageCreate(width, height, imgFormat, memFlags, &outputs[i]));
        }

        // Timing events
        VPIEvent evStart = NULL;
        CHECK_STATUS(vpiEventCreate(0, &evStart));

        std::vector<VPIEvent> evStops(numStreams);
        for (int i = 0; i < numStreams; ++i)
        {
            CHECK_STATUS(vpiEventCreate(0, &evStops[i]));
        }

        // ------------------------------
        // 4. Benchmarking
        // ------------------------------
        const int BATCH_COUNT = 5;
        const int AVERAGING_COUNT = 20;

        std::vector<float> timingsMS;

        for (int batch = 0; batch < BATCH_COUNT; ++batch)
        {
            printf("BATCH %d\n", batch);

            CHECK_STATUS(vpiEventRecord(evStart, streams[0]));

            for (int i = 0; i < AVERAGING_COUNT; ++i)
            {
                for (int streamIdx = 0; streamIdx < numStreams; streamIdx++)
                {
                    CHECK_STATUS(vpiSubmitMedianFilter(
                        streams[streamIdx],
                        backend,
                        images[streamIdx],
                        outputs[streamIdx],
                        kernelSize,
                        kernelSize,
                        kernel.data(),
                        VPI_BORDER_LIMITED));
                }
            }

            // Record stop events for each stream
            for (int i = 0; i < numStreams; ++i)
            {
                CHECK_STATUS(vpiEventRecord(evStops[i], streams[i]));
            }

            // Wait for all streams and find the max elapsed time
            float maxElapsedMS = 0.0f;
            for (int i = 0; i < numStreams; ++i)
            {
                CHECK_STATUS(vpiEventSync(evStops[i]));
                float streamElapsed;
                CHECK_STATUS(vpiEventElapsedTimeMillis(evStart, evStops[i], &streamElapsed));
                if (streamElapsed > maxElapsedMS)
                    maxElapsedMS = streamElapsed;
            }

            timingsMS.push_back(maxElapsedMS / AVERAGING_COUNT);
        }

        // ------------------------------
        // 5. Performance Analysis
        // ------------------------------
        std::nth_element(timingsMS.begin(), timingsMS.begin() + timingsMS.size() / 2, timingsMS.end());
        float medianMS = timingsMS[timingsMS.size() / 2];

        printf("Approximated elapsed time per call: %.4f ms\n", medianMS);

        // ------------------------------
        // 6. Cleanup
        // ------------------------------
        for (int i = 0; i < numStreams; ++i)
        {
            vpiStreamDestroy(streams[i]);
            vpiImageDestroy(images[i]);
            vpiImageDestroy(outputs[i]);
            vpiEventDestroy(evStops[i]);
        }

        vpiEventDestroy(evStart);
    }
    catch (std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        retval = 1;
    }

    return retval;
}

