/*
* Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
    VPIImage image   = NULL;
    VPIImage blurred = NULL;
    VPIStream stream = NULL;

    VPIEvent evStart = NULL;
    VPIEvent evStop  = NULL;

    int retval = 0;

    try
    {
        // 1. Processing of command line parameters -----------

        if (argc != 2)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] + " <cpu|pva|cuda>");
        }

        std::string strBackend = argv[1];

        // Parse the backend
        VPIBackend backend;

        if (strBackend == "cpu")
        {
            backend = VPI_BACKEND_CPU;
        }
        else if (strBackend == "cuda")
        {
            backend = VPI_BACKEND_CUDA;
        }
        else if (strBackend == "pva")
        {
            backend = VPI_BACKEND_PVA;
        }
        else
        {
            throw std::runtime_error("Backend '" + strBackend +
                                     "' not recognized, it must be either cpu, cuda or pva.");
        }

        // 2. Initialization stage ----------------------

        // Create the stream for the given backend.
        CHECK_STATUS(vpiStreamCreate(0, &stream));

        int width = 1920, height = 1080;
        VPIImageFormat imgFormat = VPI_IMAGE_FORMAT_U16;

        std::cout << "Input size: " << width << " x " << height << '\n'
                  << "Image format: " << vpiImageFormatGetName(imgFormat) << '\n'
                  << "Algorithm: 3x3 Median Filter" << std::endl;

        // Memory flags set to guarantee top performance.
        // Only the benchmarked backend is enabled, and memories
        // are guaranteed to be used by only one stream.
        uint64_t memFlags = backend | VPI_EXCLUSIVE_STREAM_ACCESS;

        // Create image with zero content
        CHECK_STATUS(vpiImageCreate(width, height, imgFormat, memFlags, &image));

        // Create a temporary image convolved with a low-pass filter.
        CHECK_STATUS(vpiImageCreate(width, height, imgFormat, memFlags, &blurred));

        // Create the events we'll need to get timing info
        CHECK_STATUS(vpiEventCreate(0, &evStart));
        CHECK_STATUS(vpiEventCreate(0, &evStop));

        int8_t kernel[3 * 3] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };

        // 3. Gather timings --------------------

        const int BATCH_COUNT     = 25;
        const int AVERAGING_COUNT = 50;

        // Collect measurements for each execution batch
        std::vector<float> timingsMS;
        for (int batch = 0; batch < BATCH_COUNT; ++batch)
        {

            printf("BATCH %d\n", batch);
            // Record stream queue when we start processing
            CHECK_STATUS(vpiEventRecord(evStart, stream));

            // Get the average running time within this batch.
            for (int i = 0; i < AVERAGING_COUNT; ++i)
            {
                // Call the algorithm to be measured.
                vpiSubmitMedianFilter(stream, backend, image, blurred, 3, 3, kernel, VPI_BORDER_ZERO);
            }

            // Record stream queue just after blurring
            CHECK_STATUS(vpiEventRecord(evStop, stream));

            // Wait until the batch processing is done
            CHECK_STATUS(vpiEventSync(evStop));

            float elapsedMS;
            CHECK_STATUS(vpiEventElapsedTimeMillis(evStart, evStop, &elapsedMS));
            timingsMS.push_back(elapsedMS / AVERAGING_COUNT);
        }

        // 4. Performance analysis ----------------------

        // Get the median of the measurements so that outliers aren't considered.
        nth_element(timingsMS.begin(), timingsMS.begin() + timingsMS.size() / 2, timingsMS.end());
        float medianMS = timingsMS[timingsMS.size() / 2];

        printf("Approximated elapsed time per call: %f ms\n", medianMS);
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }

    // 4. Clean up -----------------------------------

    // Destroy stream first, it'll make sure all processing
    // submitted to it is finished.
    vpiStreamDestroy(stream);

    // Now we can destroy other VPI objects, since they aren't being
    // used anymore.
    vpiImageDestroy(image);
    vpiImageDestroy(blurred);
    vpiEventDestroy(evStart);
    vpiEventDestroy(evStop);

    return retval;
}
