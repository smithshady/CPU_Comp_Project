/*
* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <opencv2/core/version.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/ImageFormat.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/BackgroundSubtractor.h>
#include <vpi/algo/ConvertImageFormat.h>

// Maybe don't need all of these
#include <vpi/Event.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/GaussianFilter.h>

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
    // OpenCV image that will be wrapped by a VPIImage.
    // Define it here so that it's destroyed *after* wrapper is destroyed
    cv::Mat cvCurFrame;

    // VPI objects that will be used
    VPIStream stream     = NULL;
    VPIImage imgCurFrame = NULL;
    VPIImage bgimage     = NULL;
    VPIImage fgmask      = NULL;
    VPIPayload payload   = NULL;

    // Timing objects
    VPIEvent evStart = NULL;
    VPIEvent evStop  = NULL;

    int retval = 0;

    try
    {
        if (argc != 3)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] + " <cpu|cuda> <input_video>");
        }

        // Parse input parameters
        std::string strBackend    = argv[1];
        std::string strInputVideo = argv[2];

        VPIBackend backend;
        if (strBackend == "cpu")
        {
            backend = VPI_BACKEND_CPU;
        }
        else if (strBackend == "cuda")
        {
            backend = VPI_BACKEND_CUDA;
        }
        else
        {
            throw std::runtime_error("Backend '" + strBackend + "' not recognized.");
        }

        // Load the input video
        cv::VideoCapture invid;
        if (!invid.open(strInputVideo))
        {
            throw std::runtime_error("Can't open '" + strInputVideo + "'");
        }

        int32_t width  = invid.get(cv::CAP_PROP_FRAME_WIDTH);
        int32_t height = invid.get(cv::CAP_PROP_FRAME_HEIGHT);

        ///////////// INITIALIZATION STAGE ///////////////

        // Create the stream where processing will happen. We'll use user-provided backend.
        CHECK_STATUS(vpiStreamCreate(backend, &stream));

        // Create background subtractor payload to be executed on the given backend
        // OpenCV delivers us BGR8 images, so the algorithm is configured to accept that.
        CHECK_STATUS(vpiCreateBackgroundSubtractor(backend, width, height, VPI_IMAGE_FORMAT_BGR8, &payload));

        // Memory flags set to guarantee top performance.
        // Only the benchmarked backend is enabled, and memories
        // are guaranteed to be used by only one stream.
        // uint64_t memFlags = backend | VPI_EXCLUSIVE_STREAM_ACCESS;

        // Create foreground image
        CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, 0, &fgmask));

        // Create background image
        CHECK_STATUS(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_BGR8, 0, &bgimage));

        // Create the events we'll need to get timing info
        CHECK_STATUS(vpiEventCreate(0, &evStart));
        CHECK_STATUS(vpiEventCreate(0, &evStop));

        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        double fps = invid.get(cv::CAP_PROP_FPS);

        cv::VideoWriter outVideo("fgmask_" + strBackend + ".mp4", fourcc, fps, cv::Size(width, height), false);
        if (!outVideo.isOpened())
        {
            throw std::runtime_error("Can't create output video");
        }

        cv::VideoWriter bgimageVideo("bgimage_" + strBackend + ".mp4", fourcc, fps, cv::Size(width, height));
        if (!outVideo.isOpened())
        {
            throw std::runtime_error("Can't create output video");
        }

        //////////////// MAIN LOOP, GATHER TIMINGS ////////////////////

        const int AVERAGING_COUNT = 5;

        // Fetch a new frame until video ends
        int idxFrame = 1;

        // Create timings arr
        std::vector<float> timingsMS;

        while (invid.read(cvCurFrame))
        {
            printf("Processing frame (batch) %d\n", idxFrame++);
            // Wrap frame into a VPIImage
            if (imgCurFrame == NULL)
            {
                CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(cvCurFrame, 0, &imgCurFrame));
            }
            else
            {
                CHECK_STATUS(vpiImageSetWrappedOpenCVMat(imgCurFrame, cvCurFrame));
            }

            VPIBackgroundSubtractorParams params;
            CHECK_STATUS(vpiInitBackgroundSubtractorParams(&params));
            params.learningRate = 0.01;

            // Record stream queue when we start processing
            CHECK_STATUS(vpiEventRecord(evStart, stream));

            // Get the average running time within this batch.
            for (int i = 0; i < AVERAGING_COUNT; ++i)
            {
                printf("--> Run %d\n", i);
                CHECK_STATUS(
                    vpiSubmitBackgroundSubtractor(stream, backend, payload, imgCurFrame, fgmask, bgimage, &params));
            }

            // Record stream queue just after blurring
            CHECK_STATUS(vpiEventRecord(evStop, stream));

            // Wait until the batch processing is done
            CHECK_STATUS(vpiEventSync(evStop));

            float elapsedMS;
            CHECK_STATUS(vpiEventElapsedTimeMillis(evStart, evStop, &elapsedMS));
            timingsMS.push_back(elapsedMS / AVERAGING_COUNT);

            // Wait for processing to finish.
            CHECK_STATUS(vpiStreamSync(stream));

            {
                // Now add it to the output video stream
                VPIImageData imgdata;
                CHECK_STATUS(vpiImageLockData(fgmask, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgdata));

                cv::Mat outFrame;
                CHECK_STATUS(vpiImageDataExportOpenCVMat(imgdata, &outFrame));

                outVideo << outFrame;

                CHECK_STATUS(vpiImageUnlock(fgmask));
            }

            {
                VPIImageData bgdata;
                CHECK_STATUS(vpiImageLockData(bgimage, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &bgdata));

                cv::Mat outFrame;
                CHECK_STATUS(vpiImageDataExportOpenCVMat(bgdata, &outFrame));

                bgimageVideo << outFrame;

                CHECK_STATUS(vpiImageUnlock(bgimage));
            }
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

    // Destroy all resources used
    vpiStreamDestroy(stream);
    vpiPayloadDestroy(payload);

    vpiImageDestroy(imgCurFrame);
    vpiImageDestroy(fgmask);
    vpiImageDestroy(bgimage);

    return retval;
}
