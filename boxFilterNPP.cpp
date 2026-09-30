/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved. ... (Standard NVIDIA Copyright) */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>
#include <dirent.h>   // Added for directory parsing
#include <sys/stat.h> // Added for directory checking

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

bool printfNPPinfo(int argc, char *argv[])
{
  const NppLibraryVersion *libVer = nppGetLibVersion();
  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

  return checkCudaCapabilities(1, 0);
}

int main(int argc, char *argv[])
{
  printf("%s Starting Batch Processing...\n\n", argv[0]);

  try
  {
    findCudaDevice(argc, (const char **)argv);

    if (printfNPPinfo(argc, argv) == false)
    {
      exit(EXIT_SUCCESS);
    }

    // 1. Setup CLI Arguments for Directories
    std::string inputDir = "./input_images";
    std::string outputDir = "./output_images";
    char *inPath = NULL, *outPath = NULL;

    if (checkCmdLineFlag(argc, (const char **)argv, "input_dir"))
    {
      getCmdLineArgumentString(argc, (const char **)argv, "input_dir", &inPath);
      inputDir = inPath;
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "output_dir"))
    {
      getCmdLineArgumentString(argc, (const char **)argv, "output_dir", &outPath);
      outputDir = outPath;
    }

    std::cout << "Target Input Directory: " << inputDir << std::endl;
    std::cout << "Target Output Directory: " << outputDir << std::endl;

    // 2. Open the input directory
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(inputDir.c_str())) != NULL)
    {

      // 3. Loop through every file in the directory
      while ((ent = readdir(dir)) != NULL)
      {
        std::string filename = ent->d_name;

        // Skip current and parent directory markers
        if (filename == "." || filename == "..")
          continue;

        std::string sFilename = inputDir + "/" + filename;
        std::string sResultFilename = outputDir + "/filtered_" + filename;

        std::cout << "--------------------------------------\n";
        std::cout << "Processing: " << sFilename << std::endl;

        // Wrap the NPP logic in a try-catch so one bad file doesn't crash the whole batch
        try
        {
          // --- NPP GPU PROCESSING LOGIC ---
          npp::ImageCPU_8u_C1 oHostSrc;

          // The helper library throws an exception if it's not a valid image format (like PGM)
          npp::loadImage(sFilename, oHostSrc);
          npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

          NppiSize oMaskSize = {5, 5};
          NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
          NppiPoint oSrcOffset = {0, 0};
          NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

          npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
          NppiPoint oAnchor = {oMaskSize.width / 2, oMaskSize.height / 2};

          // Run box filter kernel
          NPP_CHECK_NPP(nppiFilterBoxBorder_8u_C1R(
              oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
              oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, oMaskSize, oAnchor,
              NPP_BORDER_REPLICATE));

          npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
          oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

          saveImage(sResultFilename, oHostDst);
          std::cout << "Saved: " << sResultFilename << std::endl;

          // nppiFree(oDeviceSrc.data());
          // nppiFree(oDeviceDst.data());
          // --- END NPP GPU PROCESSING LOGIC ---
        }
        catch (npp::Exception &rException)
        {
          std::cerr << "Skipped " << filename << " (Likely not a supported image format)." << std::endl;
        }
        catch (...)
        {
          std::cerr << "Skipped " << filename << " due to an unknown error." << std::endl;
        }
      }
      closedir(dir);
    }
    else
    {
      std::cerr << "ERROR: Could not open input directory: " << inputDir << std::endl;
      std::cerr << "Make sure you created the 'input_images' folder!" << std::endl;
      return EXIT_FAILURE;
    }

    std::cout << "\nBatch processing complete!" << std::endl;
    exit(EXIT_SUCCESS);
  }
  catch (npp::Exception &rException)
  {
    std::cerr << "Fatal Program error! \n"
              << rException << std::endl;
    exit(EXIT_FAILURE);
  }
  catch (...)
  {
    std::cerr << "Fatal Program error! \n";
    exit(EXIT_FAILURE);
  }
}