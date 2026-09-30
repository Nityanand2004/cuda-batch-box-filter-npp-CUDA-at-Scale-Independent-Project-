# CUDA at Scale: Batch Image Processing with NPP

## Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Prerequisites](#prerequisites)
4. [Installation & Setup](#installation--setup)
5. [Usage & Execution](#usage--execution)
6. [Technical Architecture](#technical-architecture)
7. [Challenges & Optimizations](#challenges--optimizations)

---

## Project Overview

This repository contains my submission for the **"CUDA at Scale" Independent Project**. The objective of this project is to demonstrate GPU-accelerated image processing on a large volume of data in a single execution.

Instead of a basic single-image processor, this application implements a robust C++ batch processor using the **NVIDIA Performance Primitives (NPP)** library. It dynamically scans an input directory, filters unsupported files, and applies a GPU-accelerated box blur filter to hundreds of images sequentially, saving the processed outputs to a target directory.

---

## Directory Structure

```text
boxFilterNPP/
│
├── input_images/             # Original .pgm image dataset
├── output_images/            # GPU processed outputs
├── Common/                   # NVIDIA CUDA helper headers
│
├── boxFilterNPP.cpp          # Main CUDA/NPP application
├── Makefile                  # Build configuration
├── run.sh                    # Batch execution script
└── README.md                 # Documentation
```

---

## Prerequisites

- Linux environment (Ubuntu/Debian recommended)
- NVIDIA GPU with CUDA support
- CUDA Toolkit
- NVIDIA NPP Library
- C++11 or later

---

## Installation & Setup

### Clone Repository

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd boxFilterNPP
```

### Create Input/Output Directories

```bash
mkdir -p input_images output_images
```

Place `.pgm` images inside `input_images/`.

### Build

```bash
make
```

This generates the executable:

```text
boxFilterNPP
```

---

## Usage & Execution

### Run the Program

```bash
./boxFilterNPP -input_dir=./input_images -output_dir=./output_images
```

### Example Console Output

```text
Starting Batch Processing...
Processing: Lena1.pgm
Saved: filtered_Lena1.pgm

Processing: invalid_file.png
Skipped invalid_file.png

Batch processing complete!
```

---

## Technical Architecture

### GPU Filtering

The project uses NVIDIA NPP for optimized GPU image filtering.

Function used:

```cpp
nppiFilterBoxBorder_8u_C1R
```

The filter applies a 5×5 spatial averaging mask across the image using GPU parallelism.

### Memory Flow

```text
Host Memory
   ↓
GPU Device Memory
   ↓
NPP Filter Execution
   ↓
Host Memory
```

For each image:

1. Load image on CPU
2. Transfer to GPU
3. Execute NPP filter
4. Copy processed image back
5. Save output image

---

## Challenges & Optimizations

### Memory Management

One challenge involved repeated GPU execution inside loops. Manual memory deallocation caused CUDA context instability due to double-free behavior. Removing redundant `nppiFree()` calls resolved the issue.

### Fault Tolerance

The application safely handles unsupported files using exception handling, allowing batch execution to continue even when invalid images are encountered.