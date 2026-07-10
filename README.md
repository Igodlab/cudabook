# PMPP

Personal notes and exercise solutions for *Programming Massively Parallel Processors*, 5th Edition, by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj.

This is a learning repository, not a reference implementation. Code favors clarity over performance, and every chapter folder is meant to be read alongside the book rather than on its own. Work is in progress: the repository currently covers roughly the first third of the book, through Chapter 8, and grows as I move through it.

## Stack

CUDA C++ and CMake for GPU kernels, with a Pixi-managed environment (`pixi.toml` / `pixi.lock`) that pins CUDA 13.1, a CUDA-capable conda toolchain, OpenCV for image-based exercises, and a Jupyter/Mojo setup used for occasional comparisons against Mojo's GPU programming model. Build config lives in the root `CMakeLists.txt`, which pulls in each chapter's `src/chXX/cuda-code` subdirectory as it gets added.

```
pixi run build   # configure and build all chapters registered in CMakeLists.txt
pixi run clean   # remove the build directory
```

## Repository layout

```
pmpp/
├── CMakeLists.txt          # top-level build, one add_subdirectory() per chapter
├── pixi.toml / pixi.lock   # environment and task definitions
├── images/
│   └── chXX/                # figures and rendered exercise solutions per chapter
└── src/
    └── chXX/
        ├── cuda-code/        # kernels, exercise solutions, chapter README
        └── mojo-code/        # occasional Mojo ports, where present
```

Each chapter's `README.md` under `src/chXX/cuda-code` contains a summary table of its worked examples and exercises, short writeups of the reasoning behind each solution, and inline diagrams where a figure explains the idea faster than text. A few of those diagrams and rendered outputs are genuinely worth a look on their own, not just as by-products of getting the code to run.

## Chapters

| Chapter | Title | Contents |
|---|---|---|
| 1 | Introduction | Conceptual figures only (Amdahl's law, the "peach" heterogeneous-computing analogy); no code |
| 2 | Heterogeneous Data Parallel Computing | `vecAddKernel`, host/device memory management boilerplate |
| 3 | Multidimensional Grids and Data | Color-to-greyscale, image blur, matrix multiplication, plus a shared tensor-indexing notation used for the rest of the repo; 5 exercises |
| 4 | Compute Architecture and Scheduling | Device-property query utility; 9 exercises on warp divergence, occupancy, and resource partitioning |
| 5 | Memory Architecture and Data Locality | Tiled matrix multiplication (static, dynamic, and hardware-optimal tile sizes); 12 exercises on shared memory, the roofline model, and occupancy limits |
| 6 | Performance Considerations | WIP |
| 7 | Convolution | WIP |
| 8 | Stencil | Parallel 3D stencil kernel; exercises in progress |

Chapters 6 and 7 are placeholders reserved in the table above for continuity; the corresponding `src/` folders don't exist yet, and the README will be updated as they're filled in.

## Notable visuals

A running theme of this repo is that I try to make the supporting figures genuinely good, not just placeholder screenshots. A few worth calling out:

- **Chapter 3** turns an Opeth album cover into the color-to-greyscale and image-blur running example, so the input, greyscale, and blurred outputs sit side by side as a real before/after rather than a synthetic test pattern. The row/column matmul exercise also ships with a hand-drawn indexing diagram that makes the coalescing tradeoff obvious at a glance.
- **Chapter 5** includes a clean tiled-matmul schematic showing the cooperative shared-memory load pattern, alongside a properly rendered roofline model plot for the compute-bound-versus-memory-bound exercise and a worked 8x8-matrix tiling diagram comparing 2x2 and 4x4 tile sizes.
- **Chapter 1** keeps it simple but effective, with a clean redraw of Amdahl's law and the classic heterogeneous-computing peach analogy.

## Notation

Starting in Chapter 3, the repository fixes a consistent right-to-left (fast-to-slow varying index) subscript convention for tensors, and maps it explicitly across generic math notation, deep-learning-style dimension names, and CUDA's `threadIdx`/`blockIdx` fields. See `src/ch03/cuda-code/README.md` for the full table; later chapters assume it without repeating it.

## Notes

This is a personal study log. Solutions reflect my own reasoning and may not match the "official" or most efficient answer; corrections and alternative approaches are welcome via issues, but pull requests are not the point of this repository.
