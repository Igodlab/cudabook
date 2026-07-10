#pragma once
#include <string>
#include <vector>

/* ---------------------------------------------------------------------------
 * Simulation parameters for the 3D heat equation  dT/dt = alpha * laplacian(T)
 *
 * Grid layout (row-major, x fastest) -- matches coalesced CUDA access:
 *     idx = k * (m * n) + j * m + i
 *   i in [0, m)  -> x (width)
 *   j in [0, n)  -> y (height)   (j = n-1 is the TOP face)
 *   k in [0, p)  -> z (depth)
 * ---------------------------------------------------------------------------
 */
struct SimParams {
    int   m = 64;            /* grid points along x (width)  */
    int   n = 64;            /* grid points along y (height) */
    int   p = 64;            /* grid points along z (depth)  */

    float h       = 0.1f / 63.0f;  /* grid spacing [m] (10 cm cube default) */
    float alpha   = 9.7e-5f;       /* thermal diffusivity [m^2/s] (aluminum) */

    float ambientT = 293.0f;       /* initial metal temperature [K] */
    float lavaT    = 1400.0f;      /* peak lava temperature [K]     */

    /* Lava drop: Gaussian blob centered on the TOP face (j = n-1),
     * offset toward the upper-right corner in x and z.
     */
    float centerX  = 0.80f;        /* fraction of (m-1)              */
    float centerZ  = 0.80f;        /* fraction of (p-1)              */
    float sigma    = 7.0f;         /* Gaussian std-dev, in CELLS     */
};

/* Flat index helper (host-side twin of what the CUDA kernel will use). */
inline int idx3D(int i, int j, int k, int m, int n) {
    return k * (m * n) + j * m + i;
}

/* Fill the whole grid with the ambient temperature. */
void initializeAmbient(std::vector<float>& T, const SimParams& prm);

/* Superimpose the Gaussian lava hot spot (pure initial condition -- it will
 * simply diffuse and cool; there is no sustained source).
 */
void applyLavaDrop(std::vector<float>& T, const SimParams& prm);

/* Write the grid to CSV. Format:
 *   line 1: m,n,p
 *   line 2: h,alpha
 *   then n*p lines, each holding one x-row of m comma-separated values,
 *   in row-major order (i fastest, then j, then k) -- identical to the
 *   in-memory layout, so the solver can stream it straight into a buffer.
 */
bool writeGridToCSV(const std::vector<float>& T, const SimParams& prm,
                    const std::string& filename);

/* Read a grid back from CSV (for the CUDA host code). Returns false on
 * error; on success fills T and overwrites m, n, p, h, alpha in prm.
 */
bool readGridFromCSV(std::vector<float>& T, SimParams& prm,
                     const std::string& filename);

/* Largest stable explicit (FTCS) time step in 3D: h^2 / (6 * alpha). */
float stableTimeStep(const SimParams& prm);

/* Mean temperature -- with insulated (Neumann) faces this is conserved and
 * equals the uniform equilibrium temperature. Useful solver sanity check.
 */
float meanTemperature(const std::vector<float>& T);
