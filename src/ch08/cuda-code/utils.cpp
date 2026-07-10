#include "utils.hpp"

#include <cmath>
#include <cstdio>
#include <numeric>

void initializeAmbient(std::vector<float>& T, const SimParams& prm) {
    T.assign(static_cast<size_t>(prm.m) * prm.n * prm.p, prm.ambientT);
}

void applyLavaDrop(std::vector<float>& T, const SimParams& prm) {
    /* Blob center sits ON the top face (j = n-1), upper-right in x/z. */
    const float cx = prm.centerX * (prm.m - 1);
    const float cy = static_cast<float>(prm.n - 1);
    const float cz = prm.centerZ * (prm.p - 1);

    const float dT      = prm.lavaT - prm.ambientT;
    const float inv2s2  = 1.0f / (2.0f * prm.sigma * prm.sigma);

    for (int k = 0; k < prm.p; ++k) {
        for (int j = 0; j < prm.n; ++j) {
            for (int i = 0; i < prm.m; ++i) {
                const float dx = i - cx;
                const float dy = j - cy;
                const float dz = k - cz;
                const float r2 = dx * dx + dy * dy + dz * dz;
                const float bump = dT * std::exp(-r2 * inv2s2);
                /* max() keeps the ambient floor and makes repeated calls
                 * (multiple drops) compose sensibly.
                 */
                float& cell = T[idx3D(i, j, k, prm.m, prm.n)];
                if (cell < prm.ambientT + bump) cell = prm.ambientT + bump;
            }
        }
    }
}

bool writeGridToCSV(const std::vector<float>& T, const SimParams& prm,
                    const std::string& filename) {
    std::FILE* f = std::fopen(filename.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "writeGridToCSV: cannot open '%s'\n",
                     filename.c_str());
        return false;
    }

    std::fprintf(f, "%d,%d,%d\n", prm.m, prm.n, prm.p);
    std::fprintf(f, "%.9g,%.9g\n", prm.h, prm.alpha);

    /* One x-row per line, row-major (i fastest, then j, then k). */
    for (int k = 0; k < prm.p; ++k) {
        for (int j = 0; j < prm.n; ++j) {
            const float* row = &T[idx3D(0, j, k, prm.m, prm.n)];
            for (int i = 0; i < prm.m; ++i) {
                std::fprintf(f, i ? ",%.7g" : "%.7g", row[i]);
            }
            std::fputc('\n', f);
        }
    }

    std::fclose(f);
    return true;
}

bool readGridFromCSV(std::vector<float>& T, SimParams& prm,
                     const std::string& filename) {
    std::FILE* f = std::fopen(filename.c_str(), "r");
    if (!f) {
        std::fprintf(stderr, "readGridFromCSV: cannot open '%s'\n",
                     filename.c_str());
        return false;
    }

    if (std::fscanf(f, "%d,%d,%d", &prm.m, &prm.n, &prm.p) != 3 ||
        std::fscanf(f, "%f,%f", &prm.h, &prm.alpha) != 2) {
        std::fprintf(stderr, "readGridFromCSV: bad header in '%s'\n",
                     filename.c_str());
        std::fclose(f);
        return false;
    }

    const size_t total = static_cast<size_t>(prm.m) * prm.n * prm.p;
    T.resize(total);
    for (size_t idx = 0; idx < total; ++idx) {
        /* fscanf skips newlines/whitespace; a leading comma may or may not
         * precede each value, so consume it if present.
         */
        if (std::fscanf(f, "%f", &T[idx]) != 1) {
            int c = std::fgetc(f);           /* eat separator and retry */
            if ((c != ',' && c != '\n') || std::fscanf(f, "%f", &T[idx]) != 1) {
                std::fprintf(stderr,
                             "readGridFromCSV: bad value at index %zu\n", idx);
                std::fclose(f);
                return false;
            }
        }
        /* consume the trailing separator so the next fscanf starts clean */
        int c = std::fgetc(f);
        if (c != ',' && c != '\n' && c != EOF) std::ungetc(c, f);
    }

    std::fclose(f);
    return true;
}

float stableTimeStep(const SimParams& prm) {
    return prm.h * prm.h / (6.0f * prm.alpha);
}

float meanTemperature(const std::vector<float>& T) {
    if (T.empty()) return 0.0f;
    const double sum = std::accumulate(T.begin(), T.end(), 0.0);
    return static_cast<float>(sum / T.size());
}
