#include "mpc_offline.h"
#include "discretizer.h"
#include "condensing.h"
#include "blas_dispatch.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

PrecomputedWindow* mpc_precompute_all(const RefNode* ref_path, int n_path,
                                      const ModelParams& params,
                                      const MPCConfig& config,
                                      int& n_windows_out)
{
    int n_windows = n_path - config.N;
    if (n_windows <= 0) {
        n_windows_out = 0;
        return nullptr;
    }

    const int N = config.N;
    PrecomputedWindow* windows = new PrecomputedWindow[n_windows];

    // Precompute all interval discretizations once (instead of redundantly per-window)
    const int n_intervals = n_path - 1;
    double* all_A = new double[n_intervals * NX * NX];
    double* all_B = new double[n_intervals * NX * NU];
    for (int i = 0; i < n_intervals; ++i) {
        exact_discretize(ref_path[i], ref_path[i + 1], params,
                         all_A + i * NX * NX,
                         all_B + i * NX * NU);
    }

    // Temp arrays for per-window condensing
    double* A_list = new double[N * NX * NX];
    double* B_list = new double[N * NX * NU];
    double* x_ref_consistent = new double[(N + 1) * NX];
    double* u_ref_stacked = new double[N * NU];
    double temp_Ax[NX];
    double temp_Bu[NX];

    for (int start = 0; start < n_windows; ++start) {
        // (a) Copy pre-discretized matrices for this window's horizon
        for (int k = 0; k < N; ++k) {
            int idx = start + k;
            std::memcpy(A_list + k * NX * NX, all_A + idx * NX * NX, NX * NX * sizeof(double));
            std::memcpy(B_list + k * NX * NU, all_B + idx * NX * NU, NX * NU * sizeof(double));
        }

        // (b) Recompute dynamically-consistent reference (affine offset = 0)
        std::memcpy(x_ref_consistent, ref_path[start].x_ref, NX * sizeof(double));

        for (int k = 0; k < N; ++k) {
            const double* A_k = A_list + k * NX * NX;
            const double* B_k = B_list + k * NX * NU;
            const double* x_k = x_ref_consistent + k * NX;
            double* x_next = x_ref_consistent + (k + 1) * NX;

            // x_next = A_k * x_k
            mpc_linalg::gemv(NX, NX, A_k, x_k, temp_Ax);

            // temp_Bu = B_k * u_ref_k
            mpc_linalg::gemv(NX, NU, B_k, ref_path[start + k].u_ref, temp_Bu);

            // x_next = temp_Ax + temp_Bu
            for (int i = 0; i < NX; ++i) {
                x_next[i] = temp_Ax[i] + temp_Bu[i];
            }
        }

        // (c) Stack u_ref
        for (int k = 0; k < N; ++k) {
            std::memcpy(u_ref_stacked + k * NU, ref_path[start + k].u_ref,
                        NU * sizeof(double));
        }

        // (d) Condense window
        condense_window(A_list, B_list, x_ref_consistent, u_ref_stacked,
                        config, windows[start]);

        // (e) Copy x_ref_0
        std::memcpy(windows[start].x_ref_0, x_ref_consistent, NX * sizeof(double));
    }

    // Clean up temp arrays
    delete[] all_A;
    delete[] all_B;
    delete[] A_list;
    delete[] B_list;
    delete[] x_ref_consistent;
    delete[] u_ref_stacked;

    n_windows_out = n_windows;
    return windows;
}

int mpc_save_windows(const char* filename, const PrecomputedWindow* windows,
                     int n_windows, const MPCConfig& config)
{
    FILE* fp = std::fopen(filename, "wb");
    if (!fp) return -1;

    // Write header
    MPCFileHeader header{};
    header.magic = MPC_FILE_MAGIC;
    header.version = MPC_FILE_VERSION;
    header.n_windows = static_cast<uint32_t>(n_windows);
    header.N = static_cast<uint32_t>(config.N);
    header.nx = static_cast<uint32_t>(NX);
    header.nu = static_cast<uint32_t>(NU);
    header.u_min = config.u_min;
    header.u_max = config.u_max;
    header.dt = config.dt;

    if (std::fwrite(&header, sizeof(header), 1, fp) != 1) {
        std::fclose(fp);
        return -1;
    }

    // Write each window
    for (int i = 0; i < n_windows; ++i) {
        int n_vars = windows[i].n_vars;

        // H: n_vars x n_vars
        if (std::fwrite(windows[i].H, sizeof(double), n_vars * n_vars, fp)
            != static_cast<size_t>(n_vars * n_vars)) {
            std::fclose(fp);
            return -1;
        }

        // L: n_vars x n_vars
        if (std::fwrite(windows[i].L, sizeof(double), n_vars * n_vars, fp)
            != static_cast<size_t>(n_vars * n_vars)) {
            std::fclose(fp);
            return -1;
        }

        // F: n_vars x NX
        if (std::fwrite(windows[i].F, sizeof(double), n_vars * NX, fp)
            != static_cast<size_t>(n_vars * NX)) {
            std::fclose(fp);
            return -1;
        }

        // f_const: n_vars
        if (std::fwrite(windows[i].f_const, sizeof(double), n_vars, fp)
            != static_cast<size_t>(n_vars)) {
            std::fclose(fp);
            return -1;
        }

        // x_ref_0: NX
        if (std::fwrite(windows[i].x_ref_0, sizeof(double), NX, fp)
            != static_cast<size_t>(NX)) {
            std::fclose(fp);
            return -1;
        }

        // lambda_max: 1 double (v2+)
        if (std::fwrite(&windows[i].lambda_max, sizeof(double), 1, fp) != 1) {
            std::fclose(fp);
            return -1;
        }
    }

    std::fclose(fp);
    return 0;
}

PrecomputedWindow* mpc_load_windows(const char* filename, int& n_windows_out,
                                     MPCConfig& config_out)
{
    FILE* fp = std::fopen(filename, "rb");
    if (!fp) return nullptr;

    // Read and validate header
    MPCFileHeader header;
    if (std::fread(&header, sizeof(header), 1, fp) != 1) {
        std::fclose(fp);
        return nullptr;
    }

    if (header.magic != MPC_FILE_MAGIC || header.version != MPC_FILE_VERSION) {
        std::fclose(fp);
        return nullptr;
    }

    // Populate config from header
    config_out.N = static_cast<int>(header.N);
    config_out.u_min = header.u_min;
    config_out.u_max = header.u_max;
    config_out.dt = header.dt;

    int n_windows = static_cast<int>(header.n_windows);
    PrecomputedWindow* windows = new PrecomputedWindow[n_windows];

    for (int i = 0; i < n_windows; ++i) {
        windows[i].N = static_cast<int>(header.N);
        windows[i].n_vars = static_cast<int>(header.N * header.nu);
        int n_vars = windows[i].n_vars;

        // H: n_vars x n_vars
        if (std::fread(windows[i].H, sizeof(double), n_vars * n_vars, fp)
            != static_cast<size_t>(n_vars * n_vars)) {
            delete[] windows;
            std::fclose(fp);
            return nullptr;
        }

        // L: n_vars x n_vars
        if (std::fread(windows[i].L, sizeof(double), n_vars * n_vars, fp)
            != static_cast<size_t>(n_vars * n_vars)) {
            delete[] windows;
            std::fclose(fp);
            return nullptr;
        }

        // F: n_vars x NX
        if (std::fread(windows[i].F, sizeof(double), n_vars * NX, fp)
            != static_cast<size_t>(n_vars * NX)) {
            delete[] windows;
            std::fclose(fp);
            return nullptr;
        }

        // f_const: n_vars
        if (std::fread(windows[i].f_const, sizeof(double), n_vars, fp)
            != static_cast<size_t>(n_vars)) {
            delete[] windows;
            std::fclose(fp);
            return nullptr;
        }

        // x_ref_0: NX
        if (std::fread(windows[i].x_ref_0, sizeof(double), NX, fp)
            != static_cast<size_t>(NX)) {
            delete[] windows;
            std::fclose(fp);
            return nullptr;
        }

        // lambda_max: 1 double (v2+)
        if (std::fread(&windows[i].lambda_max, sizeof(double), 1, fp) != 1) {
            delete[] windows;
            std::fclose(fp);
            return nullptr;
        }
    }

    n_windows_out = n_windows;
    std::fclose(fp);
    return windows;
}
