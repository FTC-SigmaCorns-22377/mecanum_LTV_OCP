#include <jni.h>
#include <string>

#include "mecanum_ltv.h"

// ---------------------------------------------------------------------------
// JNI helpers
// ---------------------------------------------------------------------------
namespace {

void throw_exception(JNIEnv* env, const char* class_name, const std::string& msg) {
    if (env->ExceptionCheck()) return;
    jclass cls = env->FindClass(class_name);
    if (!cls) return;
    env->ThrowNew(cls, msg.c_str());
    env->DeleteLocalRef(cls);
}

void throw_illegal_argument(JNIEnv* env, const std::string& msg) {
    throw_exception(env, "java/lang/IllegalArgumentException", msg);
}

void throw_illegal_state(JNIEnv* env, const std::string& msg) {
    throw_exception(env, "java/lang/IllegalStateException", msg);
}

void throw_runtime(JNIEnv* env, const std::string& msg) {
    throw_exception(env, "java/lang/RuntimeException", msg);
}

MecanumLTV* from_handle(jlong handle) {
    return reinterpret_cast<MecanumLTV*>(handle);
}

bool check_handle(JNIEnv* env, jlong handle) {
    if (handle == 0) {
        throw_illegal_state(env, "MecanumLTV handle is null (destroyed or never created)");
        return false;
    }
    return true;
}

bool check_array(JNIEnv* env, jdoubleArray arr, jsize expected, const char* name) {
    if (!arr) {
        throw_illegal_argument(env, std::string(name) + " is null");
        return false;
    }
    jsize len = env->GetArrayLength(arr);
    if (len != expected) {
        throw_illegal_argument(env, std::string(name) + ": expected length " +
                               std::to_string(expected) + ", got " + std::to_string(len));
        return false;
    }
    return true;
}

// RAII wrapper for JNI double array access
class ScopedDoubleArray {
public:
    ScopedDoubleArray(JNIEnv* env, jdoubleArray arr, jint release_mode)
        : env_(env), arr_(arr), mode_(release_mode), ptr_(nullptr) {
        if (!env_->ExceptionCheck() && arr_)
            ptr_ = env_->GetDoubleArrayElements(arr_, nullptr);
    }
    ~ScopedDoubleArray() {
        if (ptr_) env_->ReleaseDoubleArrayElements(arr_, ptr_, mode_);
    }
    double* data() const { return ptr_; }
    bool valid() const { return ptr_ != nullptr; }
private:
    JNIEnv* env_;
    jdoubleArray arr_;
    jint mode_;
    double* ptr_;
};

} // namespace

// ---------------------------------------------------------------------------
// JNI exports — Java class: sigmacorns.control.ltv.MecanumLTVBridge
// ---------------------------------------------------------------------------

extern "C" {

// long nativeCreate()
JNIEXPORT jlong JNICALL
Java_sigmacorns_control_ltv_MecanumLTVBridge_nativeCreate(JNIEnv* env, jclass) {
    try {
        auto* ctrl = new MecanumLTV();
        return reinterpret_cast<jlong>(ctrl);
    } catch (const std::exception& e) {
        throw_runtime(env, std::string("Failed to create MecanumLTV: ") + e.what());
    } catch (...) {
        throw_runtime(env, "Failed to create MecanumLTV: unknown error");
    }
    return 0;
}

// void nativeDestroy(long handle)
JNIEXPORT void JNICALL
Java_sigmacorns_control_ltv_MecanumLTVBridge_nativeDestroy(JNIEnv* env, jclass, jlong handle) {
    if (!check_handle(env, handle)) return;
    delete from_handle(handle);
}

// void nativeSetModelParams(long handle, double mass, double inertia,
//     double dampingLinear, double dampingAngular, double wheelRadius,
//     double lx, double ly, double stallTorque, double freeSpeed)
JNIEXPORT void JNICALL
Java_sigmacorns_control_ltv_MecanumLTVBridge_nativeSetModelParams(
    JNIEnv* env, jclass, jlong handle,
    jdouble mass, jdouble inertia,
    jdouble dampingLinear, jdouble dampingAngular,
    jdouble wheelRadius, jdouble lx, jdouble ly,
    jdouble stallTorque, jdouble freeSpeed)
{
    if (!check_handle(env, handle)) return;

    ModelParams p{};
    p.mass = mass;
    p.inertia = inertia;
    p.damping_linear = dampingLinear;
    p.damping_angular = dampingAngular;
    p.wheel_radius = wheelRadius;
    p.lx = lx;
    p.ly = ly;
    p.stall_torque = stallTorque;
    p.free_speed = freeSpeed;

    from_handle(handle)->setModelParams(p);
}

// void nativeSetConfig(long handle, int N, double[] qDiag, double[] rDiag,
//                      double[] qfDiag, double uMin, double uMax)
JNIEXPORT void JNICALL
Java_sigmacorns_control_ltv_MecanumLTVBridge_nativeSetConfig(
    JNIEnv* env, jclass, jlong handle,
    jint N, jdoubleArray qDiag, jdoubleArray rDiag, jdoubleArray qfDiag,
    jdouble uMin, jdouble uMax)
{
    if (!check_handle(env, handle)) return;
    if (!check_array(env, qDiag, NX, "qDiag")) return;
    if (!check_array(env, rDiag, NU, "rDiag")) return;
    if (!check_array(env, qfDiag, NX, "qfDiag")) return;

    if (N < 1 || N > N_MAX) {
        throw_illegal_argument(env, "N must be in [1, " + std::to_string(N_MAX) + "]");
        return;
    }

    ScopedDoubleArray q(env, qDiag, JNI_ABORT);
    ScopedDoubleArray r(env, rDiag, JNI_ABORT);
    ScopedDoubleArray qf(env, qfDiag, JNI_ABORT);
    if (!q.valid() || !r.valid() || !qf.valid()) return;

    MPCConfig cfg{};
    cfg.N = N;
    cfg.u_min = uMin;
    cfg.u_max = uMax;
    // dt will be set in loadTrajectory

    // Fill diagonal matrices (column-major, zero off-diag already from zero-init)
    for (int i = 0; i < NX; ++i) {
        cfg.Q[i + NX * i] = q.data()[i];
        cfg.Qf[i + NX * i] = qf.data()[i];
    }
    for (int i = 0; i < NU; ++i) {
        cfg.R[i + NU * i] = r.data()[i];
    }

    from_handle(handle)->setConfig(cfg);
}

// int nativeLoadWindows(long handle, String filepath)
JNIEXPORT jint JNICALL
Java_sigmacorns_control_ltv_MecanumLTVBridge_nativeLoadWindows(
    JNIEnv* env, jclass, jlong handle, jstring filepath)
{
    if (!check_handle(env, handle)) return 0;
    if (!filepath) {
        throw_illegal_argument(env, "filepath is null");
        return 0;
    }

    const char* path_cstr = env->GetStringUTFChars(filepath, nullptr);
    if (!path_cstr) return 0;

    int result = 0;
    try {
        result = from_handle(handle)->loadWindows(path_cstr);
    } catch (const std::exception& e) {
        env->ReleaseStringUTFChars(filepath, path_cstr);
        throw_runtime(env, std::string("loadWindows failed: ") + e.what());
        return 0;
    } catch (...) {
        env->ReleaseStringUTFChars(filepath, path_cstr);
        throw_runtime(env, "loadWindows failed: unknown error");
        return 0;
    }
    env->ReleaseStringUTFChars(filepath, path_cstr);
    return result;
}

// double nativeDt(long handle)
JNIEXPORT jdouble JNICALL
Java_sigmacorns_control_ltv_MecanumLTVBridge_nativeDt(JNIEnv* env, jclass, jlong handle) {
    if (!check_handle(env, handle)) return 0.0;
    return from_handle(handle)->dt();
}

// int nativeLoadTrajectory(long handle, double[] samples, int nSamples, double dt)
//   samples is flat [t, px, py, theta, vx, vy, omega, ...] with 7 doubles per sample
JNIEXPORT jint JNICALL
Java_sigmacorns_control_ltv_MecanumLTVBridge_nativeLoadTrajectory(
    JNIEnv* env, jclass, jlong handle,
    jdoubleArray samples, jint nSamples, jdouble dt)
{
    if (!check_handle(env, handle)) return 0;

    jsize expected_len = nSamples * 7;
    if (!check_array(env, samples, expected_len, "samples")) return 0;

    ScopedDoubleArray data(env, samples, JNI_ABORT);
    if (!data.valid()) return 0;

    try {
        return from_handle(handle)->loadTrajectory(data.data(), nSamples, dt);
    } catch (const std::exception& e) {
        throw_runtime(env, std::string("loadTrajectory failed: ") + e.what());
    } catch (...) {
        throw_runtime(env, "loadTrajectory failed: unknown error");
    }
    return 0;
}

// int nativeSolve(long handle, int windowIdx, double[] x0, double[] uOut)
//   x0 is length 6, uOut is length N*4
JNIEXPORT jint JNICALL
Java_sigmacorns_control_ltv_MecanumLTVBridge_nativeSolve(
    JNIEnv* env, jclass, jlong handle,
    jint windowIdx, jdoubleArray x0, jdoubleArray uOut)
{
    if (!check_handle(env, handle)) return -1;
    if (!check_array(env, x0, NX, "x0")) return -1;

    auto* ctrl = from_handle(handle);
    jsize expected_u = static_cast<jsize>(ctrl->numVars());
    if (!check_array(env, uOut, expected_u, "uOut")) return -1;

    ScopedDoubleArray x0_arr(env, x0, JNI_ABORT);
    ScopedDoubleArray u_arr(env, uOut, 0); // writeback
    if (!x0_arr.valid() || !u_arr.valid()) return -1;

    try {
        return ctrl->solve(windowIdx, x0_arr.data(), u_arr.data());
    } catch (const std::exception& e) {
        throw_runtime(env, std::string("solve failed: ") + e.what());
    } catch (...) {
        throw_runtime(env, "solve failed: unknown error");
    }
    return -1;
}

// int nativeNumWindows(long handle)
JNIEXPORT jint JNICALL
Java_sigmacorns_control_ltv_MecanumLTVBridge_nativeNumWindows(JNIEnv* env, jclass, jlong handle) {
    if (!check_handle(env, handle)) return 0;
    return from_handle(handle)->numWindows();
}

// int nativeHorizonLength(long handle)
JNIEXPORT jint JNICALL
Java_sigmacorns_control_ltv_MecanumLTVBridge_nativeHorizonLength(JNIEnv* env, jclass, jlong handle) {
    if (!check_handle(env, handle)) return 0;
    return from_handle(handle)->horizonLength();
}

// int nativeNumVars(long handle)
JNIEXPORT jint JNICALL
Java_sigmacorns_control_ltv_MecanumLTVBridge_nativeNumVars(JNIEnv* env, jclass, jlong handle) {
    if (!check_handle(env, handle)) return 0;
    return from_handle(handle)->numVars();
}

} // extern "C"
