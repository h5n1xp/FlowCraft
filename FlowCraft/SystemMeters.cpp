#include "SystemMeters.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/resource.h>
#include <sys/time.h>
#endif

#if defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#endif

namespace SystemMeters {

double ProcessCpuSeconds() {
#if defined(_WIN32)
    FILETIME creation, exit, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) return -1.0;
    auto toSeconds = [](const FILETIME& ft) {
        ULARGE_INTEGER v;
        v.LowPart = ft.dwLowDateTime;
        v.HighPart = ft.dwHighDateTime;
        return static_cast<double>(v.QuadPart) * 1e-7;  // FILETIME counts 100ns ticks
    };
    return toSeconds(kernel) + toSeconds(user);
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) return -1.0;
    auto toSeconds = [](const struct timeval& tv) {
        return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) * 1e-6;
    };
    return toSeconds(usage.ru_utime) + toSeconds(usage.ru_stime);
#endif
}

float GpuUtilization() {
#if defined(__APPLE__)
    // Every GPU driver on macOS (Apple Silicon's AGX and the Intel/AMD
    // ones alike) publishes a "PerformanceStatistics" dictionary on its
    // IOAccelerator registry entry — the same numbers Activity Monitor's
    // GPU History reads. With more than one GPU, the busiest is reported.
    io_iterator_t iterator = 0;
    // 0 == the default main port; spelled as a literal because the named
    // constant was renamed (kIOMasterPortDefault -> kIOMainPortDefault) in
    // macOS 12 and this compiles against either SDK.
    if (IOServiceGetMatchingServices(0, IOServiceMatching("IOAccelerator"), &iterator) != KERN_SUCCESS) return -1.0f;

    float best = -1.0f;
    for (io_object_t entry = IOIteratorNext(iterator); entry != 0; entry = IOIteratorNext(iterator)) {
        CFTypeRef stats =
            IORegistryEntryCreateCFProperty(entry, CFSTR("PerformanceStatistics"), kCFAllocatorDefault, 0);
        if (stats) {
            if (CFGetTypeID(stats) == CFDictionaryGetTypeID()) {
                CFTypeRef value =
                    CFDictionaryGetValue(static_cast<CFDictionaryRef>(stats), CFSTR("Device Utilization %"));
                double percent = 0.0;
                if (value && CFGetTypeID(value) == CFNumberGetTypeID() &&
                    CFNumberGetValue(static_cast<CFNumberRef>(value), kCFNumberDoubleType, &percent)) {
                    float fraction = static_cast<float>(percent / 100.0);
                    if (fraction > best) best = fraction;
                }
            }
            CFRelease(stats);
        }
        IOObjectRelease(entry);
    }
    IOObjectRelease(iterator);
    if (best > 1.0f) best = 1.0f;
    return best;
#else
    return -1.0f;
#endif
}

}  // namespace SystemMeters
