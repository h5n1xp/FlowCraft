#pragma once

// Raw OS load queries behind the Processing window's CPU/GPU meters (see
// UpdatePerfMeters in main.cpp, which turns these into rates). Kept out of
// main.cpp because each one is a block of platform-specific system API.
namespace SystemMeters {

// Total CPU time this process has consumed so far, in seconds, summed
// across all of its threads (user + kernel). Divide the difference between
// two samples by the wall time between them to get "cores in use".
// Returns a negative value if the platform query fails.
double ProcessCpuSeconds();

// Current GPU utilisation, 0..1, as the OS reports it — for the whole
// device, not just this process, so other apps' GPU work counts too.
// Returns a negative value where there is no supported way to ask:
// currently implemented on macOS only (IOKit's accelerator statistics);
// Windows and Linux have no vendor-neutral equivalent that's cheap to
// query, so the meter is simply not shown there.
float GpuUtilization();

}  // namespace SystemMeters
