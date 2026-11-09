// Portable (SDL3-only) implementation of FileDialog.h for platforms without
// the native Cocoa version (FileDialog.mm) — i.e. Windows and Linux.
//
// SDL_ShowOpenFileDialog/SDL_ShowSaveFileDialog are asynchronous (they
// invoke a callback later), while FileDialog.h's API — and every call site
// in main.cpp — is synchronous, like the Cocoa panels' runModal. These
// wrappers bridge the two by pumping the SDL event queue until the callback
// fires, which the SDL docs explicitly permit (the callback may be invoked
// from a later SDL_PumpEvents on some platforms).
#include "FileDialog.h"

// Compiled out on Apple platforms (even if a build system picks the file up
// automatically), where FileDialog.mm provides the native versions instead.
#if !defined(__APPLE__)

#include <SDL3/SDL.h>

#include <atomic>

namespace {

struct DialogResult {
    std::atomic<bool> done{false};
    std::optional<std::string> path;
};

void SDLCALL DialogCallback(void* userdata, const char* const* fileList, int /*filter*/) {
    auto* result = static_cast<DialogResult*>(userdata);
    if (fileList && fileList[0]) {
        result->path = std::string(fileList[0]);
    }
    result->done.store(true);
}

std::optional<std::string> WaitForDialog(DialogResult& result) {
    while (!result.done.load()) {
        SDL_PumpEvents();
        SDL_Delay(10);
    }
    return result.path;
}

SDL_Window* ParentWindow() {
    // Parent the dialog to whichever window has focus, if any, so it stacks
    // correctly; a null parent is also valid.
    return SDL_GetKeyboardFocus();
}

}  // namespace

std::optional<std::string> OpenSTLFileDialog() {
    static const SDL_DialogFileFilter filters[] = {{"STL models", "stl"}};
    DialogResult result;
    SDL_ShowOpenFileDialog(DialogCallback, &result, ParentWindow(), filters, 1, nullptr, false);
    return WaitForDialog(result);
}

std::optional<std::string> OpenSessionFileDialog() {
    static const SDL_DialogFileFilter filters[] = {{"FlowCraft sessions", "fcsession"}};
    DialogResult result;
    SDL_ShowOpenFileDialog(DialogCallback, &result, ParentWindow(), filters, 1, nullptr, false);
    return WaitForDialog(result);
}

std::optional<std::string> SaveSessionFileDialog(const std::string& defaultName) {
    static const SDL_DialogFileFilter filters[] = {{"FlowCraft sessions", "fcsession"}};
    DialogResult result;
    std::string defaultLocation = defaultName + ".fcsession";
    SDL_ShowSaveFileDialog(DialogCallback, &result, ParentWindow(), filters, 1, defaultLocation.c_str());
    std::optional<std::string> path = WaitForDialog(result);
    static const std::string kExt = ".fcsession";
    if (path && (path->size() < kExt.size() || path->compare(path->size() - kExt.size(), kExt.size(), kExt) != 0)) {
        *path += kExt;  // mirror the Cocoa panel, which appends the extension if omitted
    }
    return path;
}

// There is no application menu to configure outside macOS; the app version
// is still exposed through SDL_SetAppMetadata (see main()).
void ConfigureMacOSAboutMenu() {}
void ConfigureMacOSAppIcon() {}

#endif  // !__APPLE__
