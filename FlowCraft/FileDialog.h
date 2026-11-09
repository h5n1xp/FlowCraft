#pragma once

#include <optional>
#include <string>

// Native macOS "Open" panel, filtered to .stl files. Returns the chosen
// path, or nullopt if the user cancelled.
std::optional<std::string> OpenSTLFileDialog();

// Native macOS "Open"/"Save" panels, filtered to .fcsession files (see
// SessionFile.h). `defaultName` seeds the Save panel's filename field
// (the extension is added automatically if omitted); both return nullopt
// if the user cancelled.
std::optional<std::string> OpenSessionFileDialog();
std::optional<std::string> SaveSessionFileDialog(const std::string& defaultName);

// Configures the native macOS application menu's "About FlowCraft" item
// and panel to display "FlowCraft v0.1 by Matt Parsons".
void ConfigureMacOSAboutMenu();

// Configures the native macOS application Dock and App Switcher icon.
void ConfigureMacOSAppIcon();
