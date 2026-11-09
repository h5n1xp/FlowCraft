#include "FileDialog.h"
#include "FlowCraftIcon.h"

#import <AppKit/AppKit.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

std::optional<std::string> OpenSTLFileDialog() {
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    panel.canChooseFiles = YES;
    panel.canChooseDirectories = NO;
    panel.allowsMultipleSelection = NO;
    panel.title = @"Open STL Model";

    UTType* stlType = [UTType typeWithFilenameExtension:@"stl"];
    if (stlType) {
        panel.allowedContentTypes = @[ stlType ];
    }

    // The SDL window normally owns key status; a modal run loop here is
    // simplest and matches how Cocoa file panels are typically driven from
    // a non-document-based app.
    NSInteger result = [panel runModal];
    if (result != NSModalResponseOK) return std::nullopt;

    NSURL* url = panel.URL;
    if (!url) return std::nullopt;
    return std::string(url.path.UTF8String);
}

std::optional<std::string> OpenSessionFileDialog() {
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    panel.canChooseFiles = YES;
    panel.canChooseDirectories = NO;
    panel.allowsMultipleSelection = NO;
    panel.title = @"Open FlowCraft Session";

    UTType* sessionType = [UTType typeWithFilenameExtension:@"fcsession"];
    if (sessionType) {
        panel.allowedContentTypes = @[ sessionType ];
    }

    NSInteger result = [panel runModal];
    if (result != NSModalResponseOK) return std::nullopt;

    NSURL* url = panel.URL;
    if (!url) return std::nullopt;
    return std::string(url.path.UTF8String);
}

std::optional<std::string> SaveSessionFileDialog(const std::string& defaultName) {
    NSSavePanel* panel = [NSSavePanel savePanel];
    panel.title = @"Save FlowCraft Session";
    panel.nameFieldStringValue = [NSString stringWithUTF8String:defaultName.c_str()];

    UTType* sessionType = [UTType typeWithFilenameExtension:@"fcsession"];
    if (sessionType) {
        panel.allowedContentTypes = @[ sessionType ];
    }

    NSInteger result = [panel runModal];
    if (result != NSModalResponseOK) return std::nullopt;

    NSURL* url = panel.URL;
    if (!url) return std::nullopt;
    return std::string(url.path.UTF8String);
}

@interface FlowCraftAboutTarget : NSObject
- (void)showAboutPanel:(id)sender;
@end

@implementation FlowCraftAboutTarget
- (void)showAboutPanel:(id)sender {
    (void)sender;
    NSMutableParagraphStyle* style = [[NSMutableParagraphStyle alloc] init];
    style.alignment = NSTextAlignmentCenter;
    NSAttributedString* credits = [[NSAttributedString alloc]
        initWithString:@"by Matt Parsons"
            attributes:@{
                NSFontAttributeName : [NSFont boldSystemFontOfSize:13.0],
                NSForegroundColorAttributeName : [NSColor labelColor],
                NSParagraphStyleAttributeName : style
            }];

    [NSApp orderFrontStandardAboutPanelWithOptions:@{
        NSAboutPanelOptionApplicationName : @"FlowCraft",
        NSAboutPanelOptionApplicationVersion : @"0.1",
        NSAboutPanelOptionVersion : @"alpha",
        NSAboutPanelOptionCredits : credits,
        @"Copyright" : @"Copyright ©2026"
    }];
}
@end

void ConfigureMacOSAboutMenu() {
    static FlowCraftAboutTarget* aboutTarget = nil;
    if (!aboutTarget) {
        aboutTarget = [[FlowCraftAboutTarget alloc] init];
    }

    NSMenu* mainMenu = [NSApp mainMenu];
    if (!mainMenu || mainMenu.numberOfItems == 0) return;

    NSMenu* appMenu = [mainMenu itemAtIndex:0].submenu;
    if (!appMenu || appMenu.numberOfItems == 0) return;

    NSMenuItem* aboutItem = [appMenu itemAtIndex:0];
    aboutItem.title = @"About FlowCraft";
    aboutItem.target = aboutTarget;
    aboutItem.action = @selector(showAboutPanel:);
}

void ConfigureMacOSAppIcon() {
    @autoreleasepool {
        NSImage* iconImage = nil;

        // 1. Try loading FlowCraft.icns from application bundle resources
        NSString* icnsPath = [[NSBundle mainBundle] pathForResource:@"FlowCraft" ofType:@"icns"];
        if (icnsPath && [[NSFileManager defaultManager] fileExistsAtPath:icnsPath]) {
            iconImage = [[NSImage alloc] initWithContentsOfFile:icnsPath];
        }

        // 2. Try loading from adjacent or local FlowCraft/Assets/FlowCraft.icns
        if (!iconImage) {
            NSString* execPath = [[NSBundle mainBundle] executablePath];
            if (execPath) {
                NSString* execDir = [execPath stringByDeletingLastPathComponent];
                NSArray<NSString*>* searchCandidates = @[
                    [execDir stringByAppendingPathComponent:@"FlowCraft.icns"],
                    [execDir stringByAppendingPathComponent:@"FlowCraft/Assets/FlowCraft.icns"],
                    [execDir stringByAppendingPathComponent:@"Assets/FlowCraft.icns"]
                ];
                for (NSString* candidate in searchCandidates) {
                    if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) {
                        iconImage = [[NSImage alloc] initWithContentsOfFile:candidate];
                        if (iconImage) break;
                    }
                }
            }
        }

        // 3. Fallback to embedded high-resolution 128x128 RGBA icon bitmap
        if (!iconImage) {
            NSBitmapImageRep* rep = [[NSBitmapImageRep alloc]
                initWithBitmapDataPlanes:NULL
                              pixelsWide:kFlowCraftIconWidth
                              pixelsHigh:kFlowCraftIconHeight
                           bitsPerSample:8
                         samplesPerPixel:4
                                hasAlpha:YES
                                isPlanar:NO
                          colorSpaceName:NSDeviceRGBColorSpace
                            bitmapFormat:0
                             bytesPerRow:kFlowCraftIconWidth * 4
                            bitsPerPixel:32];
            if (rep) {
                std::memcpy([rep bitmapData], GetFlowCraftIconRgba(), kFlowCraftIconWidth * kFlowCraftIconHeight * 4);
                iconImage = [[NSImage alloc] initWithSize:NSMakeSize(kFlowCraftIconWidth, kFlowCraftIconHeight)];
                [iconImage addRepresentation:rep];
            }
        }

        if (iconImage) {
            [NSApp setApplicationIconImage:iconImage];
        }
    }
}
