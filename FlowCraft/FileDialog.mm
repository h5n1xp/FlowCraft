#include "FileDialog.h"

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
