// ptx_modifier.h
// PTX modification utilities for converting entry points to device functions

#pragma once

#include <string>
#include <regex>
#include <fstream>
#include <sstream>

class PTXModifier {
public:
    // Convert .entry to .func in PTX to make kernel callable from device code
    static std::string entryToFunc(const std::string& ptx) {
        std::string modified = ptx;

        // Replace .visible .entry with .visible .func
        modified = std::regex_replace(modified,
                                     std::regex("\\.visible\\s+\\.entry\\s+"),
                                     ".visible .func ");

        // Also handle .entry without .visible
        modified = std::regex_replace(modified,
                                     std::regex("\\.entry\\s+"),
                                     ".func ");

        // Remove .ptr and .align attributes from .param declarations
        // These are only valid for entry functions, not device functions
        // Pattern: .param .u64 .ptr .align N becomes .param .u64
        modified = std::regex_replace(modified,
                                     std::regex("\\.param\\s+(\\.u\\d+)\\s+\\.ptr\\s+\\.align\\s+\\d+"),
                                     ".param $1");

        return modified;
    }

    // Extract PTX from binary and modify it
    static bool extractAndModifyPTX(const char* binaryPath,
                                   const char* outputPath) {
        // Extract PTX using cuobjdump
        std::string cmd = std::string("cuobjdump -ptx ") + binaryPath +
                         " > " + outputPath + ".raw 2>&1";

        int ret = system(cmd.c_str());
        if (ret != 0) {
            fprintf(stderr, "Failed to extract PTX\n");
            return false;
        }

        // Read the raw output
        std::ifstream raw(std::string(outputPath) + ".raw");
        if (!raw.is_open()) {
            fprintf(stderr, "Failed to open raw PTX file\n");
            return false;
        }

        std::stringstream buffer;
        buffer << raw.rdbuf();
        std::string ptxContent = buffer.str();
        raw.close();

        // Find the actual PTX code (skip cuobjdump headers)
        size_t ptxStart = ptxContent.find(".version");
        if (ptxStart == std::string::npos) {
            fprintf(stderr, "No PTX code found in output\n");
            return false;
        }

        std::string ptxCode = ptxContent.substr(ptxStart);

        // Modify entry points to functions
        std::string modifiedPTX = entryToFunc(ptxCode);

        // Write modified PTX
        std::ofstream out(outputPath);
        if (!out.is_open()) {
            fprintf(stderr, "Failed to write modified PTX\n");
            return false;
        }

        out << modifiedPTX;
        out.close();

        // Clean up raw file
        std::remove((std::string(outputPath) + ".raw").c_str());

        return true;
    }

    // Show what was changed
    static void showModifications(const std::string& original,
                                 const std::string& modified) {
        printf("\n=== PTX Modifications ===\n");

        // Count replacements
        size_t entryCount = 0;
        size_t pos = 0;
        while ((pos = original.find(".entry", pos)) != std::string::npos) {
            entryCount++;
            pos += 6;
        }

        printf("Converted %zu .entry declarations to .func\n", entryCount);
        printf("This makes kernels callable from device code!\n");

        // Show a sample
        size_t samplePos = modified.find(".visible .func");
        if (samplePos != std::string::npos) {
            size_t lineStart = modified.rfind('\n', samplePos);
            size_t lineEnd = modified.find('\n', samplePos);
            if (lineStart != std::string::npos && lineEnd != std::string::npos) {
                std::string line = modified.substr(lineStart + 1, lineEnd - lineStart - 1);
                printf("Example: %s\n", line.c_str());
            }
        }
    }
};
