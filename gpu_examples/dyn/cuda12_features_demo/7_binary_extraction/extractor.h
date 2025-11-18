// extractor.h
// Binary extraction and JIT rewriting framework

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvJitLink.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <regex>
#include "ptx_modifier.h"

// Error checking
#define CHECK_CU(call) do { \
    CUresult res = call; \
    if (res != CUDA_SUCCESS) { \
        const char* name = nullptr; \
        cuGetErrorName(res, &name); \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
                __FILE__, __LINE__, name ? name : "unknown"); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_NVJITLINK(call) do { \
    nvJitLinkResult res = call; \
    if (res != NVJITLINK_SUCCESS) { \
        fprintf(stderr, "nvJitLink Error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

class BinaryExtractor {
private:
    std::string binaryPath;
    std::vector<std::string> extractedFiles;

public:
    BinaryExtractor(const char* path) : binaryPath(path) {
        CHECK_CU(cuInit(0));
    }

    // Extract using cuobjdump
    bool extractAllCubins() {
        printf("\n=== Extracting Kernels from Binary ===\n");
        printf("Binary: %s\n", binaryPath.c_str());

        // List available ELF files
        std::string listCmd = "cuobjdump -lelf " + binaryPath + " 2>&1";
        FILE* pipe = popen(listCmd.c_str(), "r");
        if (!pipe) {
            fprintf(stderr, "Failed to run cuobjdump\n");
            return false;
        }

        char buffer[256];
        std::vector<std::string> elfFiles;
        while (fgets(buffer, sizeof(buffer), pipe)) {
            std::string line(buffer);
            if (line.find("ELF file") != std::string::npos) {
                // Extract filename
                size_t colonPos = line.find(':');
                if (colonPos != std::string::npos) {
                    std::string filename = line.substr(colonPos + 2);
                    // Remove newline
                    filename.erase(filename.find_last_not_of(" \n\r\t") + 1);
                    elfFiles.push_back(filename);
                    printf("Found: %s\n", filename.c_str());
                }
            }
        }
        pclose(pipe);

        if (elfFiles.empty()) {
            printf("No ELF files found in binary\n");
            return false;
        }

        // Extract each ELF file
        for (const auto& elf : elfFiles) {
            std::string extractCmd = "cuobjdump -xelf \"" + elf + "\" " + binaryPath;
            printf("Extracting: %s\n", elf.c_str());

            int ret = system(extractCmd.c_str());
            if (ret == 0) {
                extractedFiles.push_back(elf);
                printf("✓ Extracted: %s\n", elf.c_str());
            } else {
                printf("✗ Failed to extract: %s\n", elf.c_str());
            }
        }

        printf("✓ Extracted %zu kernel file(s)\n", extractedFiles.size());
        return !extractedFiles.empty();
    }

    const std::vector<std::string>& getExtractedFiles() const {
        return extractedFiles;
    }

    // Get symbols from extracted cubin
    void listSymbols(const char* cubinFile) {
        printf("\n=== Symbols in %s ===\n", cubinFile);
        std::string cmd = "cuobjdump -symbols " + std::string(cubinFile);
        system(cmd.c_str());
    }

    // Extract PTX and convert entry points to device functions
    bool extractAndConvertPTX(const char* outputPTX) {
        printf("\n=== Extracting PTX from Binary ===\n");
        printf("Binary: %s\n", binaryPath.c_str());
        printf("Output: %s\n", outputPTX);

        if (!PTXModifier::extractAndModifyPTX(binaryPath.c_str(), outputPTX)) {
            fprintf(stderr, "Failed to extract and modify PTX\n");
            return false;
        }

        printf("✓ Extracted PTX and converted .entry → .func\n");
        printf("  Original kernels are now callable from device code!\n");

        return true;
    }
};

class JITRewriter {
private:
    std::vector<char> extractedPTX;  // Changed from CUBIN to PTX
    std::vector<char> policyPTX;
    std::vector<char> linkedCubin;
    CUmodule module;
    bool loaded;

    std::vector<char> readFile(const char* filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            fprintf(stderr, "Failed to open %s\n", filename);
            return {};
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            fprintf(stderr, "Failed to read %s\n", filename);
            return {};
        }

        return buffer;
    }

public:
    JITRewriter() : module(nullptr), loaded(false) {}

    ~JITRewriter() {
        if (module) {
            cuModuleUnload(module);
        }
    }

    bool loadExtractedPTX(const char* ptxFile) {
        printf("\n=== Loading Extracted PTX (Modified) ===\n");
        printf("File: %s\n", ptxFile);

        extractedPTX = readFile(ptxFile);
        if (extractedPTX.empty()) {
            return false;
        }

        // Ensure null-termination for PTX
        if (extractedPTX.back() != '\0') {
            extractedPTX.push_back('\0');
        }

        printf("✓ Loaded extracted PTX (%zu bytes)\n", extractedPTX.size());
        printf("  (Converted from .entry to .func for device-level linking)\n");
        return true;
    }

    bool loadPolicy(const char* policyFile) {
        printf("\n=== Loading Policy ===\n");
        printf("File: %s\n", policyFile);

        policyPTX = readFile(policyFile);
        if (policyPTX.empty()) {
            return false;
        }

        // Ensure null-termination for PTX
        if (policyPTX.back() != '\0') {
            policyPTX.push_back('\0');
        }

        printf("✓ Loaded policy PTX (%zu bytes)\n", policyPTX.size());
        return true;
    }

    bool linkAndLoad(int major, int minor, bool useNvJitLink = true) {
        if (policyPTX.empty()) {
            fprintf(stderr, "Policy not loaded!\n");
            return false;
        }

        if (!useNvJitLink || extractedPTX.empty()) {
            // Fallback: Load standalone policy
            printf("\n=== Loading Policy Module (Standalone) ===\n");
            CHECK_CU(cuModuleLoadData(&module, policyPTX.data()));
            printf("✓ Loaded policy module\n");
            loaded = true;
            return true;
        }

        // Real nvJitLink approach with extracted PTX
        printf("\n=== Linking with nvJitLink ===\n");
        printf("Linking extracted PTX (modified) + policy PTX...\n");

        nvJitLinkHandle handle;

        char archOpt[32];
        snprintf(archOpt, sizeof(archOpt), "-arch=sm_%d%d", major, minor);

        const char* options[] = {archOpt, "-O3"};

        CHECK_NVJITLINK(nvJitLinkCreate(&handle, 2, options));
        printf("✓ Created nvJitLink handle\n");
        printf("  Options: %s -O3\n", archOpt);

        // Add extracted PTX (now with .func instead of .entry)
        nvJitLinkResult addResult = nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX,
                                         extractedPTX.data(), extractedPTX.size(),
                                         "extracted_kernels");
        if (addResult != NVJITLINK_SUCCESS) {
            size_t logSize;
            nvJitLinkGetErrorLogSize(handle, &logSize);
            if (logSize > 0) {
                std::vector<char> log(logSize);
                nvJitLinkGetErrorLog(handle, log.data());
                fprintf(stderr, "Failed to add extracted PTX:\n%s\n", log.data());
            }
            nvJitLinkDestroy(&handle);
            return false;
        }
        printf("✓ Added extracted PTX (%zu bytes)\n", extractedPTX.size());

        // Add policy wrapper PTX
        CHECK_NVJITLINK(nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX,
                                         policyPTX.data(), policyPTX.size(),
                                         "policy_wrapper"));
        printf("✓ Added policy wrapper PTX (%zu bytes)\n", policyPTX.size());

        // Complete the link
        printf("Linking...\n");
        nvJitLinkResult linkResult = nvJitLinkComplete(handle);

        if (linkResult != NVJITLINK_SUCCESS) {
            size_t logSize;
            nvJitLinkGetErrorLogSize(handle, &logSize);
            if (logSize > 0) {
                std::vector<char> log(logSize);
                nvJitLinkGetErrorLog(handle, log.data());
                fprintf(stderr, "Link error:\n%s\n", log.data());
            }
            nvJitLinkDestroy(&handle);
            return false;
        }

        printf("✓ Linking completed successfully!\n");

        // Get linked CUBIN
        size_t cubinSize;
        CHECK_NVJITLINK(nvJitLinkGetLinkedCubinSize(handle, &cubinSize));
        linkedCubin.resize(cubinSize);
        CHECK_NVJITLINK(nvJitLinkGetLinkedCubin(handle, linkedCubin.data()));

        printf("✓ Generated linked CUBIN (%zu bytes)\n", cubinSize);
        printf("  Extracted: %zu bytes + Policy: %zu bytes → Linked: %zu bytes\n",
               extractedPTX.size(), policyPTX.size(), cubinSize);

        nvJitLinkDestroy(&handle);

        // Load the linked module
        CHECK_CU(cuModuleLoadData(&module, linkedCubin.data()));
        printf("✓ Loaded linked module\n");

        loaded = true;
        return true;
    }

    bool getKernel(CUfunction* kernel, const char* name) {
        if (!loaded) {
            fprintf(stderr, "Module not loaded!\n");
            return false;
        }

        CUresult res = cuModuleGetFunction(kernel, module, name);
        if (res != CUDA_SUCCESS) {
            const char* errName = nullptr;
            cuGetErrorName(res, &errName);
            fprintf(stderr, "Failed to get kernel: %s (%s)\n", name, errName ? errName : "unknown");
            return false;
        }

        printf("✓ Got kernel: %s\n", name);
        return true;
    }
};
