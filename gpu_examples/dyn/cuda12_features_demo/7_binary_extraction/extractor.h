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
};

class JITRewriter {
private:
    std::vector<char> extractedCubin;
    std::vector<char> policyPTX;
    std::vector<char> linkedCubin;
    CUlibrary library;
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
    JITRewriter() : library(nullptr), loaded(false) {}

    ~JITRewriter() {
        if (library) {
            cudaLibraryUnload(library);
        }
    }

    bool loadExtractedCubin(const char* cubinFile) {
        printf("\n=== Loading Extracted Kernel ===\n");
        printf("File: %s\n", cubinFile);

        extractedCubin = readFile(cubinFile);
        if (extractedCubin.empty()) {
            return false;
        }

        printf("✓ Loaded extracted CUBIN (%zu bytes)\n", extractedCubin.size());
        return true;
    }

    bool loadPolicy(const char* policyFile) {
        printf("\n=== Loading Policy ===\n");
        printf("File: %s\n", policyFile);

        policyPTX = readFile(policyFile);
        if (policyPTX.empty()) {
            return false;
        }

        printf("✓ Loaded policy PTX (%zu bytes)\n", policyPTX.size());
        return true;
    }

    bool linkAndLoad(int major, int minor) {
        printf("\n=== Linking with nvJitLink ===\n");

        // For this demo, we'll directly load the policy PTX
        // In a real scenario, you would link extracted CUBIN with policy PTX

        // Option 1: Load policy directly (simpler for demo)
        printf("Loading policy as library...\n");

        cudaError_t err = cudaLibraryLoadData(&library, policyPTX.data(), nullptr, nullptr, 0,
                                              nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to load library: %s\n", cudaGetErrorString(err));
            return false;
        }

        printf("✓ Loaded policy library\n");
        loaded = true;
        return true;
    }

    bool getKernel(CUkernel* kernel, const char* name) {
        if (!loaded) {
            fprintf(stderr, "Library not loaded!\n");
            return false;
        }

        cudaError_t err = cudaLibraryGetKernel(kernel, library, name);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get kernel: %s (%s)\n", name, cudaGetErrorString(err));
            return false;
        }

        printf("✓ Got kernel: %s\n", name);
        return true;
    }
};
