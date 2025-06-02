CXX = g++
CXXFLAGS = -Wall -O3 -std=c++17
OPENVINO_CFLAGS = $(shell pkg-config --cflags openvino 2>/dev/null || echo "-DOPENVINO_NOT_FOUND")
OPENVINO_LIBS = $(shell pkg-config --libs openvino 2>/dev/null || echo "")

all: npu_matrix_mul

npu_matrix_mul: npu_matrix_mul.cpp
	@if [ "$(OPENVINO_CFLAGS)" = "-DOPENVINO_NOT_FOUND" ]; then \
		echo "OpenVINO not found. Cannot build NPU example."; \
		echo "To build this example, install OpenVINO development package:"; \
		echo "  1. Download from https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html"; \
		echo "  2. Follow installation instructions"; \
		echo "  3. Set up environment variables"; \
		exit 1; \
	else \
		$(CXX) $(CXXFLAGS) -DHAVE_OPENVINO $(OPENVINO_CFLAGS) -o npu_matrix_mul npu_matrix_mul.cpp $(OPENVINO_LIBS); \
	fi

clean:
	rm -f npu_matrix_mul 