# NMake Makefile for QuickLM (Windows MSVC compiler)

# Suppress all command echo banners in NMake
.SILENT:

CXX = cl.exe
# Source lives in domain subfolders (tensor/ ops/ model/ io/). Includes are bare
# filenames (#include "tensor.hpp"), so add every subfolder to the include search
# path and the preprocessor resolves them regardless of which folder a file is in.
INCLUDES = /Isrc\tensor /Isrc\ops /Isrc\model /Isrc\io /Isrc\arch
CXXFLAGS = /nologo /O2 /Oi /Ot /EHsc /arch:AVX2 /std:c++17 /DNOMINMAX /W3 $(INCLUDES)

OBJECTS = dist\cache\math_ops.obj dist\cache\safetensors.obj dist\cache\tokenizer.obj dist\cache\chat_template.obj dist\cache\weights.obj dist\cache\model.obj dist\cache\registry.obj dist\cache\qwen3_5.obj dist\cache\generic_transformer.obj dist\cache\main.obj dist\cache\attention.obj dist\cache\speculative.obj
TARGET = dist\quicklm.exe

# All headers. Every object depends on these so that editing any header (e.g.
# changing the Tensor struct) forces a full recompile. NMake does not scan
# #include dependencies itself; without this, stale objects compiled against an
# old struct layout link against new ones -> ABI mismatch / heap corruption.
HEADERS = src\tensor\tensor.hpp src\ops\math_ops.hpp src\io\json_parser.hpp src\io\safetensors.hpp src\io\tokenizer.hpp src\io\chat_template.hpp src\model\config.hpp src\model\weights.hpp src\model\modules.hpp src\model\model.hpp src\model\attention.hpp src\model\speculative.hpp src\arch\architecture.hpp src\arch\registry.hpp src\arch\qwen3_5.hpp src\arch\generic_transformer.hpp

all: $(TARGET)

# Link step
$(TARGET): $(OBJECTS)
	if not exist dist mkdir dist
	$(CXX) /nologo /Fe$(TARGET) $(OBJECTS)
	del /q dist\*.ilk dist\*.exp dist\*.lib 2>nul

# Explicit compilation rules for each module to ensure they output to dist\cache
dist\cache\math_ops.obj: src\ops\math_ops.cpp $(HEADERS)
	if not exist dist\cache mkdir dist\cache
	$(CXX) $(CXXFLAGS) /c src\ops\math_ops.cpp /Fodist\cache\math_ops.obj

dist\cache\safetensors.obj: src\io\safetensors.cpp $(HEADERS)
	if not exist dist\cache mkdir dist\cache
	$(CXX) $(CXXFLAGS) /c src\io\safetensors.cpp /Fodist\cache\safetensors.obj

dist\cache\tokenizer.obj: src\io\tokenizer.cpp $(HEADERS)
	if not exist dist\cache mkdir dist\cache
	$(CXX) $(CXXFLAGS) /c src\io\tokenizer.cpp /Fodist\cache\tokenizer.obj

dist\cache\chat_template.obj: src\io\chat_template.cpp $(HEADERS)
	if not exist dist\cache mkdir dist\cache
	$(CXX) $(CXXFLAGS) /c src\io\chat_template.cpp /Fodist\cache\chat_template.obj

dist\cache\weights.obj: src\model\weights.cpp $(HEADERS)
	if not exist dist\cache mkdir dist\cache
	$(CXX) $(CXXFLAGS) /c src\model\weights.cpp /Fodist\cache\weights.obj

dist\cache\model.obj: src\model\model.cpp $(HEADERS)
	if not exist dist\cache mkdir dist\cache
	$(CXX) $(CXXFLAGS) /c src\model\model.cpp /Fodist\cache\model.obj

dist\cache\registry.obj: src\arch\registry.cpp $(HEADERS)
	if not exist dist\cache mkdir dist\cache
	$(CXX) $(CXXFLAGS) /c src\arch\registry.cpp /Fodist\cache\registry.obj

dist\cache\qwen3_5.obj: src\arch\qwen3_5.cpp $(HEADERS)
	if not exist dist\cache mkdir dist\cache
	$(CXX) $(CXXFLAGS) /c src\arch\qwen3_5.cpp /Fodist\cache\qwen3_5.obj

dist\cache\main.obj: src\main.cpp $(HEADERS)
	if not exist dist\cache mkdir dist\cache
	$(CXX) $(CXXFLAGS) /c src\main.cpp /Fodist\cache\main.obj

dist\cache\attention.obj: src\model\attention.cpp $(HEADERS)
	if not exist dist\cache mkdir dist\cache
	$(CXX) $(CXXFLAGS) /c src\model\attention.cpp /Fodist\cache\attention.obj

dist\cache\speculative.obj: src\model\speculative.cpp $(HEADERS)
	if not exist dist\cache mkdir dist\cache
	$(CXX) $(CXXFLAGS) /c src\model\speculative.cpp /Fodist\cache\speculative.obj

dist\cache\generic_transformer.obj: src\arch\generic_transformer.cpp $(HEADERS)
	if not exist dist\cache mkdir dist\cache
	$(CXX) $(CXXFLAGS) /c src\arch\generic_transformer.cpp /Fodist\cache\generic_transformer.obj

clean:
	if exist dist rmdir /s /q dist 2>nul || exit 0
