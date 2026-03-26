HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17

SOURCES := $(wildcard outer_product*.cc)
TARGETS := $(SOURCES:.cc=)

ROCBLAS_SOURCES := outer_product_gemm.cc outer_product_ger.cc
ROCBLAS_TARGETS := $(ROCBLAS_SOURCES:.cc=)
NON_ROCBLAS_TARGETS := $(filter-out $(ROCBLAS_TARGETS),$(TARGETS))

.PHONY: all clean list

all: $(TARGETS)

$(NON_ROCBLAS_TARGETS): %: %.cc
	$(HIPCC) $(CXXFLAGS) $< -o $@

$(ROCBLAS_TARGETS): %: %.cc
	$(HIPCC) $(CXXFLAGS) $< -o $@ -lrocblas

list:
	@printf '%s\n' $(TARGETS)

clean:
	rm -f $(TARGETS)
