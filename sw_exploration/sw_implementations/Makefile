CC      = hipcc
CFLAGS  = -x hip -O2 -march=native
CLFLAGS  = -shared -fPIC
CINCLUDES := -Ikseqpp_lib/include

CXX      = hipcc
CXXFLAGS = -x hip -O2 -std=c++17 -lz
CXXINCLUDES := -I../../kseqpp_lib/include

libhipdiagonal.so: diagonal.c
	$(CC) $(CFLAGS) $(CLFLAGS) $(INCLUDES) -o $@ $<

# little wrapper for dev, not necessary
dp_test: test_wrap.cpp
	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -o $@ $<

clean:
	rm -f libhipdiagonal.so
	rm -f dp_test

.PHONY: clean dp_test
