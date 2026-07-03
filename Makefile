CXX      ?= g++
CXXFLAGS ?= -O2 -shared -fPIC -std=c++17
INCLUDES := -Ikseqpp_lib/include
LIBS     := -lz

kseq_wrapper.so: kseq_wrapper.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f kseq_wrapper.so

.PHONY: clean
