CC      = hipcc
CFLAGS  = -x hip -O2 -march=native
LFLAGS  = -shared -fPIC
#INCLUDES := -Idiagonal.h

libhipdiagonal.so: diagonal.c
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) -o $@ $<

# little wrapper for dev, not necessary
dp_test: test_wrap.cpp
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f libhipdiagonal.so
	rm -f dp_test

.PHONY: clean dp_test
