CC = gcc
#CC = icc

CFLAGS = -O3 -march=nocona -ffast-math -fomit-frame-pointer -fopenmp
#CFLAGS = -lm -msse2 -O2 -march=nocona -ffast-math -fomit-frame-pointer -fopenmp 

#OMPFLAGS = -fopenmp

#CC=icc
#CFLAGS = -xP -fast
#OMPFLAGS = -openmp

LIB_TARGETS = libresize.so libdynamic.so libexcorr.so libhog.so libfastpegasos.so
all:	$(LIB_TARGETS)

libexcorr.so: excorr.c Makefile dynamic.c
	$(CC) $(CFLAGS) -shared -Wl,-soname=libexcorr.so -fPIC dynamic.c excorr.c -o libexcorr.so #libmyrmf.so.1.0.1

libdynamic.so: dynamic.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libdynamic.so -fPIC -lc -rdynamic dynamic.c -o libdynamic.so

libfastpegasos.so: fast_pegasos.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libfastpegasos.so -fPIC -lc fast_pegasos.c -o libfastpegasos.so

libresize.so:	resize.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libresize.so -fPIC resize.c -o libresize.so

libhog.so:	features2.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libhog.so -fPIC features2.c -o libhog.so

clean:
	rm -f *.o *.pyc $(EXE_TARGETS) $(LIB_TARGETS)


