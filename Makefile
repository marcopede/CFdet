CC = gcc
CP = g++#icc#g++
#CC = icc

CFLAGS = -O3 -march=nocona -ffast-math -fomit-frame-pointer -fopenmp
#CFLAGS = -lm -msse2 -O2 -march=nocona -ffast-math -fomit-frame-pointer -fopenmp 

#OMPFLAGS = -fopenmp

#CC=icc
#CFLAGS = -xP -fast
#OMPFLAGS = -openmp

LIB_TARGETS = libresize.so libdynamic.so libexcorr.so libhog.so libfastpegasos.so #libcrf2.so #libcudahog.so
all:	$(LIB_TARGETS)

libcrf2.so: ./MRF2.1/myexample2.cpp Makefile
	$(CP) $(CFLAGS) -shared -Wl,-soname=libcrf2.so -DUSE_64_BIT_PTR_CAST -fPIC ./MRF2.1/myexample2.cpp ./MRF2.1/GCoptimization.cpp ./MRF2.1/maxflow.cpp ./MRF2.1/graph.cpp ./MRF2.1/LinkedBlockList.cpp ./MRF2.1/TRW-S.cpp ./MRF2.1/BP-S.cpp ./MRF2.1/ICM.cpp ./MRF2.1/MaxProdBP.cpp ./MRF2.1/mrf.cpp ./MRF2.1/regions-maxprod.cpp -o libcrf2.so

libexcorr.so: excorr.c Makefile dynamic.c
	$(CC) $(CFLAGS) -lm -msse2 -O2 -shared -Wl,-soname=libexcorr.so -fPIC dynamic.c excorr.c -o libexcorr.so #libmyrmf.so.1.0.1

libdynamic.so: dynamic.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libdynamic.so -fPIC -lc -rdynamic dynamic.c -o libdynamic.so

libfastpegasos.so: fast_pegasos.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libfastpegasos.so -fPIC -lc fast_pegasos.c -o libfastpegasos.so

libresize.so:	resize.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libresize.so -fPIC resize.c -o libresize.so

libhog.so:	features2.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libhog.so -fPIC features2.c -o libhog.so

libcudahog.so:	process.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libcudahog.so -fPIC process.c -o libcudahog.so

clean:
	rm -f *.o *.pyc $(EXE_TARGETS) $(LIB_TARGETS)


