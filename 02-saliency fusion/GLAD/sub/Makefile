default: em em_mex

clean:
	rm -f *.o em

# Set these variables to the point to the correct locations
CBLAS_LIB=/usr/local/lib
GSL_LIB=/usr/local/lib
GSL_INCLUDE=/usr/local/include/gsl
MATLAB=/usr/local/MATLAB/R2018a
#/usr/local/MATLAB/MATLAB_Production_Server/R2013a

#SET(MANDATORY_LIBRARIES ${MANDATORY_LIBRARIES} ${GSL_GSL_LIBRARY} cblas)

em_mex: em_mex.c em.c data.c data.h prob_functions.c prob_functions.h
	$(MATLAB)/bin/mex $(GSL_LIB)/libgsl.so $(CBLAS_LIB)/libgslcblas.so -I$(GSL_INCLUDE) -arch=glnxa64 em_mex.c em.c data.c prob_functions.c

em: em.c data.c data.h prob_functions.c prob_functions.h
	gcc -m64 -O3 $(GSL_LIB)/libgsl.a $(CBLAS_LIB)/libgslcblas.a -I$(GSL_INCLUDE) -L$(GSL_LIB)  -lgsl -lgslcblas -lm -D HAVE_INLINE -std=c99 -o em em.c data.c prob_functions.c


