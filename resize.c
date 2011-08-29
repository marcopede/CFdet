#include <math.h>
#include <assert.h>
#include <string.h>

// struct used for caching interpolation values
struct alphainfo {
  int si, di;
  double alpha;
};

// copy src into dst using precomputed interpolation values
void alphacopy(double *src, double *dst, struct alphainfo *ofs, int n) {
  struct alphainfo *end = ofs + n;
  while (ofs != end) {
    dst[ofs->di] += ofs->alpha * src[ofs->si];
    ofs++;
  }
}

// resize along each column
// result is transposed, so we can apply it twice for a complete resize
void resize1dtran(double *src, int sheight, double *dst, int dheight, 
		  int width, int chan) {
  double scale = (double)dheight/(double)sheight;
  double invscale = (double)sheight/(double)dheight;
  
  // we cache the interpolation values since they can be 
  // shared among different columns
  int len = (int)ceil(dheight*invscale) + 2*dheight;
  struct alphainfo ofs[len];
  int k = 0;
  int dy,sy;
  for (dy = 0; dy < dheight; dy++) {
    double fsy1 = dy * invscale;
    double fsy2 = fsy1 + invscale;
    int sy1 = (int)ceil(fsy1);
    int sy2 = (int)floor(fsy2);       

    if (sy1 - fsy1 > 1e-3) {
      assert(k < len);
      assert(sy1 >= 0);
      ofs[k].di = dy*width;
      ofs[k].si = sy1-1;
      ofs[k++].alpha = (sy1 - fsy1) * scale;
    }

    for (sy = sy1; sy < sy2; sy++) {
      assert(k < len);
      assert(sy < sheight);
      ofs[k].di = dy*width;
      ofs[k].si = sy;
      ofs[k++].alpha = scale;
    }

    if (fsy2 - sy2 > 1e-3) {
      assert(k < len);
      assert(sy2 < sheight);
      ofs[k].di = dy*width;
      ofs[k].si = sy2;
      ofs[k++].alpha = (fsy2 - sy2) * scale;
    }
  }

  // resize each column of each color channel
  bzero(dst, chan*width*dheight*sizeof(double));
  int c,x;
  for (c = 0; c < chan; c++) {
    for (x = 0; x < width; x++) {
      double *s = src + c*width*sheight + x*sheight;
      double *d = dst + c*width*dheight + x;
      alphacopy(s, d, ofs, k);
    }
  }
}

