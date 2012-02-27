/*
Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick
Copyright (C) 2007 Pedro Felzenszwalb, Deva Ramanan

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// small value, used to avoid division by zero
#define eps 0.0001
#define ftype float//double
#define ftype2 float//double

// unit vectors used to compute gradient orientation
/*ftype2 uu[9] = {1.0000, 
		0.939692620785908, 
		0.76604444311897, 
		0.500, 
		0.1736481776669304, 
		-0.1736481776669304, 
		-0.5000, 
		-0.76604444311897, 
		-0.939692620785908};
ftype2 vv[9] = {0.0000, 
		0.342020143325668, 
		0.642787609686539, 
		0.866025403784438,
		0.984807753012208, 
		0.984807753012208, 
		0.866025403784438, 
		0.642787609686539, 
		0.342020143325668};*/

ftype uu[9] = {1.0000, 
		0.9397, 
		0.7660, 
		0.500, 
		0.1736, 
		-0.1736, 
		-0.5000, 
		-0.7660, 
		-0.9397};
ftype vv[9] = {0.0000, 
		0.3420, 
		0.6428, 
		0.8660, 
		0.9848, 
		0.9848, 
		0.8660, 
		0.6428, 
		0.3420};

static inline ftype2 min(ftype2 x, ftype2 y) { return (x <= y ? x : y); }
static inline ftype2 max(ftype2 x, ftype2 y) { return (x <= y ? y : x); }

static inline int mini(int x, int y) { return (x <= y ? x : y); }
static inline int maxi(int x, int y) { return (x <= y ? y : x); }

// main function:
// takes a ftype color image and a bin size 
// returns HOG features
void process(ftype *im, int dimy, int dimx, int sbin, ftype *feat, int hy, int hx, int hz) {

  // memory for caching orientation histograms & their norms
  int blocks[2];
  blocks[0] = (int)round((ftype2)dimy/(ftype2)sbin);
  blocks[1] = (int)round((ftype2)dimx/(ftype2)sbin);
  ftype2 *hist = (ftype2 *)calloc(blocks[0]*blocks[1]*18, sizeof(ftype2));
  ftype2 *norm = (ftype2 *)calloc(blocks[0]*blocks[1], sizeof(ftype2));

  // memory for HOG features
  int out[3];
  out[0] = max(blocks[0]-2, 0);
  out[1] = max(blocks[1]-2, 0);
  out[2] = 27+4+1;
  if (hy!=out[0] || hx!=out[1] || hz!=out[2])
  {
    printf("Error in hog shape\n");
    return;
  }
    
  int visible[2];
  visible[0] = blocks[0]*sbin;
  visible[1] = blocks[1]*sbin;
  
  int x,y,o;
  for (x = 1; x < visible[1]-1; x++) {
    for (y = 1; y < visible[0]-1; y++) {
      // first color channel
      ftype *s = im + mini(x, dimx-2)*dimy + mini(y, dimy-2);
      ftype2 dy = *(s+1) - *(s-1);
      ftype2 dx = *(s+dimy) - *(s-dimy);
      ftype2 v = dx*dx + dy*dy;

      // second color channel
      s += dimy*dimx;
      ftype2 dy2 = *(s+1) - *(s-1);
      ftype2 dx2 = *(s+dimy) - *(s-dimy);
      ftype2 v2 = dx2*dx2 + dy2*dy2;

      // third color channel
      s += dimy*dimx;
      ftype2 dy3 = *(s+1) - *(s-1);
      ftype2 dx3 = *(s+dimy) - *(s-dimy);
      ftype2 v3 = dx3*dx3 + dy3*dy3;

      // pick channel with strongest gradient
      if (v2 > v) {
	v = v2;
	dx = dx2;
	dy = dy2;
      } 
      if (v3 > v) {
	v = v3;
	dx = dx3;
	dy = dy3;
      }

      // snap to one of 18 orientations
      ftype2 dot,best_dot = 0;
      int best_o = 0, o = 0;
      for (o = 0; o < 9; o++) {
	dot = uu[o]*dx + vv[o]*dy;
	if (dot > best_dot) {
	  best_dot = dot;
	  best_o = o;
	} else if (-dot > best_dot) {
	  best_dot = -dot;
	  best_o = o+9;
	}
    /*if (dy==0) //wiothout this make things not symmetric
    {
        if (dx<=0)
        {
            best_o = 4 + rand()%2;
        }
        else
        {
            best_o = 4+9 + rand()%2;
        }
    }*/
      }
      
      // add to 4 histograms around pixel using linear interpolation
      ftype2 xp = ((ftype2)x+0.5)/(ftype2)sbin - 0.5;
      ftype2 yp = ((ftype2)y+0.5)/(ftype2)sbin - 0.5;
      int ixp = (int)floor(xp);
      int iyp = (int)floor(yp);
      ftype2 vx0 = xp-ixp;
      ftype2 vy0 = yp-iyp;
      ftype2 vx1 = 1.0-vx0;
      ftype2 vy1 = 1.0-vy0;
      v = sqrt(v);

      if (ixp >= 0 && iyp >= 0) {
	*(hist + ixp*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) += 
	  vx1*vy1*v;
      }

      if (ixp+1 < blocks[1] && iyp >= 0) {
	*(hist + (ixp+1)*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) += 
	  vx0*vy1*v;
      }

      if (ixp >= 0 && iyp+1 < blocks[0]) {
	*(hist + ixp*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) += 
	  vx1*vy0*v;
      }

      if (ixp+1 < blocks[1] && iyp+1 < blocks[0]) {
	*(hist + (ixp+1)*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) += 
	  vx0*vy0*v;
      }
    }
  }
  // compute energy in each block by summing over orientations
  for (o = 0; o < 9; o++) {
    ftype2 *src1 = hist + o*blocks[0]*blocks[1];
    ftype2 *src2 = hist + (o+9)*blocks[0]*blocks[1];
    ftype2 *dst = norm;
    ftype2 *end = norm + blocks[1]*blocks[0];
    while (dst < end) {
      *(dst++) += (*src1 + *src2) * (*src1 + *src2);
      src1++;
      src2++;
    }
  }

  // compute features
  for (x = 0; x < out[1]; x++) {
    for (y = 0; y < out[0]; y++) {
      ftype *dst = feat + x*out[0] + y;      
      ftype2 *src, *p, n1, n2, n3, n4;

      p = norm + (x+1)*blocks[0] + y+1;
      n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
      p = norm + (x+1)*blocks[0] + y;
      n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
      p = norm + x*blocks[0] + y+1;
      n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
      p = norm + x*blocks[0] + y;      
      n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);

      ftype2 h1,t1 = 0;
      ftype2 h2,t2 = 0;
      ftype2 h3,t3 = 0;
      ftype2 h4,t4 = 0;

      // contrast-sensitive features
      src = hist + (x+1)*blocks[0] + (y+1);
      for (o = 0; o < 18; o++) {
	    h1 = min(*src * n1, 0.2);
	    h2 = min(*src * n2, 0.2);
	    h3 = min(*src * n3, 0.2);
	    h4 = min(*src * n4, 0.2);
	*dst = 0.5 * (h1 + h2 + h3 + h4);
	t1 += h1;
	t2 += h2;
	t3 += h3;
	t4 += h4;
	dst += out[0]*out[1];
	src += blocks[0]*blocks[1];
      }

      // contrast-insensitive features
      src = hist + (x+1)*blocks[0] + (y+1);
      for (o = 0; o < 9; o++) {
	ftype2 sum = *src + *(src + 9*blocks[0]*blocks[1]);
	ftype2 h1 = min(sum * n1, 0.2);
	ftype2 h2 = min(sum * n2, 0.2);
	ftype2 h3 = min(sum * n3, 0.2);
	ftype2 h4 = min(sum * n4, 0.2);
	*dst = 0.5 * (h1 + h2 + h3 + h4);
	dst += out[0]*out[1];
	src += blocks[0]*blocks[1];
      }

      // texture gradients
      *dst = 0.2357 * t1;
      dst += out[0]*out[1];
      *dst = 0.2357 * t2;
      dst += out[0]*out[1];
      *dst = 0.2357 * t3;
      dst += out[0]*out[1];
      *dst = 0.2357 * t4;
    }
  }
  free(hist);
  free(norm);
}




