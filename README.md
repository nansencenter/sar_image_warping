# Efficient algorithm fow warping SAR imagery with motion compensation
## Anton Korosov$^1$, Anna Telegina$^2$
### 1. Nansen Environmenatl and Remote Sensing Centre, Bergen, Norway
### 2. Norwegian Arctic University, Tromso, Norway
---------
## Introduction

Imagery from synthetic aperture radar (SAR) provides invaluable information about sea ice surface and used widely for operational monitoring.

Sea ice is in continuous motion under influence of wind and ocean currents. Therefore signature of sea ice on two SAR images of the same place acquired with some time separation cannot be directly compared. 

We developed a method for co-registration of SAR images with compensation for ice drift (or any other *a priori* known motion of ground targets).

## Methodology

Co-registration is performed by warping of one or both images from their original geometries to a new, common geometry. Co-registration with drift compensation is also performed by warping, but motion is taken into account during warping of one of the images.

If we denote the brightness of the pixels of the first SAR image by $\mathbf{I_1}$ then the brightness of the warped image can be comuted as:

$ \mathbf{I_w} = \mathcal{M_I} (\mathbf{I_1}, \mathbf{R_w}, \mathbf{C_w}) $.

$\mathcal{M_I}$ is the operation of resampling from the original coordinates of $\mathbf{I_0}$ to the new coordinates $\mathbf{R_w}$ and $\mathbf{C_w}$. The resampling is performed using function `map_coordinates` from the Python library `scipy` with nearest neigbour inteporlation. The original coordinates used are the row and column coordinates of all pixels of the image $\mathbf{I_0}$, and the new coordinates $\mathbf{R_w}$ and $\mathbf{C_w}$ are the row and column coordinates of all pixels on the warped image. Note that bold roman is used to denote $\mathbf{R_w}$ and $\mathbf{C_w}$ as matrices with the size of $\mathbf{I_w}$. These are computed as:

$ \mathbf{R_w} = \mathcal{M_R} (\mathbf{R_2}, \mathbf{C_2}) $

$ \mathbf{C_w} = \mathcal{M_C} (\mathbf{R_2}, \mathbf{C_2}) $

$\mathcal{M_R}$ and $\mathcal{M_C}$ are linear interpolators. They are applied to full size matrices of row/column coordinates of the second image, $\mathbf{R_2}$ and $\mathbf{C_2}$. 

The interpolators are trained unsing the function `LinearNDInterpolator` from the Python library `scipy` as follows:

$ \mathcal{M_R} = \mathcal{T} (r_2, c_2, r_1) $

$ \mathcal{M_C} = \mathcal{T} (r_2, c_2, c_1) $

Training $\mathcal{T}$ is performed on vectors of row and column coordinates on the first and the second SAR images: ($r_1$, $c_1$) and ($r_2$, $c_2$) (which usually have much smaller size than the full size matrices).

In the simplest case, when no drift compensation is performed, the coordinates ($r_1$, $c_1$) and ($r_2$, $c_2$) are the coordinates of the same geopgraphical points but in the coordinate systems of the first image and the second image. Therefore, ($r_2$, $c_2$) can be computed from ($r_1$, $c_1$) as follows:

$r_2, c_2 = \mathcal{D'_2} (x, y)$,

where ($x$, $y$) are the geopgraphic coordinates of the points:

$x, y = \mathcal{D_1} (r_1, c_1)$

Here $\mathcal{D}$ is the function to convert from a row/column coordinate system to a greopgraphical coordinate system, and $\mathcal{D'}$ is the inverse function. For SAR images in "swath-projection" (L1 or L2 ground range data) the $\mathcal{D}$ and $\mathcal{D'}$ operators are defined using ground control points (GCPs). For projected data (e.g., model data or satellite L3 and L4 products) the $\mathcal{D}$ and $\mathcal{D'}$ operators are defined using a spatial reference system (a.k.a., projection, e.g. north polar stereographic) and knowledge of the raster size and resolution. It should be noted that two SAR images have different geometry (unless they were taken on the same orbit) and, therefore, $\mathcal{D_1} \neq \mathcal{D_2}$ and $\mathcal{D'_1} \neq \mathcal{D'_2}$.

In a more complex case, when motion compensation should be included in warping, the coordinates on the second image are computed as follows:

$r_2, c_2 = \mathcal{D'_2} (x + \Delta x, y + \Delta y)$ ,

where $\Delta x$ and $\Delta y$ are displacement of the points due to motion (e.g. sea ice drift).

The algorithm for image warping with motion compensation can be formulated as follows:

1. Define a set of initial points on the first image with row/column coordinates $r_1$, $c_1$. The number of these points can be much smaller than the size of the image, but it should be sufficiently high for resolving the spatial variability of underlying motion. For example, for sea ice drift a SAR image with 40 m spacing is used, but the resolution of the ice drift product is usually about 4 - 10 km. Therefore, the number of initial points is 10.000 - 62.500 times smaller.

2. Compute geographic coordinates $x$, $y$ and corresponding displacement $\Delta x$, $\Delta y$ for these inital points.

3. Compute row/column coordinates of these points in the system of the second image $r_w$, $c_2$.

4. Train interpolators the $\mathcal{M_R}$ and $\mathcal{M_C}$

5. Create grids with coordinates of the second image $\mathbf{R_2}$, $\mathbf{C_2}$

6. Apply the interpolators to compute the coordinates of the warped image $\mathbf{R_w}$, $\mathbf{C_w}$

7. Apply resampling to compute the warped image $\mathbf{I_w}$

## Results

The new method is realised as Python code in the notebook [warping.ipynb](warping.ipynb). Examples of code application are provided on real SAR data from Sentinel-1.