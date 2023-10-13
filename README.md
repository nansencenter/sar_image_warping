# Efficient algorithm fow warping SAR imagery with motion compensation
## Anton Korosov<sup>1</sup>, Anna Telegina<sup>2</sup>
### 1. Nansen Environmenatl and Remote Sensing Centre (NERSC), Bergen, Norway
### 2. The Arctic University of Norway (UiT), Tromso, Norway
---------
## Introduction

Imagery from synthetic aperture radar (SAR) provides invaluable information about sea ice surface and used widely for operational monitoring.

Sea ice is in continuous motion under influence of wind and ocean currents. Therefore signature of sea ice on two SAR images of the same place acquired with some time separation cannot be directly compared. 

We developed a method for co-registration of SAR images with compensation for ice drift (or any other *a priori* known motion of ground targets).

## Methodology

Co-registration is performed by warping of one or both images from their original geometries to a new, common geometry. Co-registration with drift compensation is also performed by warping, but motion is taken into account during warping of one of the images.

If we denote the brightness of the pixels of the first SAR image by $\mathbf{I_1}$ then the brightness of the warped image can be comuted as:

$\mathbf{I_w} = \mathcal{M_I} (\mathbf{I_1}, \mathbf{R_w}, \mathbf{C_w})$.            (1)

$\mathcal{M_I}$ is the operation of resampling from the original coordinates of $\mathbf{I_0}$ to the new coordinates $\mathbf{R_w}$ and $\mathbf{C_w}$. The resampling is performed using function `map_coordinates` from the Python library `scipy` with nearest neigbour inteporlation. The original coordinates used are the row and column coordinates of all pixels of the image $\mathbf{I_0}$, and the new coordinates $\mathbf{R_w}$ and $\mathbf{C_w}$ are the row and column coordinates of all pixels on the warped image. Note that bold roman is used to denote $\mathbf{R_w}$ and $\mathbf{C_w}$ as matrices with the size of $\mathbf{I_w}$. These are computed as:

$\mathbf{R_w} = \mathcal{M_R} (\mathbf{R_2}, \mathbf{C_2})$            (2)

$\mathbf{C_w} = \mathcal{M_C} (\mathbf{R_2}, \mathbf{C_2})$            (3)

$\mathcal{M_R}$ and $\mathcal{M_C}$ are linear interpolators. They are applied to full size matrices of row/column coordinates of the second image, $\mathbf{R_2}$ and $\mathbf{C_2}$. 

The interpolators are trained unsing the function `LinearNDInterpolator` from the Python library `scipy` as follows:

$\mathcal{M_R} = \mathcal{T} (r_2, c_2, r_1)$            (4)

$\mathcal{M_C} = \mathcal{T} (r_2, c_2, c_1)$            (5)

Training $\mathcal{T}$ is performed on vectors of row and column coordinates on the first and the second SAR images: ($r_1$, $c_1$) and ($r_2$, $c_2$) (which usually have much smaller size than the full size matrices).

In the simplest case, when no drift compensation is performed, the coordinates ($r_1$, $c_1$) and ($r_2$, $c_2$) are the coordinates of the same geopgraphical points but in the coordinate systems of the first image and the second image. Therefore, ($r_2$, $c_2$) can be computed from ($r_1$, $c_1$) as follows:

$r_2, c_2 = \mathcal{D'_2} (x, y)$,            (6)

where ($x$, $y$) are the geopgraphic coordinates of the points:

$x, y = \mathcal{D_1} (r_1, c_1)$            (7)

Here $\mathcal{D}$ is the function to convert from a row/column coordinate system to a greopgraphical coordinate system, and $\mathcal{D'}$ is the inverse function. For SAR images in "swath-projection" (L1 or L2 ground range data) the $\mathcal{D}$ and $\mathcal{D'}$ operators are defined using ground control points (GCPs). For projected data (e.g., model data or satellite L3 and L4 products) the $\mathcal{D}$ and $\mathcal{D'}$ operators are defined using a spatial reference system (a.k.a., projection, e.g. north polar stereographic) and knowledge of the raster size and resolution. It should be noted that two SAR images have different geometry (unless they were taken on the same orbit) and, therefore, $\mathcal{D_1} \neq \mathcal{D_2}$ and $\mathcal{D'_1} \neq \mathcal{D'_2}$.

In a more complex case, when motion compensation should be included in warping, the coordinates on the second image are computed as follows:

$r_2, c_2 = \mathcal{D'_2} (x + \Delta x, y + \Delta y)$ ,            (8)

where $\Delta x$ and $\Delta y$ are displacement of the points due to motion (e.g. sea ice drift).

The algorithm for image warping with motion compensation can be formulated as follows:

1. Define a set of initial points on the first image with row/column coordinates $r_1$, $c_1$. The number of these points can be much smaller than the size of the image, but it should be sufficiently high for resolving the spatial variability of underlying motion. For example, for sea ice drift a SAR image with 40 m spacing is used, but the resolution of the ice drift product is usually about 4 - 10 km. Therefore, the number of initial points is 10.000 - 62.500 times smaller than the size of a SAR image.

2. Compute geographic coordinates $x$, $y$ using Eq. 7. Optionally, find corresponding displacement $\Delta x$, $\Delta y$ of these inital points using, for example, an ice drift algorithm.

3. Compute row/column coordinates of these points in the system of the second image $r_2$, $c_2$ using Eq. 8 (or Eq. 6 if $\Delta x = 0$ and $\Delta y = 0$).

4. Train interpolators the $\mathcal{M_R}$ and $\mathcal{M_C}$ using Eqs. 4 and 5.

5. Create grids with coordinates of the second image $\mathbf{R_2}$, $\mathbf{C_2}$, for example using function `meshgrid` from Python library `numpy`.

6. Apply the interpolators to compute the coordinates of the warped image $\mathbf{R_w}$, $\mathbf{C_w}$ using Eqs. 2 and 3. 

7. Apply resampling to compute the warped image $\mathbf{I_w}$ using Eq. 1.

## Results

The new method is realised as Python code below. Examples of code application are provided on real SAR data from Sentinel-1.


```python
import matplotlib.pyplot as plt
from nansat import Nansat, Domain, NSR
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import distance_transform_edt
from sea_ice_drift import get_n
from sea_ice_drift.ftlib import feature_tracking 
from sea_ice_drift.pmlib import pattern_matching
```


```python
def get_dst_rows_cols(dst_dom):
    """ Create grids with row, column coordinates of the destination domain """
    rows2, cols2 = np.meshgrid(
        np.arange(0, dst_dom.shape()[0]),
        np.arange(0, dst_dom.shape()[1]),
        indexing='ij',
    )
    return rows2, cols2

def warp_with_rowcol(src_dom, src_img, c1, r1, c2, r2, dst_dom):
    """ Train interpolators of coordinates and apply to full resolution coordinates to computed a warped image """
    interp_r1 = LinearNDInterpolator(list(zip(r2, c2)), r1)
    interp_c1 = LinearNDInterpolator(list(zip(r2, c2)), c1)
    rows2, cols2 = get_dst_rows_cols(dst_dom)
    r1a = np.clip(interp_r1((rows2, cols2)), 0, src_dom.shape()[0])
    c1a = np.clip(interp_c1((rows2, cols2)), 0, src_dom.shape()[1])
    dst_img = map_coordinates(src_img, (r1a, c1a), order=0)
    return dst_img

def warp_distance(dst_dom, lon1, lat1, mask):
    """ Create a matrix with distance to the nearest valid drift and warp it onto the destination domain """
    c2_dist, r2_dist = dst_dom.transform_points(lon1.flatten(), lat1.flatten(), DstToSrc=1)
    mask_dist = distance_transform_edt(mask)
    interp_dist = LinearNDInterpolator(list(zip(r2_dist, c2_dist)), mask_dist.flatten())
    rows2, cols2 = get_dst_rows_cols(dst_dom)
    dst_dist = interp_dist((rows2, cols2))
    return dst_dist

def warp_with_lonlat(src_dom, src_img, lon1, lat1, lon2, lat2, dst_dom):
    """ Warp input image on destination domain if vectors of lon,lat source and destination points are knwown """
    c1, r1 = src_dom.transform_points(lon1.flatten(), lat1.flatten(), DstToSrc=1)
    c2, r2 = dst_dom.transform_points(lon2.flatten(), lat2.flatten(), DstToSrc=1)
    dst_img = warp_with_rowcol(src_dom, src_img, c1, r1, c2, r2, dst_dom)
    return dst_img

def warp(src_dom, src_img, dst_dom, step=None):
    """ Warp input image on destination domain (without drift compensation) """
    if step is None:
        step = int(src_dom.shape()[0]/100)
    src_lon, src_lat = src_dom.get_geolocation_grids(step)
    dst_img = warp_with_lonlat(src_dom, src_img, src_lon, src_lat, src_lon, src_lat, dst_dom)
    return dst_img

def warp_and_mask_with_lonlat(src_dom, src_img, lon1, lat1, lon2, lat2, mask, dst_dom, max_dist=2, fill_value=0):
    """ Warp input image on destination domain with drift compensation and masking if lon,lat,mask matrices are given """
    lon1v, lat1v, lon2v, lat2v = [i[~mask] for i in [lon1, lat1, lon2, lat2]]
    dst_img = warp_with_lonlat(src_dom, src_img, lon1v, lat1v, lon2v, lat2v, dst_dom)
    dst_dist = warp_distance(dst_dom, lon1, lat1, mask)
    dst_img[(dst_dist > max_dist) + np.isnan(dst_dist)] = fill_value
    return dst_img

def warp_with_uv(src_dom, src_img, uv_dom, u, v, mask, dst_dom):
    """ Warp input image on destination domain with drift compensation and masking if U,V,mask matrices are given """
    uv_srs = NSR(uv_dom.vrt.get_projection()[0])
    lon1uv, lat1uv = uv_dom.get_geolocation_grids()
    x1, y1, _ = uv_dom.vrt.transform_coordinates(NSR(), (lon1uv[~mask], lat1uv[~mask]), uv_srs)
    x2 = x1 + u[~mask]
    y2 = y1 + v[~mask]
    lon2uv, lat2uv, _ = uv_dom.vrt.transform_coordinates(uv_srs, (x2, y2), NSR())
    inp_img = np.array(src_img)
    inp_img[0] = 0
    inp_img[-1] = 0
    inp_img[:, 0] = 0
    inp_img[:, -1] = 0
    dst_img = warp_with_lonlat(src_dom, inp_img, lon1uv[~mask], lat1uv[~mask], lon2uv, lat2uv, dst_dom)
    return dst_img
```


```python
# use original Sentinel-1 SAR files (download from colhub.met.no)
f1 = 'S1B_EW_GRDM_1SDH_20200123T120618_20200123T120718_019944_025BA1_D4A2.SAFE'
f2 = 'S1B_EW_GRDM_1SDH_20200125T114955_20200125T115055_019973_025C81_EC1A.SAFE'

# create Nansat objects with one band only. 
n1 = get_n(f1, bandName='sigma0_HH', remove_spatial_mean=True)
n2 = get_n(f2, bandName='sigma0_HH', remove_spatial_mean=True)
```

    VMIN:  -3.348351526260376
    VMAX:  4.359616050720215
    VMIN:  -3.3655088901519776
    VMAX:  4.466654739379884



```python
# example model domain
mod_srs = NSR('+proj=stere +lon_0=-45 +lat_0=85')
mod_dom = Domain(mod_srs, '-te -400000 -400000 400000 400000 -tr 5000 5000')

# show model domain and SAR image domains
fig, axs = plt.subplots(1,1,figsize=(5,5))
axs.plot(*mod_dom.get_border(), '.-')
axs.plot(*n1.get_border(), '.-')
axs.plot(*n2.get_border(), '.-')
plt.show()
```


    
![png](README_files/README_4_0.png)
    



```python
# Derive sea ice drift from two SAR images
# Origins of drift vectors are located in the pixels of the example model domain
c1, r1, c2, r2 = feature_tracking(n1, n2, nFeatures=20000, ratio_test=0.6)
lon1pm, lat1pm = mod_dom.get_geolocation_grids()
upm, vpm, apm, rpm, hpm, lon2pm, lat2pm = pattern_matching(lon1pm, lat1pm, n1, c1, r1, n2, c2, r2, srs=mod_srs)
```

    85% 01257.0 03374.2 0000nan 0000nan +0nan 0nan 0nan
     Pattern matching - OK! (  5 sec)



```python
x, y = mod_dom.get_geolocation_grids(dst_srs=mod_srs)

# show SAR-derived ice drift and quality metric (max cross-correlation times Hessian)
fig, axs = plt.subplots(1,2, figsize=(10,5), sharex=True, sharey=True)
stp = 5
axs[0].quiver(x[::stp, ::stp], y[::stp, ::stp], upm[::stp, ::stp], vpm[::stp, ::stp], width=0.01, scale=50000)
axs[1].pcolormesh(x, y, rpm*hpm, clim=[0,15])
```




    <matplotlib.collections.QuadMesh at 0x7f1a9f455b70>




    
![png](README_files/README_6_1.png)
    



```python
# create example model drift
ccc, rrr = np.meshgrid(np.arange(mod_dom.shape()[0]), np.arange(mod_dom.shape()[1]), indexing='ij')
umo = (ccc - ccc.mean()) * 3000
vmo = (rrr - rrr.mean()) * 3000

fig, axs = plt.subplots(1,1,figsize=(5,5))
stp = 10
axs.quiver(x[::stp, ::stp], y[::stp, ::stp], umo[::stp,::stp], vmo[::stp,::stp], angles='xy', scale_units='xy', scale=1)
plt.show()
```


    
![png](README_files/README_7_0.png)
    



```python
# Use case 1
# All images are warped on the same Domain
# 1. SAR_1 is warped with SAR-drift compensation
# 2. SAR_1 is warped with Model-drift compensation
# 3. SAR_2 is warped without drift compensation

# Define destination domain
dst_srs = NSR('+proj=stere +lon_0=-40 +lat_0=84.5')
dst_dom = Domain(dst_srs, '-te -450000 -450000 450000 450000 -tr 1000 1000')
plt.plot(*dst_dom.get_border(), '.-')
plt.plot(*n1.get_border(), '.-')
plt.plot(*n2.get_border(), '.-')
plt.show()
```


    
![png](README_files/README_8_0.png)
    



```python
# Warp SAR1 with SAR-drift compenstaion
good_pixels = (rpm*hpm) > 5
mask_pm = ~good_pixels # mask out low quality or NaN
s1_dst_dom_S = warp_with_uv(n1, n1[1], mod_dom, upm, vpm, mask_pm, dst_dom)

# Warp SAR1 with model-drift compenstaion
mask_mo = np.zeros(umo.shape, bool) # mask nothing
s1_dst_dom_M = warp_with_uv(n1, n1[1], mod_dom, umo, vmo, mask_mo, dst_dom)

# Warp SAR2
s2_dst_dom = warp(n2, n2[1], dst_dom)
```


```python
kwargs = dict(
    cmap='gray',
    clim=[0, 255],
)

fig, axs = plt.subplots(1,3, figsize=(15,5))
axs[0].imshow(s1_dst_dom_S, **kwargs)
axs[1].imshow(s1_dst_dom_M, **kwargs)
axs[2].imshow(s2_dst_dom, **kwargs)
plt.show()
```


    
![png](README_files/README_10_0.png)
    



```python
# Use Case 2
# 1st SAR image is warped on the second SAR image with SAR-drift compensation
s1_S2_S = warp_with_uv(n1, n1[1], mod_dom, upm, vpm, mask_pm, n2)
```


```python
fig, axs = plt.subplots(1,2, figsize=(15,7))
axs[0].imshow(n2[1], **kwargs)
axs[1].imshow(s1_S2_S, **kwargs)
plt.show()
```


    
![png](README_files/README_12_0.png)
    



```python

```
