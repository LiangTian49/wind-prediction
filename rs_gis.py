"""GIS and numerical building blocks for native-grid static feature preparation."""

import numpy as np

LAND_CODES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)


def weighted_statistics(values, valid, weights):
    mask = valid & np.isfinite(values) & (weights > 0)
    area = float(weights[mask].sum())
    if area == 0:
        return np.nan, np.nan, 0.
    mean = float(np.sum(values[mask]*weights[mask])/area)
    variance = float(np.sum(weights[mask]*(values[mask]-mean)**2)/area)
    return mean, float(np.sqrt(max(variance, 0))), area


def landcover_statistics(values, valid, weights):
    known = valid & np.isfinite(values) & np.isin(values, LAND_CODES)
    areas = np.array([weights[known & (values == code)].sum() for code in LAND_CODES], dtype=float)
    valid_area = float(areas.sum())
    fractions = areas/valid_area if valid_area > 0 else np.full(11, np.nan)
    unknown_area = float(weights[valid & ~np.isin(values, LAND_CODES)].sum())
    nodata_area = float(weights[~valid].sum())
    return fractions, valid_area, unknown_area, nodata_area


def horn_slope(elevation, resolution):
    """Horn 3x3 slope in degrees; all nine heights must be valid."""
    z = np.asarray(elevation, dtype=float)
    if z.ndim != 2 or min(z.shape) < 3 or resolution <= 0:
        raise ValueError('Require a 2D grid of at least 3x3 and positive metre spacing.')
    blocks = [z[dy:dy+z.shape[0]-2, dx:dx+z.shape[1]-2]
              for dy in range(3) for dx in range(3)]
    valid = np.logical_and.reduce([np.isfinite(block) for block in blocks])
    a, b, c, d, _, f, g, h, i = blocks
    gx = ((c+2*f+i)-(a+2*d+g))/(8*resolution)
    gy = ((g+2*h+i)-(a+2*b+c))/(8*resolution)
    slope = np.full(z.shape, np.nan)
    inner = np.rad2deg(np.arctan(np.hypot(gx, gy)))
    inner[~valid] = np.nan
    slope[1:-1, 1:-1] = inner
    return slope


def native_projection(header):
    from pyproj import CRS, Proj

    if header['gridType'] != 'lambert' or header['scanningMode'] != 64 or header['shapeOfTheEarth'] != 6:
        raise ValueError('This protocol expects the audited spherical Lambert grid and scan mode 64.')
    text = (f'+proj=lcc +lat_1={header["Latin1InDegrees"]} +lat_2={header["Latin2InDegrees"]} '
            f'+lat_0={header["LaDInDegrees"]} +lon_0={header["LoVInDegrees"]} '
            f'+R={header["radius"]} +units=m +no_defs')
    crs = CRS.from_proj4(text)
    projection = Proj(crs)
    longitude = (header['longitudeOfFirstGridPointInDegrees']+180) % 360-180
    origin = projection(longitude, header['latitudeOfFirstGridPointInDegrees'])
    return crs, projection, origin


def grid_cell(row, header, projection, origin):
    dx, dy = header['DxInMetres'], header['DyInMetres']
    x, y = origin[0]+row.center_x*dx, origin[1]+row.center_y*dy
    actual_x, actual_y = projection(row.grid_lon, row.grid_lat)
    error = float(np.hypot(x-actual_x, y-actual_y))
    bounds = (x-dx/2, y-dy/2, x+dx/2, y+dy/2)
    left, bottom, right, top = bounds
    s = np.linspace(0, 1, 33)
    xs = np.concatenate([left+(right-left)*s, np.full(33, right),
                         right-(right-left)*s, np.full(33, left)])
    ys = np.concatenate([np.full(33, bottom), bottom+(top-bottom)*s,
                         np.full(33, top), top-(top-bottom)*s])
    lon, lat = projection(xs, ys, inverse=True)
    coordinates = np.column_stack([lon, lat]).tolist()
    coordinates.append(coordinates[0])
    geo_bounds = (float(lon.min())-.001, float(lat.min())-.001,
                  float(lon.max())+.001, float(lat.max())+.001)
    return dict(x=x, y=y, bounds=bounds, geo_bounds=geo_bounds,
                center_error_m=error, coordinates=coordinates)


def projected_pixel_weights(transform, shape, source_crs, target_crs, bounds):
    """Area of each projected native raster-pixel quadrilateral inside a target cell."""
    from pyproj import Transformer

    h, w = shape
    cols, rows = np.meshgrid(np.arange(w+1), np.arange(h+1))
    xx = transform.c+transform.a*cols+transform.b*rows
    yy = transform.f+transform.d*cols+transform.e*rows
    project = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    xx, yy = project.transform(xx, yy)
    if not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError('Raster-to-native projection produced invalid coordinates.')
    left, bottom, right, top = bounds
    # Use local coordinates to keep the shoelace area numerically stable.
    xx, yy = xx-left, yy-bottom
    width, height = right-left, top-bottom
    x = [xx[:-1, :-1], xx[:-1, 1:], xx[1:, 1:], xx[1:, :-1]]
    y = [yy[:-1, :-1], yy[:-1, 1:], yy[1:, 1:], yy[1:, :-1]]
    inside = np.logical_and.reduce([(a >= 0) & (a <= width) & (b >= 0) & (b <= height)
                                   for a, b in zip(x, y)])
    candidate = ((np.maximum.reduce(x) >= 0) & (np.minimum.reduce(x) <= width)
                 & (np.maximum.reduce(y) >= 0) & (np.minimum.reduce(y) <= height))
    area = np.zeros((h, w))
    polygon_area = np.abs(sum(x[j]*y[(j+1) % 4]-x[(j+1) % 4]*y[j] for j in range(4)))/2
    area[inside] = polygon_area[inside]
    edge = candidate & ~inside
    if edge.any():
        import shapely

        corners = np.stack([np.stack([a[edge], b[edge]], axis=-1) for a, b in zip(x, y)], axis=1)
        pixels = shapely.polygons(corners)
        area[edge] = shapely.area(shapely.intersection(pixels, shapely.box(0, 0, width, height)))
    return area


def regular_pixel_weights(transform, shape, bounds):
    h, w = shape
    if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
        raise ValueError('Expected a north-up metric raster.')
    left, bottom, right, top = bounds
    xs = transform.c+np.arange(w)*transform.a
    yt = transform.f+np.arange(h)*transform.e
    widths = np.maximum(0, np.minimum(xs+transform.a, right)-np.maximum(xs, left))
    heights = np.maximum(0, np.minimum(yt, top)-np.maximum(yt+transform.e, bottom))
    return heights[:, None]*widths[None, :]


def intersects(dataset, geo_bounds):
    a, b, c, d = geo_bounds
    e, f, g, h = dataset.bounds
    return c > e and a < g and d > f and b < h


def read_native_window(dataset, geo_bounds, crs, bounds):
    from rasterio.windows import Window, from_bounds

    window = from_bounds(*geo_bounds, transform=dataset.transform)
    c0 = max(0, int(np.floor(window.col_off))-1)
    r0 = max(0, int(np.floor(window.row_off))-1)
    c1 = min(dataset.width, int(np.ceil(window.col_off+window.width))+1)
    r1 = min(dataset.height, int(np.ceil(window.row_off+window.height))+1)
    if c1 <= c0 or r1 <= r0:
        raise ValueError('Empty raster window after bounds intersection.')
    window = Window(c0, r0, c1-c0, r1-r0)
    image = dataset.read(1, window=window, masked=True)
    values = np.asarray(image.data, dtype=float)
    valid = ~np.ma.getmaskarray(image) & np.isfinite(values)
    weights = projected_pixel_weights(dataset.window_transform(window), values.shape,
                                      dataset.crs, crs, bounds)
    return values, valid, weights


def warp_mosaic(datasets, crs, transform, shape, categorical=False):
    """Align source tiles to one small metric grid, retaining masks and NaNs."""
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    result = np.full(shape, np.nan)
    maximum_overlap_difference = 0.
    for dataset in datasets:
        with WarpedVRT(dataset, crs=crs.to_wkt(), transform=transform, width=shape[1], height=shape[0],
                       dtype='float64', nodata=np.nan, init_dest_nodata=True,
                       resampling=Resampling.nearest if categorical else Resampling.bilinear) as vrt:
            warped = vrt.read(1, masked=True)
            valid = ~np.ma.getmaskarray(warped) & np.isfinite(warped.data)
            overlap = valid & np.isfinite(result)
            if overlap.any():
                maximum_overlap_difference = max(maximum_overlap_difference,
                    float(np.abs(result[overlap]-warped.data[overlap]).max()))
            fill = valid & ~np.isfinite(result)
            result[fill] = warped.data[fill]
    return result, maximum_overlap_difference


def extract_cell(cell, crs, land_sources, dsm_sources, slope_resolution):
    from rasterio.transform import from_origin

    bounds, geo = cell['bounds'], cell['geo_bounds']
    area = (bounds[2]-bounds[0])*(bounds[3]-bounds[1])
    land_sources = [source for source in land_sources if intersects(source, geo)]
    dsm_sources = [source for source in dsm_sources if intersects(source, geo)]
    class_areas = np.zeros(11)
    lc_geometric = lc_unknown = lc_nodata = 0.
    dsm_values, dsm_valid, dsm_weights = [], [], []
    for source in land_sources:
        values, valid, weights = read_native_window(source, geo, crs, bounds)
        fractions, valid_area, unknown_area, nodata_area = landcover_statistics(values, valid, weights)
        if valid_area > 0:
            class_areas += fractions*valid_area
        lc_geometric += weights.sum()
        lc_unknown += unknown_area
        lc_nodata += nodata_area
    valid_area = class_areas.sum()
    fractions = class_areas/valid_area if valid_area > 0 else np.full(11, np.nan)
    dsm_geometric = dsm_nodata = 0.
    for source in dsm_sources:
        values, valid, weights = read_native_window(source, geo, crs, bounds)
        dsm_geometric += weights.sum()
        dsm_nodata += weights[~valid].sum()
        dsm_values.append(values.ravel())
        dsm_valid.append(valid.ravel())
        dsm_weights.append(weights.ravel())
    if dsm_values:
        mean, sd, dsm_area = weighted_statistics(np.concatenate(dsm_values),
                                                 np.concatenate(dsm_valid), np.concatenate(dsm_weights))
    else:
        mean, sd, dsm_area = np.nan, np.nan, 0.
    left, bottom, right, top = bounds
    resolution = slope_resolution
    x0 = np.floor(left/resolution)*resolution-2*resolution
    y0 = np.ceil(top/resolution)*resolution+2*resolution
    width = int(np.ceil((right-x0)/resolution))+2
    height = int(np.ceil((y0-bottom)/resolution))+2
    transform = from_origin(x0, y0, resolution, resolution)
    dsm, overlap_difference = warp_mosaic(dsm_sources, crs, transform, (height, width))
    slope = horn_slope(dsm, resolution)
    metric_weights = regular_pixel_weights(transform, slope.shape, bounds)
    slope_mean, _, slope_area = weighted_statistics(slope, np.isfinite(slope), metric_weights)
    features = {f'lc_{code}': float(value) for code, value in zip(LAND_CODES, fractions)}
    features.update(dsm_mean_m=mean, dsm_sd_m=sd, dsm_slope_mean_deg=slope_mean)
    quality = dict(lc_valid_fraction=float(valid_area/area), lc_geometric_fraction=float(lc_geometric/area),
                   lc_nodata_fraction=float(lc_nodata/area), lc_unknown_fraction=float(lc_unknown/area),
                   lc_uncovered_fraction=float(max(0, 1-lc_geometric/area)),
                   dsm_valid_fraction=float(dsm_area/area), dsm_geometric_fraction=float(dsm_geometric/area),
                   dsm_nodata_fraction=float(dsm_nodata/area),
                   dsm_uncovered_fraction=float(max(0, 1-dsm_geometric/area)),
                   slope_valid_fraction=float(slope_area/area), slope_nodata_fraction=float(max(0, 1-slope_area/area)),
                   dsm_warp_overlap_max_difference_m=overlap_difference,
                   lc_sources=[source.name for source in land_sources],
                   dsm_sources=[source.name for source in dsm_sources])
    display = dict(dsm=dsm, slope=slope, transform=transform, land_sources=land_sources)
    return features, quality, display
