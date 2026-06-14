module CTMV1Core

using Dates
using NCDatasets
using Printf

# -----------------------------------------------------------------------------
# Constants and species indexing
# -----------------------------------------------------------------------------

const AVOGADRO = 6.02214076e23
const MW_NO_KG_PER_MOL = 30.006e-3
const DEFAULT_MIXING_DEPTH_M = 1000.0
const KM_PER_DEG_LAT = 111.32

const SPECIES = [:NO, :NO2, :O3, :OH, :HO2, :VOC, :RO2, :HCHO, :CO, :HNO3]
const NSPEC = length(SPECIES)
const SIDX = Dict(s => i for (i, s) in enumerate(SPECIES))
const GAS_CONSTANT = 8.314462618 # J/(mol K)

# -----------------------------------------------------------------------------
# Configuration containers
# -----------------------------------------------------------------------------

struct Grid2D
    nx::Int
    ny::Int
    xmin_km::Float64
    xmax_km::Float64
    ymin_km::Float64
    ymax_km::Float64
    dx_km::Float64
    dy_km::Float64
    lon0_deg::Float64
    lat0_deg::Float64
end

"""
Build grid from resolution targets dx_km and dy_km.
Realized spacing is adjusted to exactly span domain bounds.
"""
function make_grid(; dx_km=25.0, dy_km=25.0,
    xmin_km=-250.0, xmax_km=250.0,
    ymin_km=-500.0, ymax_km=500.0,
    lon0_deg=-2.0, lat0_deg=54.0)

    lx = xmax_km - xmin_km
    ly = ymax_km - ymin_km
    lx > 0.0 || error("xmax_km must exceed xmin_km")
    ly > 0.0 || error("ymax_km must exceed ymin_km")
    dx_km > 0.0 || error("dx_km must be positive")
    dy_km > 0.0 || error("dy_km must be positive")

    nx = max(1, ceil(Int, lx / dx_km))
    ny = max(1, ceil(Int, ly / dy_km))

    dx_real = lx / nx
    dy_real = ly / ny

    return Grid2D(nx, ny, xmin_km, xmax_km, ymin_km, ymax_km, dx_real, dy_real, lon0_deg, lat0_deg)
end

function x_center_km(grid::Grid2D, i)
    return grid.xmin_km + (i - 0.5) * grid.dx_km
end

function y_center_km(grid::Grid2D, j)
    return grid.ymin_km + (j - 0.5) * grid.dy_km
end

function km_to_lonlat(x_km, y_km, lon0_deg, lat0_deg)
    lat = lat0_deg + y_km / KM_PER_DEG_LAT
    lon = lon0_deg + x_km / (KM_PER_DEG_LAT * cosd(lat0_deg))
    return lon, lat
end

function build_lonlat_fields(grid::Grid2D)
    lon_field = zeros(Float64, grid.nx, grid.ny)
    lat_field = zeros(Float64, grid.nx, grid.ny)

    for j in 1:grid.ny
        yk = y_center_km(grid, j)
        for i in 1:grid.nx
            xk = x_center_km(grid, i)
            lon_field[i, j], lat_field[i, j] = km_to_lonlat(xk, yk, grid.lon0_deg, grid.lat0_deg)
        end
    end

    return lon_field, lat_field
end

function _time_bracket_weights(times::AbstractVector{DateTime}, t::DateTime)
    n = length(times)
    if t <= times[1]
        return 1, 1, 0.0
    elseif t >= times[n]
        return n, n, 0.0
    end

    i1 = searchsortedlast(times, t)
    i2 = i1 + 1
    dt_tot = Dates.value(times[i2] - times[i1])
    dt_off = Dates.value(t - times[i1])
    w = dt_off / dt_tot
    return i1, i2, w
end

function _wrap_lon(lon, lon0, dlon, nlon)
    span = dlon * nlon
    x = lon
    while x < lon0
        x += span
    end
    while x >= lon0 + span
        x -= span
    end
    return x
end

"""
Bilinear interpolation on a regular lon-lat grid with cyclic longitudes.

field2d is indexed as (lon, lat).
"""
function _bilinear_regular_lonlat(field2d, lon_vec, lat_vec, lon_t, lat_t)
    nlon = length(lon_vec)
    nlat = length(lat_vec)
    dlon = lon_vec[2] - lon_vec[1]
    dlat = lat_vec[2] - lat_vec[1]

    lonw = _wrap_lon(lon_t, lon_vec[1], dlon, nlon)

    fx_raw = (lonw - lon_vec[1]) / dlon
    i0 = floor(Int, fx_raw) + 1
    fx = fx_raw - floor(fx_raw)
    i0 = clamp(i0, 1, nlon)
    i1 = (i0 == nlon) ? 1 : i0 + 1

    fy_raw = (lat_t - lat_vec[1]) / dlat
    j0 = floor(Int, fy_raw) + 1
    fy = fy_raw - floor(fy_raw)
    j0 = clamp(j0, 1, nlat - 1)
    j1 = j0 + 1

    f00 = field2d[i0, j0]
    f10 = field2d[i1, j0]
    f01 = field2d[i0, j1]
    f11 = field2d[i1, j1]

    return (1 - fx) * (1 - fy) * f00 + fx * (1 - fy) * f10 + (1 - fx) * fy * f01 + fx * fy * f11
end

function _remap_to_model_grid!(out_field, src_field, lon_src, lat_src, lon_tgt, lat_tgt)
    for j in axes(out_field, 2)
        for i in axes(out_field, 1)
            out_field[i, j] = _bilinear_regular_lonlat(src_field, lon_src, lat_src, lon_tgt[i, j], lat_tgt[i, j])
        end
    end
    return nothing
end

function _wind_divergence!(div, u, v, grid::Grid2D)
    nx, ny = size(u)
    dx = grid.dx_km * 1000.0
    dy = grid.dy_km * 1000.0

    for j in 1:ny
        jm = max(1, j - 1)
        jp = min(ny, j + 1)
        for i in 1:nx
            im = max(1, i - 1)
            ip = min(nx, i + 1)
            dudx = (u[ip, j] - u[im, j]) / ((ip - im) * dx)
            dvdy = (v[i, jp] - v[i, jm]) / ((jp - jm) * dy)
            div[i, j] = dudx + dvdy
        end
    end
    return nothing
end

"""
Project cell-centered wind field toward horizontal non-divergence.

Solves a Poisson problem for velocity potential phi with a simple Jacobi method,
then subtracts grad(phi) from (u,v).
"""
function project_wind_mass_consistent!(u, v, grid::Grid2D; niter=120)
    nx, ny = size(u)
    div = zeros(Float64, nx, ny)
    _wind_divergence!(div, u, v, grid)

    dx = grid.dx_km * 1000.0
    dy = grid.dy_km * 1000.0
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    denom = 2.0 * (inv_dx2 + inv_dy2)

    phi = zeros(Float64, nx, ny)
    phi_new = similar(phi)

    for _ in 1:niter
        for j in 1:ny
            jm = max(1, j - 1)
            jp = min(ny, j + 1)
            for i in 1:nx
                im = max(1, i - 1)
                ip = min(nx, i + 1)
                phi_new[i, j] = ((phi[ip, j] + phi[im, j]) * inv_dx2 + (phi[i, jp] + phi[i, jm]) * inv_dy2 - div[i, j]) / denom
            end
        end
        phi, phi_new = phi_new, phi
    end

    # u <- u - dphi/dx, v <- v - dphi/dy
    for j in 1:ny
        jm = max(1, j - 1)
        jp = min(ny, j + 1)
        for i in 1:nx
            im = max(1, i - 1)
            ip = min(nx, i + 1)
            dphidx = (phi[ip, j] - phi[im, j]) / ((ip - im) * dx)
            dphidy = (phi[i, jp] - phi[i, jm]) / ((jp - jm) * dy)
            u[i, j] -= dphidx
            v[i, j] -= dphidy
        end
    end

    return nothing
end

struct RunConfig
    t_start::DateTime
    t_end::DateTime
    dt_sec::Float64
    receptor_lat_deg::Float64
    receptor_lon_deg::Float64
    mixing_depth_m::Float64
end

function default_config()
    return RunConfig(
        DateTime(2015, 1, 1, 0, 0, 0),
        DateTime(2015, 1, 2, 0, 0, 0),
        300.0,
        51.498926,
        -0.174777,
        DEFAULT_MIXING_DEPTH_M,
    )
end

# -----------------------------------------------------------------------------
# Initial and boundary helpers
# -----------------------------------------------------------------------------

"""
Return realistic background concentration vector [mol/m^3] in SPECIES order.
"""
function default_background_vector()
    c = zeros(Float64, NSPEC)
    c[SIDX[:NO]] = 2.0e-8
    c[SIDX[:NO2]] = 6.0e-7
    c[SIDX[:O3]] = 1.4e-6
    c[SIDX[:OH]] = 1.5e-12
    c[SIDX[:HO2]] = 8.0e-10
    c[SIDX[:VOC]] = 8.0e-7
    c[SIDX[:RO2]] = 1.5e-10
    c[SIDX[:HCHO]] = 8.0e-8
    c[SIDX[:CO]] = 6.0e-6
    c[SIDX[:HNO3]] = 8.0e-9
    return c
end

function initialize_state(grid::Grid2D; perturb=true)
    c0 = default_background_vector()
    c = zeros(Float64, grid.nx, grid.ny, NSPEC)

    for j in 1:grid.ny
        for i in 1:grid.nx
            @inbounds c[i, j, :] .= c0
            if perturb
                # Mild spatial structure helps avoid trivial chemistry transients.
                ξ = 1.0 + 0.05 * sin(2pi * i / grid.nx) * cos(2pi * j / grid.ny)
                c[i, j, SIDX[:NO]] *= ξ
                c[i, j, SIDX[:NO2]] *= ξ
                c[i, j, SIDX[:O3]] *= (2.0 - ξ)
            end
        end
    end

    return c
end

"""
Analytic boundary concentration (mol/m^3) by side and species.
"""
function boundary_concentration(side::Symbol, species::Symbol, lon_deg, lat_deg, t_utc::DateTime; bc_mode=:analytic)
    if bc_mode == :zero
        return 0.0
    elseif bc_mode != :analytic
        error("Unknown bc_mode $bc_mode. Choose :analytic or :zero")
    end

    hour = Dates.hour(t_utc) + Dates.minute(t_utc) / 60
    phase = 2pi * hour / 24.0

    if species == :NO
        base = 2.0e-8
    elseif species == :NO2
        base = 6.0e-7
    elseif species == :O3
        base = 1.4e-6
    elseif species == :CO
        base = 6.0e-6
    elseif species == :VOC
        base = 8.0e-7
    else
        base = default_background_vector()[SIDX[species]]
    end

    side_factor = if side == :west
        1.0 + 0.2 * sin(phase + 0.03 * lat_deg)
    elseif side == :east
        0.9 + 0.1 * cos(phase + 0.02 * lat_deg)
    elseif side == :south
        0.95 + 0.1 * cos(phase + 0.03 * lon_deg)
    else
        1.05 + 0.1 * sin(phase - 0.03 * lon_deg)
    end

    return max(0.0, base * side_factor)
end

# -----------------------------------------------------------------------------
# Emissions conversion and controls
# -----------------------------------------------------------------------------

"""
Convert NO emissions from kg/m^2/s to mol/m^3/s for a 2D mixed-layer model.
"""
function emissions_no_to_tendency_molm3s(emis_kgm2s; mixing_depth_m=DEFAULT_MIXING_DEPTH_M)
    return emis_kgm2s / MW_NO_KG_PER_MOL / mixing_depth_m
end

"""
Apply hourly emissions controls (piecewise-constant in UTC hour bin) for NO only.

alpha has size (nx, ny, 24) for v1.
baseline_no has size (nx, ny, 24) in kg/m^2/s.
"""
function apply_emissions_no!(state, baseline_no, alpha, dt_sec, utc_hour; mixing_depth_m=DEFAULT_MIXING_DEPTH_M)
    h = utc_hour + 1
    no_idx = SIDX[:NO]

    @boundscheck begin
        size(state, 1) == size(baseline_no, 1) || error("nx mismatch")
        size(state, 2) == size(baseline_no, 2) || error("ny mismatch")
        size(alpha, 3) == 24 || error("alpha must have 24 hourly bins")
    end

    for j in axes(state, 2)
        for i in axes(state, 1)
            e_eff = alpha[i, j, h] * baseline_no[i, j, h]
            sdot = emissions_no_to_tendency_molm3s(e_eff; mixing_depth_m=mixing_depth_m)
            state[i, j, no_idx] += dt_sec * sdot
        end
    end

    return nothing
end

# -----------------------------------------------------------------------------
# Solar geometry and photolysis scaling
# -----------------------------------------------------------------------------

"""
Approximate cosine solar zenith at UTC time and location.
Formula is lightweight and suitable for v1 prototyping.
"""
function cos_sza(dt_utc::DateTime, lat_deg, lon_deg)
    n = dayofyear(Date(dt_utc))
    hour = Dates.hour(dt_utc) + Dates.minute(dt_utc) / 60 + Dates.second(dt_utc) / 3600

    # Fractional year [rad].
    γ = 2pi / 365.0 * (n - 1 + (hour - 12) / 24)

    # Declination [rad] and equation of time [minutes].
    δ = 0.006918 - 0.399912 * cos(γ) + 0.070257 * sin(γ) - 0.006758 * cos(2γ) +
        0.000907 * sin(2γ) - 0.002697 * cos(3γ) + 0.00148 * sin(3γ)
    eot_min = 229.18 * (0.000075 + 0.001868 * cos(γ) - 0.032077 * sin(γ) - 0.014615 * cos(2γ) - 0.040849 * sin(2γ))

    time_offset_min = eot_min + 4.0 * lon_deg
    tst_min = hour * 60.0 + time_offset_min
    ha_deg = tst_min / 4.0 - 180.0

    lat = deg2rad(lat_deg)
    ha = deg2rad(ha_deg)

    μ = sin(lat) * sin(δ) + cos(lat) * cos(δ) * cos(ha)
    return max(0.0, μ)
end

"""
Scale noon-clear-sky J by solar zenith cosine clipped at zero.
"""
photolysis_rate(j_noon, dt_utc, lat_deg, lon_deg) = j_noon * cos_sza(dt_utc, lat_deg, lon_deg)

# -----------------------------------------------------------------------------
# Transport operator (MUSCL + limiter, conservative update)
# -----------------------------------------------------------------------------

minmod(a, b) = (a * b <= 0.0) ? 0.0 : sign(a) * min(abs(a), abs(b))
vanleer(a, b) = (a * b <= 0.0) ? 0.0 : (2.0 * a * b) / (a + b)
mc_limiter(a, b) = (a * b <= 0.0) ? 0.0 : sign(a) * min(0.5 * abs(a + b), 2.0 * abs(a), 2.0 * abs(b))

function _vanleer_partials(a, b)
    if a * b <= 0.0
        return 0.0, 0.0
    end
    den = a + b
    if abs(den) < 1.0e-30
        return 0.0, 0.0
    end
    dphi_da = 2.0 * b * b / (den * den)
    dphi_db = 2.0 * a * a / (den * den)
    return dphi_da, dphi_db
end

function get_limiter(name::Symbol)
    if name == :minmod
        return minmod
    elseif name == :vanleer
        return vanleer
    elseif name == :mc
        return mc_limiter
    else
        error("Unknown limiter $name. Choose :minmod, :vanleer, or :mc")
    end
end

function _fill_transport_ghosts!(qg, state, grid::Grid2D, lat_field, lon_field, t_utc::DateTime; bc_mode=:analytic, ng=2)
    nx, ny, ns = size(state)
    qg[(ng + 1):(ng + nx), (ng + 1):(ng + ny), :] .= state

    # West/east ghost columns.
    for s in 1:ns
        sp = SPECIES[s]
        for j in 1:ny
            jg = ng + j
            lat = lat_field[1, j]
            lon_w = lon_field[1, j]
            lon_e = lon_field[end, j]
            for layer in 1:ng
                qg[ng + 1 - layer, jg, s] = boundary_concentration(:west, sp, lon_w, lat, t_utc; bc_mode=bc_mode)
                qg[ng + nx + layer, jg, s] = boundary_concentration(:east, sp, lon_e, lat, t_utc; bc_mode=bc_mode)
            end
        end
    end

    # South/north ghost rows.
    for s in 1:ns
        sp = SPECIES[s]
        for i in 1:nx
            ig = ng + i
            lon = lon_field[i, 1]
            lat_s = lat_field[i, 1]
            lat_n = lat_field[i, end]
            for layer in 1:ng
                qg[ig, ng + 1 - layer, s] = boundary_concentration(:south, sp, lon, lat_s, t_utc; bc_mode=bc_mode)
                qg[ig, ng + ny + layer, s] = boundary_concentration(:north, sp, lon, lat_n, t_utc; bc_mode=bc_mode)
            end
        end
    end

    # Fill corners from nearest edges.
    for s in 1:ns
        qg[1:ng, 1:ng, s] .= qg[ng + 1, ng + 1, s]
        qg[(ng + nx + 1):end, 1:ng, s] .= qg[ng + nx, ng + 1, s]
        qg[1:ng, (ng + ny + 1):end, s] .= qg[ng + 1, ng + ny, s]
        qg[(ng + nx + 1):end, (ng + ny + 1):end, s] .= qg[ng + nx, ng + ny, s]
    end

    return nothing
end

"""
Advance advection by one explicit conservative step.

state: (nx, ny, ns) in mol/m^3
u_field, v_field: (nx, ny) in m/s (cell-centered)
"""
function apply_transport_step!(state, u_field, v_field, grid::Grid2D, dt_sec, t_utc::DateTime, lat_field, lon_field; limiter=:vanleer, bc_mode=:analytic)
    nx, ny, ns = size(state)
    ng = 2
    phi = get_limiter(limiter)

    qg = zeros(Float64, nx + 2 * ng, ny + 2 * ng, ns)
    _fill_transport_ghosts!(qg, state, grid, lat_field, lon_field, t_utc; bc_mode=bc_mode, ng=ng)

    sx = zeros(Float64, size(qg)...)
    sy = zeros(Float64, size(qg)...)

    nxg, nyg, _ = size(qg)
    for s in 1:ns
        for j in 2:(nyg - 1)
            for i in 2:(nxg - 1)
                sx[i, j, s] = phi(qg[i, j, s] - qg[i - 1, j, s], qg[i + 1, j, s] - qg[i, j, s])
                sy[i, j, s] = phi(qg[i, j, s] - qg[i, j - 1, s], qg[i, j + 1, s] - qg[i, j, s])
            end
        end
    end

    fx = zeros(Float64, nx + 1, ny, ns)
    fy = zeros(Float64, nx, ny + 1, ns)

    for s in 1:ns
        for j in 1:ny
            jg = ng + j
            for iface in 1:(nx + 1)
                il = ng + iface - 1
                ir = il + 1

                ql = qg[il, jg, s] + 0.5 * sx[il, jg, s]
                qr = qg[ir, jg, s] - 0.5 * sx[ir, jg, s]

                uf = if iface == 1
                    u_field[1, j]
                elseif iface == nx + 1
                    u_field[nx, j]
                else
                    0.5 * (u_field[iface - 1, j] + u_field[iface, j])
                end

                fx[iface, j, s] = (uf >= 0.0) ? (uf * ql) : (uf * qr)
            end
        end
    end

    for s in 1:ns
        for jface in 1:(ny + 1)
            jb = ng + jface - 1
            jt = jb + 1
            for i in 1:nx
                ig = ng + i

                qb = qg[ig, jb, s] + 0.5 * sy[ig, jb, s]
                qt = qg[ig, jt, s] - 0.5 * sy[ig, jt, s]

                vf = if jface == 1
                    v_field[i, 1]
                elseif jface == ny + 1
                    v_field[i, ny]
                else
                    0.5 * (v_field[i, jface - 1] + v_field[i, jface])
                end

                fy[i, jface, s] = (vf >= 0.0) ? (vf * qb) : (vf * qt)
            end
        end
    end

    dx_m = grid.dx_km * 1000.0
    dy_m = grid.dy_km * 1000.0

    for s in 1:ns
        for j in 1:ny
            for i in 1:nx
                div = (fx[i + 1, j, s] - fx[i, j, s]) / dx_m + (fy[i, j + 1, s] - fy[i, j, s]) / dy_m
                state[i, j, s] -= dt_sec * div
                state[i, j, s] = max(0.0, state[i, j, s])
            end
        end
    end

    return nothing
end

"""
Pull back one transport step adjoint: lambda_prev += (dT/dstate_prev)' * lambda_next.

Current support is limited to `limiter=:vanleer` and any boundary mode handled by
`_fill_transport_ghosts!` in the primal transport update.
"""
function transport_pullback_step!(lambda_prev, lambda_next, state_prev, u_field, v_field, grid::Grid2D, dt_sec, t_utc::DateTime, lat_field, lon_field; limiter=:vanleer, bc_mode=:analytic)
    nx, ny, ns = size(state_prev)
    size(lambda_next) == (nx, ny, ns) || error("lambda_next size mismatch")
    size(lambda_prev) == (nx, ny, ns) || error("lambda_prev size mismatch")
    limiter == :vanleer || error("transport_pullback_step! currently supports limiter=:vanleer only")

    ng = 2
    qg = zeros(Float64, nx + 2 * ng, ny + 2 * ng, ns)
    _fill_transport_ghosts!(qg, state_prev, grid, lat_field, lon_field, t_utc; bc_mode=bc_mode, ng=ng)

    sx = zeros(Float64, size(qg)...)
    sy = zeros(Float64, size(qg)...)
    nxg, nyg, _ = size(qg)
    for s in 1:ns
        for j in 2:(nyg - 1)
            for i in 2:(nxg - 1)
                sx[i, j, s] = vanleer(qg[i, j, s] - qg[i - 1, j, s], qg[i + 1, j, s] - qg[i, j, s])
                sy[i, j, s] = vanleer(qg[i, j, s] - qg[i, j - 1, s], qg[i, j + 1, s] - qg[i, j, s])
            end
        end
    end

    fx = zeros(Float64, nx + 1, ny, ns)
    fy = zeros(Float64, nx, ny + 1, ns)
    for s in 1:ns
        for j in 1:ny
            jg = ng + j
            for iface in 1:(nx + 1)
                il = ng + iface - 1
                ir = il + 1
                ql = qg[il, jg, s] + 0.5 * sx[il, jg, s]
                qr = qg[ir, jg, s] - 0.5 * sx[ir, jg, s]
                uf = if iface == 1
                    u_field[1, j]
                elseif iface == nx + 1
                    u_field[nx, j]
                else
                    0.5 * (u_field[iface - 1, j] + u_field[iface, j])
                end
                fx[iface, j, s] = (uf >= 0.0) ? (uf * ql) : (uf * qr)
            end
        end
    end

    for s in 1:ns
        for jface in 1:(ny + 1)
            jb = ng + jface - 1
            jt = jb + 1
            for i in 1:nx
                ig = ng + i
                qb = qg[ig, jb, s] + 0.5 * sy[ig, jb, s]
                qt = qg[ig, jt, s] - 0.5 * sy[ig, jt, s]
                vf = if jface == 1
                    v_field[i, 1]
                elseif jface == ny + 1
                    v_field[i, ny]
                else
                    0.5 * (v_field[i, jface - 1] + v_field[i, jface])
                end
                fy[i, jface, s] = (vf >= 0.0) ? (vf * qb) : (vf * qt)
            end
        end
    end

    dx_m = grid.dx_km * 1000.0
    dy_m = grid.dy_km * 1000.0
    fill!(lambda_prev, 0.0)

    bar_qg = zeros(Float64, nxg, nyg)
    bar_sx = zeros(Float64, nxg, nyg)
    bar_sy = zeros(Float64, nxg, nyg)
    bar_fx = zeros(Float64, nx + 1, ny)
    bar_fy = zeros(Float64, nx, ny + 1)
    lambda_z = zeros(Float64, nx, ny)

    for s in 1:ns
        fill!(bar_qg, 0.0)
        fill!(bar_sx, 0.0)
        fill!(bar_sy, 0.0)
        fill!(bar_fx, 0.0)
        fill!(bar_fy, 0.0)

        for j in 1:ny
            for i in 1:nx
                div = (fx[i + 1, j, s] - fx[i, j, s]) / dx_m + (fy[i, j + 1, s] - fy[i, j, s]) / dy_m
                z = state_prev[i, j, s] - dt_sec * div
                lambda_z[i, j] = (z > 0.0) ? lambda_next[i, j, s] : 0.0
                lambda_prev[i, j, s] += lambda_z[i, j]
            end
        end

        for j in 1:ny
            for i in 1:nx
                bar_div = -dt_sec * lambda_z[i, j]
                bar_fx[i + 1, j] += bar_div / dx_m
                bar_fx[i, j] -= bar_div / dx_m
                bar_fy[i, j + 1] += bar_div / dy_m
                bar_fy[i, j] -= bar_div / dy_m
            end
        end

        for j in 1:ny
            jg = ng + j
            for iface in 1:(nx + 1)
                il = ng + iface - 1
                ir = il + 1
                uf = if iface == 1
                    u_field[1, j]
                elseif iface == nx + 1
                    u_field[nx, j]
                else
                    0.5 * (u_field[iface - 1, j] + u_field[iface, j])
                end

                bfx = bar_fx[iface, j]
                if uf >= 0.0
                    bar_ql = uf * bfx
                    bar_qg[il, jg] += bar_ql
                    bar_sx[il, jg] += 0.5 * bar_ql
                else
                    bar_qr = uf * bfx
                    bar_qg[ir, jg] += bar_qr
                    bar_sx[ir, jg] += -0.5 * bar_qr
                end
            end
        end

        for jface in 1:(ny + 1)
            jb = ng + jface - 1
            jt = jb + 1
            for i in 1:nx
                ig = ng + i
                vf = if jface == 1
                    v_field[i, 1]
                elseif jface == ny + 1
                    v_field[i, ny]
                else
                    0.5 * (v_field[i, jface - 1] + v_field[i, jface])
                end

                bfy = bar_fy[i, jface]
                if vf >= 0.0
                    bar_qb = vf * bfy
                    bar_qg[ig, jb] += bar_qb
                    bar_sy[ig, jb] += 0.5 * bar_qb
                else
                    bar_qt = vf * bfy
                    bar_qg[ig, jt] += bar_qt
                    bar_sy[ig, jt] += -0.5 * bar_qt
                end
            end
        end

        for j in 2:(nyg - 1)
            for i in 2:(nxg - 1)
                a = qg[i, j, s] - qg[i - 1, j, s]
                b = qg[i + 1, j, s] - qg[i, j, s]
                dpa, dpb = _vanleer_partials(a, b)
                bs = bar_sx[i, j]
                ba = dpa * bs
                bb = dpb * bs
                bar_qg[i, j] += ba - bb
                bar_qg[i - 1, j] -= ba
                bar_qg[i + 1, j] += bb
            end
        end

        for j in 2:(nyg - 1)
            for i in 2:(nxg - 1)
                a = qg[i, j, s] - qg[i, j - 1, s]
                b = qg[i, j + 1, s] - qg[i, j, s]
                dpa, dpb = _vanleer_partials(a, b)
                bs = bar_sy[i, j]
                ba = dpa * bs
                bb = dpb * bs
                bar_qg[i, j] += ba - bb
                bar_qg[i, j - 1] -= ba
                bar_qg[i, j + 1] += bb
            end
        end

        for j in 1:ny
            for i in 1:nx
                lambda_prev[i, j, s] += bar_qg[ng + i, ng + j]
            end
        end
    end

    return nothing
end

# -----------------------------------------------------------------------------
# Chemistry operator (toy Stewart-1995-like mechanism)
# -----------------------------------------------------------------------------

"""
Compute chemistry tendency for a single cell in mol/m^3/s.
Rates use molecule/cm^3 internal units and convert back.
"""
function chemistry_tendency_cell!(dc, c, t_utc::DateTime, temp_k, lat_deg, lon_deg)
    fill!(dc, 0.0)

    # Convert concentrations from mol/m^3 -> molecule/cm^3.
    conv = AVOGADRO / 1e6
    y = clamp.(max.(c, 0.0) .* conv, 0.0, 1.0e16)

    no = y[SIDX[:NO]]
    no2 = y[SIDX[:NO2]]
    o3 = y[SIDX[:O3]]
    oh = y[SIDX[:OH]]
    ho2 = y[SIDX[:HO2]]
    voc = y[SIDX[:VOC]]
    ro2 = y[SIDX[:RO2]]
    hcho = y[SIDX[:HCHO]]
    co = y[SIDX[:CO]]

    T = (isfinite(temp_k) && temp_k > 150.0) ? temp_k : 280.0

    # Photolysis rates [s^-1], scaled by solar zenith.
    j_no2 = photolysis_rate(8.0e-3, t_utc, lat_deg, lon_deg)
    j_o3_eff = photolysis_rate(3.0e-6, t_utc, lat_deg, lon_deg)
    j_hcho = photolysis_rate(3.0e-5, t_utc, lat_deg, lon_deg)

    # Thermal rate constants [cm^3 molecule^-1 s^-1].
    k2 = 3.0e-12 * exp(-1500.0 / T)
    k4 = 1.1e-11
    k5 = 3.0e-13 * exp(460.0 / T)
    k6 = 1.5e-13
    k7 = 3.5e-12 * exp(250.0 / T)
    k8 = 1.0e-14 * exp(-715.0 / T)
    k9 = 1.7e-12 * exp(-940.0 / T)
    k10 = 1.0e-11
    k11 = 2.5e-12 * exp(300.0 / T)
    k12 = 3.8e-13 * exp(900.0 / T)
    k13 = 5.5e-12 * exp(125.0 / T)

    # Reaction rates [molecule cm^-3 s^-1].
    r1 = j_no2 * no2
    r2 = k2 * no * o3
    r3 = j_o3_eff * o3
    r4 = k4 * oh * no2
    r5 = k5 * ho2 * ho2
    r6 = k6 * co * oh
    r7 = k7 * ho2 * no
    r8 = k8 * ho2 * o3
    r9 = k9 * oh * o3
    r10 = k10 * voc * oh
    r11 = k11 * ro2 * no
    r12 = k12 * ro2 * ho2
    r13 = k13 * hcho * oh
    r14 = j_hcho * hcho

    # Species tendencies in molecule cm^-3 s^-1.
    d = zeros(Float64, NSPEC)

    d[SIDX[:NO]] += +r1 - r2 - r7 - r11
    d[SIDX[:NO2]] += -r1 + r2 - r4 + r7 + r11
    d[SIDX[:O3]] += +r1 - r2 - r3 - r8 - r9
    d[SIDX[:OH]] += +2r3 - r4 + r7 + r8 - r9 - r10 - r13
    d[SIDX[:HO2]] += -2r5 + r6 - r7 - r8 + r9 + r11 - r12 + r13 + 2r14
    d[SIDX[:VOC]] += -r10
    d[SIDX[:RO2]] += +r10 - r11 - r12
    d[SIDX[:HCHO]] += +r11 - r13 - r14
    d[SIDX[:CO]] += -r6 + r13 + r14
    d[SIDX[:HNO3]] += +r4

    # Convert molecule/cm^3/s back to mol/m^3/s.
    d ./= conv
    for s in 1:NSPEC
        if !isfinite(d[s])
            d[s] = 0.0
        end
    end

    dc .= d
    return nothing
end

"""
Chemistry forward-Euler with adaptive local substepping.

The cell-local step size is limited so no species changes by more than
`rel_change_limit` fraction of a local scale per micro-step.
"""
function apply_chemistry!(state, dt_sec, t_utc::DateTime, temp_k_field, lat_field, lon_field; nsub=4, rel_change_limit=0.15, max_local_substeps=200)
    nx, ny, ns = size(state)
    ns == NSPEC || error("state species dimension mismatch")

    dcdt = zeros(Float64, NSPEC)
    dts = dt_sec / nsub
    scale_floor = 1.0e-18

    for _ in 1:nsub
        for j in 1:ny
            for i in 1:nx
                remaining = dts
                nlocal = 0

                while remaining > 0.0 && nlocal < max_local_substeps
                    nlocal += 1
                    chemistry_tendency_cell!(dcdt, @view(state[i, j, :]), t_utc, temp_k_field[i, j], lat_field[i, j], lon_field[i, j])

                    # Compute local stable chemistry substep from relative tendency.
                    dt_lim = remaining
                    @inbounds for s in 1:NSPEC
                        rate = abs(dcdt[s])
                        if rate > 0.0 && isfinite(rate)
                            scale = max(state[i, j, s], scale_floor)
                            dt_s = rel_change_limit * scale / rate
                            dt_lim = min(dt_lim, dt_s)
                        end
                    end
                    dt_loc = min(remaining, max(dt_lim, 1.0e-6 * dts))

                    @inbounds for s in 1:NSPEC
                        newv = state[i, j, s] + dt_loc * dcdt[s]
                        if !isfinite(newv)
                            newv = 0.0
                        end
                        state[i, j, s] = max(0.0, newv)
                    end
                    remaining -= dt_loc
                end
            end
        end
    end

    return nothing
end

# -----------------------------------------------------------------------------
# Receptor objective
# -----------------------------------------------------------------------------

"""
Bilinear interpolation of species concentration from local lat/lon gridded fields.
lat_field and lon_field are cell-centered arrays (nx, ny).
"""
function bilinear_receptor_value(field2d, lat_field, lon_field, lat0, lon0)
    i0, i1, j0, j1, w00, w10, w01, w11 = _bilinear_receptor_stencil(lat_field, lon_field, lat0, lon0)
    f00 = field2d[i0, j0]
    f10 = field2d[i1, j0]
    f01 = field2d[i0, j1]
    f11 = field2d[i1, j1]

    return w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11
end

function _bilinear_receptor_stencil(lat_field, lon_field, lat0, lon0)
    nx, ny = size(lat_field)
    lon_vec = lon_field[:, 1]
    lat_vec = lat_field[1, :]

    # Clamp to domain limits for local non-cyclic interpolation.
    lon_t = clamp(lon0, min(lon_vec[1], lon_vec[end]), max(lon_vec[1], lon_vec[end]))
    lat_t = clamp(lat0, min(lat_vec[1], lat_vec[end]), max(lat_vec[1], lat_vec[end]))

    dlon = lon_vec[2] - lon_vec[1]
    dlat = lat_vec[2] - lat_vec[1]

    fx_raw = (lon_t - lon_vec[1]) / dlon
    fy_raw = (lat_t - lat_vec[1]) / dlat

    i0 = clamp(floor(Int, fx_raw) + 1, 1, nx - 1)
    j0 = clamp(floor(Int, fy_raw) + 1, 1, ny - 1)
    i1 = i0 + 1
    j1 = j0 + 1

    fx = fx_raw - floor(fx_raw)
    fy = fy_raw - floor(fy_raw)

    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy

    return i0, i1, j0, j1, w00, w10, w01, w11
end

"""
Accumulate pullback of a receptor interpolation objective into state adjoint.

Adds `weight * d(receptor_value)/d(state)` for one species to `lambda_state`.
"""
function accumulate_objective_pullback!(lambda_state, species_symbol::Symbol, weight, lat_field, lon_field, receptor_lat, receptor_lon)
    i0, i1, j0, j1, w00, w10, w01, w11 = _bilinear_receptor_stencil(lat_field, lon_field, receptor_lat, receptor_lon)
    sidx = SIDX[species_symbol]

    lambda_state[i0, j0, sidx] += weight * w00
    lambda_state[i1, j0, sidx] += weight * w10
    lambda_state[i0, j1, sidx] += weight * w01
    lambda_state[i1, j1, sidx] += weight * w11
    return nothing
end

mutable struct DailyMeanAccumulator
    integral::Float64
    elapsed_sec::Float64
end

DailyMeanAccumulator() = DailyMeanAccumulator(0.0, 0.0)

function accumulate_objective!(acc::DailyMeanAccumulator, state, species_symbol::Symbol, dt_sec, lat_field, lon_field, receptor_lat, receptor_lon)
    sidx = SIDX[species_symbol]
    val = bilinear_receptor_value(@view(state[:, :, sidx]), lat_field, lon_field, receptor_lat, receptor_lon)
    acc.integral += val * dt_sec
    acc.elapsed_sec += dt_sec
    return nothing
end

objective_daily_mean(acc::DailyMeanAccumulator) = acc.integral / max(acc.elapsed_sec, eps())

# -----------------------------------------------------------------------------
# External forcing readers and interpolation
# -----------------------------------------------------------------------------

"""
Load U, V from A3dyn and T from I3 and interpolate them to model time.

Arguments:
- u_field, v_field, t_field: target arrays (nx, ny)
- dt_utc: simulation time
- paths: NamedTuple with keys
    a3dyn, i3, lon_field, lat_field
  and optional key lev_index (default 1).

Notes:
- A3dyn U,V are time-averaged samples in the source product.
- I3 T is treated as instantaneous snapshots.
"""
function load_forcing_fields!(u_field, v_field, t_field, dt_utc::DateTime, paths)
    a3_path = paths.a3dyn
    i3_path = paths.i3
    lon_tgt = paths.lon_field
    lat_tgt = paths.lat_field
    lev = get(paths, :lev_index, 1)

    ds_dyn = NCDataset(a3_path)
    ds_i3 = NCDataset(i3_path)

    try
        lon_src = ds_dyn["lon"][:]
        lat_src = ds_dyn["lat"][:]
        times_dyn = ds_dyn["time"][:]
        times_i3 = ds_i3["time"][:]

        i1, i2, w = _time_bracket_weights(times_dyn, dt_utc)
        u1 = ds_dyn["U"][:, :, lev, i1]
        u2 = ds_dyn["U"][:, :, lev, i2]
        v1 = ds_dyn["V"][:, :, lev, i1]
        v2 = ds_dyn["V"][:, :, lev, i2]
        u_src = (1.0 - w) .* u1 .+ w .* u2
        v_src = (1.0 - w) .* v1 .+ w .* v2

        k1, k2, wt = _time_bracket_weights(times_i3, dt_utc)
        t1 = ds_i3["T"][:, :, lev, k1]
        t2 = ds_i3["T"][:, :, lev, k2]
        t_src = (1.0 - wt) .* t1 .+ wt .* t2

        _remap_to_model_grid!(u_field, u_src, lon_src, lat_src, lon_tgt, lat_tgt)
        _remap_to_model_grid!(v_field, v_src, lon_src, lat_src, lon_tgt, lat_tgt)
        _remap_to_model_grid!(t_field, t_src, lon_src, lat_src, lon_tgt, lat_tgt)
    finally
        close(ds_dyn)
        close(ds_i3)
    end

    return nothing
end

"""
Load and remap monthly CEDS NO emissions fields (NO_tra + NO_ene + NO_ind).

baseline_no_hourly target shape: (nx, ny, 24), units kg/m^2/s.
For v1, monthly means are copied to each hour in the day.

Keyword options:
- month_index: 1..12 (default 1 for January)
- lon_field, lat_field: optional precomputed model lon/lat arrays

"""
function load_no_emissions_hourly!(baseline_no_hourly, grid::Grid2D, path_ceds::AbstractString; month_index=1, lon_field=nothing, lat_field=nothing)
    if lon_field === nothing || lat_field === nothing
        lon_field, lat_field = build_lonlat_fields(grid)
    end

    ds = NCDataset(path_ceds)
    try
        lon_src = ds["lon"][:]
        lat_src = ds["lat"][:]

        nmonth = size(ds["NO_tra"], 3)
        m = clamp(month_index, 1, nmonth)

        no_src = ds["NO_tra"][:, :, m] .+ ds["NO_ene"][:, :, m] .+ ds["NO_ind"][:, :, m]

        no_model = zeros(Float64, grid.nx, grid.ny)
        _remap_to_model_grid!(no_model, no_src, lon_src, lat_src, lon_field, lat_field)

        for h in 1:24
            baseline_no_hourly[:, :, h] .= no_model
        end
    finally
        close(ds)
    end

    return nothing
end

# -----------------------------------------------------------------------------
# Controls helpers (for inverse and gradient-check workflows)
# -----------------------------------------------------------------------------

"""
Number of NO control parameters (nx * ny * 24 hourly bins).
"""
alpha_control_size(grid::Grid2D) = grid.nx * grid.ny * 24

"""
Pack alpha controls (nx, ny, 24) into a vector.
"""
function pack_alpha_controls(alpha_controls, grid::Grid2D)
    size(alpha_controls) == (grid.nx, grid.ny, 24) || error("alpha_controls must have size (nx, ny, 24)")
    return vec(copy(alpha_controls))
end

"""
Unpack a control vector into alpha controls (nx, ny, 24).
"""
function unpack_alpha_controls(alpha_vec, grid::Grid2D)
    length(alpha_vec) == alpha_control_size(grid) || error("alpha_vec length must be nx*ny*24")
    return reshape(copy(alpha_vec), grid.nx, grid.ny, 24)
end

#-----------------------------------------------------------------------------
# State saving
#-----------------------------------------------------------------------------

function save_state(path, state, t, lat_field, lon_field, air_mol_m3)
    ds = NCDataset(path, "c")
    try
        defDim(ds, "x", size(lon_field, 1))
        defDim(ds, "y", size(lon_field, 2))
        #defDim(ds, "species", size(state, 3))
        defDim(ds, "time", 1)

        #var_lon = defVar(ds, "lon", Float64, ("x",))
        #var_lat = defVar(ds, "lat", Float64, ("y",))
        var_lon = defVar(ds, "lon", Float64, ("y","x"))
        var_lat = defVar(ds, "lat", Float64, ("y","x"))
        var_time = defVar(ds, "time", Float64, ("time",))
        #var_state = defVar(ds, "state", Float64, ("lon", "lat", "species"))
        #var_state[:, :, :] = state

        var_lon[:,:] = lon_field[:, :]'
        var_lat[:,:] = lat_field[:, :]'
        var_time[1] = Dates.value(t) / 1000.0
        # Save out species data in v/v units by dividing by air mol/m^3.
        for s in 1:size(state, 3)
            var_state = defVar(ds, string(SPECIES[s]), Float64, ("time", "y", "x"))
            var_state[1,:, :] = (state[:, :, s] ./ air_mol_m3[:, :])'
        end
    finally
        close(ds)
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Coupled forward driver
# -----------------------------------------------------------------------------

"""
Run one-day coupled forward model (transport + NO emissions + chemistry).

This is a v1 integration driver focused on correctness and clarity.
"""
function run_forward_day(; grid=make_grid(),
    config=default_config(),
    forcing_paths,
    ceds_no_path::AbstractString,
    alpha_controls=nothing,
    wind_mode::Symbol=:raw_met,
    objective_species::Symbol=:O3,
    bc_mode::Symbol=:analytic,
    limiter::Symbol=:vanleer,
    lev_index::Int=1,
    chemistry_substeps::Int=4,
    chemistry_rel_change_limit::Float64=0.15,
    verbose::Bool=false,
    save_dir::Union{Nothing,AbstractString}=nothing,
    save_interval_seconds::Float64=3600.0)

    lon_field, lat_field = build_lonlat_fields(grid)
    state = initialize_state(grid; perturb=true)

    # Load NO baseline monthly mean and apply hourly controls (unity by default).
    baseline_no_hourly = zeros(Float64, grid.nx, grid.ny, 24)
    load_no_emissions_hourly!(baseline_no_hourly, grid, ceds_no_path; month_index=1, lon_field=lon_field, lat_field=lat_field)
    alpha = if alpha_controls === nothing
        ones(Float64, grid.nx, grid.ny, 24)
    else
        size(alpha_controls) == (grid.nx, grid.ny, 24) || error("alpha_controls must have size (nx, ny, 24)")
        alpha_controls
    end

    u = zeros(Float64, grid.nx, grid.ny)
    v = zeros(Float64, grid.nx, grid.ny)
    tfield = zeros(Float64, grid.nx, grid.ny)
    pressure = zeros(Float64, grid.nx, grid.ny) # Placeholder if needed for chemistry
    pressure[:, :] .= 101325.0 # Pa, constant surface pressure for chemistry
    air_mol_m3 = pressure ./ (GAS_CONSTANT * tfield) # mol/m^3, for chemistry if needed

    paths = (a3dyn=forcing_paths.a3dyn, i3=forcing_paths.i3, lon_field=lon_field, lat_field=lat_field, lev_index=lev_index)

    acc = DailyMeanAccumulator()
    t = config.t_start
    dt = config.dt_sec
    nsteps = 0
    t_wall_start = time()

    while t < config.t_end
        dt_loc = min(dt, Float64(Dates.value(config.t_end - t)) / 1000.0)

        load_forcing_fields!(u, v, tfield, t, paths)
        if wind_mode == :mass_consistent_projected
            project_wind_mass_consistent!(u, v, grid)
        elseif wind_mode != :raw_met
            error("Unknown wind_mode $wind_mode. Choose :raw_met or :mass_consistent_projected")
        end

        # Update derived meteorology
        air_mol_m3 .= pressure ./ (GAS_CONSTANT * tfield)

        apply_transport_step!(state, u, v, grid, dt_loc, t, lat_field, lon_field; limiter=limiter, bc_mode=bc_mode)
        apply_emissions_no!(state, baseline_no_hourly, alpha, dt_loc, Dates.hour(t); mixing_depth_m=config.mixing_depth_m)
        apply_chemistry!(state, dt_loc, t, tfield, lat_field, lon_field; nsub=chemistry_substeps, rel_change_limit=chemistry_rel_change_limit)
        accumulate_objective!(acc, state, objective_species, dt_loc, lat_field, lon_field, config.receptor_lat_deg, config.receptor_lon_deg)

        if verbose && nsteps % 10 == 0
            wall_min = (time() - t_wall_start) / 60.0
            sim_hrs  = acc.elapsed_sec / 3600.0
            rate_str = wall_min > 0.005 ? "$(round(sim_hrs / wall_min, digits=1)) sim-h/wall-min" : "—"
            obj_str  = @sprintf("%.3e", objective_daily_mean(acc))
            println("t = $(t) | obj = $(obj_str) mol/m^3 | $(rate_str)")
        end

        if save_dir !== nothing && nsteps % Int(save_interval_seconds / dt) == 0
            ts = Dates.format(t, "yyyymmdd_HHMMSS")
            save_state(joinpath(save_dir, "state_$(ts).nc"), state, t, lat_field, lon_field, air_mol_m3)
        end

        t += Millisecond(round(Int, 1000 * dt_loc))
        nsteps += 1
    end

    return (
        state=state,
        daily_mean=objective_daily_mean(acc),
        steps=nsteps,
        elapsed_hours=acc.elapsed_sec / 3600.0,
        lon_field=lon_field,
        lat_field=lat_field,
    )
end

"""
Evaluate daily-mean objective for a flattened alpha control vector.

This is a convenience wrapper for inverse and gradient-check scripts.
"""
function objective_for_alpha_vector(alpha_vec;
    grid=make_grid(),
    config=default_config(),
    forcing_paths,
    ceds_no_path::AbstractString,
    wind_mode::Symbol=:raw_met,
    objective_species::Symbol=:O3,
    bc_mode::Symbol=:analytic,
    limiter::Symbol=:vanleer,
    lev_index::Int=1,
    chemistry_substeps::Int=4,
    chemistry_rel_change_limit::Float64=0.15)

    alpha_controls = unpack_alpha_controls(alpha_vec, grid)
    out = run_forward_day(
        grid=grid,
        config=config,
        forcing_paths=forcing_paths,
        ceds_no_path=ceds_no_path,
        alpha_controls=alpha_controls,
        wind_mode=wind_mode,
        objective_species=objective_species,
        bc_mode=bc_mode,
        limiter=limiter,
        lev_index=lev_index,
        chemistry_substeps=chemistry_substeps,
        chemistry_rel_change_limit=chemistry_rel_change_limit,
    )
    return out.daily_mean
end

end # module
