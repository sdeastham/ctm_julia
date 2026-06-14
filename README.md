Very simple CTM in Julia.

Example script to run:

```
using Serialization, Dates
cd(raw"e:\OneDrive - Imperial College London\Code\Julia")
mkpath("archive")

include(raw"ctm_julia\ctm_v1_core.jl")
using .CTMV1Core

archive_dir = "E:/Data/ModelOutput/CTM_v1/forward_test"
mkpath(archive_dir)

println("Running forward test...")
grid = CTMV1Core.make_grid(dx_km=5.0, dy_km=5.0, xmin_km=-250, xmax_km=250, ymin_km=-500, ymax_km=500, lon0_deg=-2, lat0_deg=54)
dt = 30.0
dt_store = 300.0
cfg  = CTMV1Core.RunConfig(DateTime(2015,1,1,0,0,0), DateTime(2015,1,2,0,0,0), dt, 51.498926, -0.174777, CTMV1Core.DEFAULT_MIXING_DEPTH_M)

out = CTMV1Core.run_forward_day(
    grid=grid, config=cfg,
    forcing_paths=(a3dyn=raw"E:\Data\ExtData\GEOS_0.5x0.625\MERRA2\2015\01\01\MERRA2.20150101.A3dyn.05x0625.nc4",
                   i3=raw"E:\Data\ExtData\GEOS_0.5x0.625\MERRA2\2015\01\01\MERRA2.20150101.I3.05x0625.nc4"),
    ceds_no_path=raw"E:\Data\ExtData\HEMCO\CEDS\v2024-06\2015\CEDS_NO_0.1x0.1_2015.nc",
    limiter=:vanleer, bc_mode=:analytic, verbose=true, save_interval_seconds=dt_store, save_dir=archive_dir
)

serialize(joinpath(archive_dir, "forward_state.jls"), out.state)           # 3D concentration grid
serialize(joinpath(archive_dir, "forward_lonlat.jls"), (out.lon_field, out.lat_field))
open(joinpath(archive_dir, "forward_summary.txt"),"w") do io
    println(io, "daily_mean=", out.daily_mean)
    println(io, "steps=", out.steps)
    println(io, "elapsed_hours=", out.elapsed_hours)
end

println("Forward test completed. State, lon/lat fields, and summary saved to ", archive_dir)
```

Example script to view the output:

```
include(raw"ctm_julia\ctm_v1_core.jl")
using .CTMV1Core
using CairoMakie
using Dates
using GeoMakie
using NCDatasets
using Serialization

archive_dir = "E:/Data/ModelOutput/CTM_v1/forward_test"
state = deserialize(joinpath(archive_dir, "forward_state.jls"))
lon_field, lat_field = deserialize(joinpath(archive_dir, "forward_lonlat.jls"))
open(joinpath(archive_dir, "forward_summary.txt"),"r") do io
    for line in eachline(io)
        println(line)
    end
end

#heatmap(lon_field, lat_field, state[:, :, 1]', title="NO Concentration (mol/m3)", xlabel="Longitude", ylabel="Latitude", aspect_ratio=:equal, colorbar_title="mol/m3")
#heatmap(lon_field, lat_field, state[:, :, 1]')

lat0 = lat_field[1]
lon0 = lon_field[1]
proj = "+proj=eqc +lat_ts=$(lat0) +lon_0=$(lon0)"

lon_vec = lon_field[:, cld(size(lon_field, 2), 2)]
lat_vec = lat_field[cld(size(lat_field, 1), 2), :]
no_field = permutedims(state[:, :, 1])

fig = Figure(size=(1500, 650))

n_species = size(state, 3)
n_species = 3
for species in 1:n_species
    ga = GeoAxis(fig[1, species*2 - 1]; dest=proj, title=species == 1 ? "NO (mol/m3)" : species == 2 ? "NO2 (mol/m3)" : "O3 (mol/m3)")
    cax = fig[1, species*2]
    hm = heatmap!(ga, lon_vec, lat_vec, permutedims(state[:, :, species])'; colormap=:viridis)
    Colorbar(cax, hm, label="mol/m3")
    lines!(ga, GeoMakie.coastlines(), color = :black, linewidth = 1.2)
    limits!(ga, (minimum(lon_vec), maximum(lon_vec)), (minimum(lat_vec), maximum(lat_vec)))
end

out_fig = joinpath(archive_dir, "forward_test_final_state.png")
save(out_fig, fig)
println("Saved figure: " * out_fig)

# Now animate the entire simulation using the saved-out states
file_list = filter(f -> endswith(f, ".nc"), readdir(archive_dir))
sort!(file_list)  # Ensure files are in chronological order

fig = Figure(size=(1500, 650))
ga1 = GeoAxis(fig[1, 1]; dest=proj, title="NO (ppb)")
cax1 = fig[1, 2]
ga2 = GeoAxis(fig[1, 3]; dest=proj, title="O3 (ppb)")
cax2 = fig[1, 4]

no_data = zeros(size(state, 1), size(state, 2))
o3_data = zeros(size(state, 1), size(state, 2))
no_obs = Observable(no_data)
o3_obs = Observable(o3_data)
hm1 = heatmap!(ga1, lon_vec, lat_vec, no_obs; colormap=:viridis, colorrange=(0.0, 2.5))
Colorbar(cax1, hm1, label="ppb")

hm2 = heatmap!(ga2, lon_vec, lat_vec, o3_obs; colormap=:plasma, colorrange=(0.0, 120.0))
Colorbar(cax2, hm2, label="ppb")

for ga in (ga1, ga2)
    lines!(ga, GeoMakie.coastlines(), color=:black, linewidth=1.2)
    limits!(ga, (minimum(lon_vec), maximum(lon_vec)), (minimum(lat_vec), maximum(lat_vec)))
end

out_anim = joinpath(archive_dir, "forward_test_animation.mp4")
# Want to show one hour per second
dt_store = 300.0
framerate = Int(3600.0 / dt_store)

record(fig, out_anim, 1:length(file_list); framerate=framerate) do step
    file_path = joinpath(archive_dir, file_list[step])
    ds = NCDataset(file_path)
    # Convert from mol/mol to ppb
    # Dimensions are (time, y, x)
    no_ppb = ds["NO"][1, :, :] * 1.0e9
    o3_ppb = ds["O3"][1, :, :] * 1.0e9
    close(ds)
    no_obs[] = no_ppb'
    o3_obs[] = o3_ppb'
end

println("Saved animation: " * out_anim)
```
