# Halo_scan_design
Tool for Halo lidar scan design

# Description
This package allows the user to identify the optimal scan parameters for a Halo lidar scans of different geometry. The applications are restricted to 2-D (PPI or RHI) and volumetric scans that use azimuth and elevation with a Cartesian configuration. The limits of azimuth and elevation angles must also be symmetrical. Future releases may address more complex scan geometry.

The user defines the inputs in an Excel table. Then the LiSBOA_Pareto_design is run to calculate the three objective functions. Finally, the Pareto front and the full geometry of a selected scan can be visualized through the LiSBOA_design_postpro.

# Inputs
The inputs of the scan optimizer are defined in an Excel table like the data/Scan_info.xlsx. The full list of inputs is:
- Run [boolean]: if TRUE, the entry is processed; if FALSE, it is skipped
- Scan name [string]: the name of the scan
- vec_dazi [deg]: the list of azimuth resolutions to be tested
- vec_azi_max [deg]: the list azimuth limits to be tested. The tested azimuth will go from -azi_max to +azi_max with a dazi step
- vec_dele [deg]: the list of elevation resolution to be tested
- vec_ele_max [deg]: the list of elevation limits to be tested. The tested elevation will go from -ele_max to +ele_max with a dele step
- d_cartestian [boolean]: if TRUE, the azimuth and elevation resolutions are multiplied cartesianly so that all the possible combination are tested; if FALSE, the azimuth and elevation resolutions are tested in pairs (they must have the same legnth in this case)
- max_cartestian [boolean]: if TRUE, the azimuth and elevation limits are multiplied cartesianly so that all the possible combination are tested; if FALSE, the azimuth and elevation limits are tested in pairs (they must have the same legnth in this case)
- Dn0 [m]: vector of fundamental half-wavelengths
- dr [m]: range gate length of the lidar
- rmin [m]: minimum range of the lidar
- rmax [m]: maximum range of the lidar
- mins [m]: lower boundary of the domain in x,y,z
- maxs [m]: upper boundary of the domain in x,y,z
- sigma [multiple of fundamental half-wavelengths]: smoothing parameter of the LiSBOA (1/4 recommended)
- T_tot [s]: total duration of the experiment
- tau [s]: mean integral time scale of the flow
- sampl_time [s]: sampling time of the lidar needed to measure a beam
- U [m/s]: mean wind speed

The LiSBOA_Pareto_design only requires as input the path to the input table. 

The LiSBOA_design_postpro needs as inputs:
- root [string]: the location of the folder containing the Pareto front results
- source [string]: the name of the specific scan to plot
- sel [deg]: the set of azimuth resolution, azimuth limit, elevation resolution, elevation limit whose geometry is plot in details
- save_fig [boolean]: if TRUE, figures are automatically saved in figure/date_time_input/scan.

# Outputs
LiSBOA_Pareto_v3 produces netcdf fiels that are saved in data/{date}_{time}_input.nc.

LiSBOA_postpro_v3 produces plots that can be automatically saved.

# References
https://amt.copernicus.org/articles/14/2065/2021/

https://iopscience.iop.org/article/10.1088/1742-6596/2505/1/012048/meta
