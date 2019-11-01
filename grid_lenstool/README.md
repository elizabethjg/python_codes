# grid_lenstool
Scripts to make grid for non-parametric lenstool.
The following scripts are included:

  ## fit_hex.py: make multiscale (or regular) grid from input density/mass map.
  
  ### USAGE:
  
  `fit_hex.py fits_map T [-ll LL] [-hl HL] [-r R] [--reg]`
 
  Positional arguments:
  
    fits_map     Fits file with density map, input to create the grid
    T            Threshold on the density map to split triangles

  Optional arguments:
  
    -h, --help   show this help message and exit
    -ll LL       Maximum number of triangle split, default=7
    -hl HL       Minimum number of triangle split, default=1
    -r R         Ratio rcut/rc, default=3
    --reg        Creates regular grid, with lowlim triangle splits,
                 default=False
  
  
  ### OUTPUT:
  **sdens_tT.dat**: catalogue where each line coresponds to a grid node, the columns are
  Id x y rc rcore 0
  The coordinates x y are given in arsec relative to the reference RA Dec written in the header.
  
  ### DETAILS:
  It starts with a hexagonal grid with seven nodes:
  
             --------
            / \ 1  / \
           / 2 \  / 0 \
          --------------
           \ 3 /  \ 5 /
            \ / 4  \ /
             --------
 
  Each triangular region will then be split into 4 new subtriangles if there is a least one pixel with a value larger than
  the chosen threshold T: 
  
                   2
                 /   \
                / [2] \
               /       \
            5 ---------- 4
           /   \       /   \
          / [0] \ [3] / [1] \
         /       \   /       \
        0 -------  3 -------- 1
        
  This process is recursively applied to each triangle until the maximum number of splits LL in reached.
  
  The minimum (HL) and maximum (LL) number of splits can be adjusted and are by default set to 1 and 7 respectively.
  To each node of the grid is associated a mass "pixel" described by a PIEMD profile with a core radius equivalent to 
  the size of the largest surrounding triangle. The ratio cut radius/core radius of the PIEMD profiles R is set by 
  default to 3.
  
  If the option --reg is set, the output will be a regular grid where each triangle has been split LL times.



  ## mk_holes.py: Add mask column to sdens catalogue corresponding to input DS9 region
  This script allows to filter out the grid potentials that are located in the strong lensing region.


  ### USAGE:
  `mk_holes.py [-h] sdens region`
  
  Positional arguments:
  
    sdens       Catalogue of grid potentials, built with fit_hex.py
    region      DS9 region file to filter

  Optional arguments:
  
    -h, --help  show this help message and exit

  ### OUTPUT: 
  **sdens_holes.dat**: same as sdens_tT.dat, but with additional last column, containing 1 if the potential is in the SL 
  region to exclude, 0 otherwise

  The input region file needs to be a DS9 polygon region saved in WCS degree coordinates.
  
  
  
  ## build_hex.py: Create Lenstool input file for grid mode
  
  ### USAGE:
  `build_hex.py [-h] [--mask] sdens cmcat zl srccat`
  
  Positional arguments:
  
    sdens       Catalogue of grid potentials, built with fit_hex.py
    cmcat       Galaxy catalogue
    zl          Lens redshift
    srccat      Source catalogue

  Optional arguments:
  
    -h, --help  show this help message and exit
    --mask      Define if there is a mask column in sdens to show the SL region
                
  ### OUPUT:
  **hex.par**: Lenstool input file containing the grid potentials.
  !!! The value of the parameters in the potfile section need to be updated by hand.
  If any FIXED parametric potential is added, don't forget to increase the value of nmsgrid and nlentille in the grille
  section.
                
