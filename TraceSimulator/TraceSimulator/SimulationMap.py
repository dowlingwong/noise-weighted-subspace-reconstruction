import sys
import os
import numpy as np
from .PolygonBinning import *
from .CylindricalBinning import *

class SimulationMap():
    """
    Class to manage simulation data and map it using polygon binning.

    Attributes:
        file_path (str): Path to the .npz file containing simulation data.
        num_LAMCAL (int): Number of LAMCAL bins.
        per_LAMCAL (bool): Flag indicating whether to use per LAMCAL binning.
        is_distribution (bool): Flag indicating whether the data is a distribution.
    """
        
    def __init__(self, file_path=None, num_LAMCAL=56, per_LAMCAL=True, is_distribution=False, shift_theta=0., is_cyl=True):
        """
        Initialize the SimulationMap object.

        Args:
            file_path (str, optional): Path to the .npz file. Defaults to None.
            num_LAMCAL (int, optional): Number of LAMCAL bins. Defaults to 56.
            per_LAMCAL (bool, optional): Flag for per LAMCAL binning. Defaults to True.
            is_distribution (bool, optional): Flag for distribution data. Defaults to False.
        """        
        self.file_path = file_path
        self.num_LAMCAL = num_LAMCAL
        self.per_LAMCAL = per_LAMCAL
        self.is_distribution = is_distribution
        self._shift_theta = shift_theta
        self.binning_type = 'cyl' if is_cyl else 'poly'
        if is_cyl:
            self.binning_class = CylindricalBinning
            self.binning_dist_class = CylindricalBinningDistribution
        else:
            self.binning_class = PolygonBinning
            self.binning_dist_class = PolygonBinningDistribution
        self._load_file()
        
        
    def _load_file(self):
        """
        Load the data from the specified file path.
        """        
        if self.file_path is None:
            raise ValueError("File path not specified")
        if not self.file_path.endswith('.npz'):
            raise ValueError("Specified file should be .npz format.")
            
        tmp = np.load(self.file_path)
        self.dimension = tmp['dimension']
        self._data = tmp['data']
        if self.is_distribution:
            self._set_distribution_map()
        else:
            self._set_map()
        
    
    def _set_map(self):
        """
        Set up the binning map based on the loaded data.
        """        
        if self.binning_type == 'poly':
            args_bin = (self.dimension[0], *np.asarray(self.dimension[1:5], dtype=int), [self.dimension[5], self.dimension[6]])
        elif self.binning_type == 'cyl':
            args_bin = (self.dimension[0], *np.asarray(self.dimension[1:3], dtype=int), [self.dimension[3], self.dimension[4]])
        if self.per_LAMCAL:
            self.map = [self.binning_class(*args_bin, shift_theta=self._shift_theta) for _ in range(self.num_LAMCAL)]
            for iLAMCAL in range(self.num_LAMCAL):
                tmp = self._data[iLAMCAL].copy()
                self.map[iLAMCAL].set_distribution(tmp)
        else:
            self.map = self.binning_class(*args_bin)
            self.map.set_distribution(np.sum(self._data, axis=0))
            
            
    def _set_distribution_map(self):
        """
        Set up the binning distribution map based on the loaded data.
        """        
        if self.binning_type == 'poly':
            args_bin = (self.dimension[0], *np.asarray(self.dimension[1:6], dtype=int), [self.dimension[6], self.dimension[7]])
        elif self.binning_type == 'cyl':
            args_bin = (self.dimension[0], *np.asarray(self.dimension[1:4], dtype=int), [self.dimension[4], self.dimension[5]])
        self.map = [self.binning_dist_class(*args_bin, shift_theta=self._shift_theta) for _ in range(self.num_LAMCAL)]
        for iLAMCAL in range(self.num_LAMCAL):
            tmp = self._data[iLAMCAL].copy()
            self.map[iLAMCAL].set_distribution(tmp)

            
    def get_value(self, x, y, z, LAMCAL_ID=None):
        """
        Get the value from the binning map for given coordinates.

        Args:
            x (float or array): X-coordinate(s).
            y (float or array): Y-coordinate(s).
            z (float or array): Z-coordinate(s).
            LAMCAL_ID (int, optional): LAMCAL ID. Defaults to None.

        Returns:
            array: Value(s) from the binning map.
        """        
        if LAMCAL_ID is None:
            return self.map(x, y, z)
        else:
            return self.map[LAMCAL_ID](x, y, z)
            
            
    def __call__(self, x, y ,z, LAMCAL_ID=None):
        """
        Callable method to get the value from the binning map for given coordinates.

        Args:
            x (float or array): X-coordinate(s).
            y (float or array): Y-coordinate(s).
            z (float or array): Z-coordinate(s).
            LAMCAL_ID (int, optional): LAMCAL ID. Defaults to None.

        Returns:
            array: Value(s) from the binning map.
        """        
        return self.get_value(x, y, z, LAMCAL_ID)
    
    
    def get_LAMCAL_ID(self, num, LAMCAL_sub=19, N_side=0, N_rows=0):
        """
        Get the LAMCAL ID for a given number.

        Args:
            num (int): The number to get the LAMCAL ID for.
            LAMCAL_sub (int, optional): Submerged LAMCALs. Defaults to 9.
            N_side (int, optional): Number of sides. Defaults to 0.
            N_rows (int, optional): Number of rows. Defaults to 0.

        Returns:
            int: Corresponding LAMCAL ID.
        """    
        for j, min_side in enumerate(LAMCAL_cap + N_side * np.arange(N_rows)):
            if (num >= min_side) & (num < min_side + N_side): 
                return 1000 + num - min_side + 100 * j
        return num if num < LAMCAL_sub else 2000 + num - (LAMCAL_cap - N_side * N_rows)
        
        
    def sample(self, x, y, z, LAMCAL_ID, size):
        """
        Sample data from the statistical distribution if is_distribution is True,
        otherwise it returns the value as in get_value(x, y, z, LAMCAL_ID) attribute.

        Args:
            x (float or array): X-coordinate(s).
            y (float or array): Y-coordinate(s).
            z (float or array): Z-coordinate(s).
            size (int): Number of samples to generate.

        Returns:
            array: Sampled data.
        """        
        if not is_distribution:
            return self.get_value(x, y, z, LAMCAL_ID)
        else:
            return self.map[LAMCAL_ID].sample(x, y, z, size)