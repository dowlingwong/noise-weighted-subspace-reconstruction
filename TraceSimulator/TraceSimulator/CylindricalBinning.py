import sys
import os

import numpy as np
import pandas as pd
import uproot
import scipy
import tqdm
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import matplotlib.patches as patches
from matplotlib.colors import LogNorm

class CylindricalBinning(object):
    """
    A class for binning data in a 3D cylindrical coordinate system.

    This class allows you to define a binning scheme based on 
    a cylindrical coordinate system. It provides functionalities
    for calculating bin indices for given data points, calculating 
    distributions over the bins, and plotting the distributions.

    Attributes:
        r (float): The radius of the cylinder.
        Nr (int): The number of radial bins.
        Nz (int): The number of z bins.
        z_lim (list): The limits of the z-axis.
        data (numpy.ndarray, optional): The data array (x, y, z, data).
        distribution (numpy.ndarray): The distribution calculated over the bins.
        area_bins (numpy.ndarray): The area of each bin.
        corners (numpy.ndarray): The corners of each bin.
        centers (numpy.ndarray): The centers of each bin.
    """
    
    def __init__(self, r, N_radial, N_z, z_lim, shift_theta=None):
        """
        Initializes a CylindricalBinning object.

        Args:
            r (float): The radius of the cylinder.
            N_radial (int): The number of radial bins.
            N_z (int): The number of z bins.
            z_lim (list): The limits of the z-axis ([z_min, z_max]).
            shift_theta (float, optional): A shift applied to the theta coordinate (default: 0.).
        """
    
        self.radius = r
        self.z_lim = np.array(z_lim)
        
        self.Nr = N_radial
        self.Nz = N_z
        self.dtheta = 2 * np.pi / (2 * np.arange(self.Nr) + 1)
        self.r_edges = np.linspace(0, self.radius, self.Nr + 1)
        self.dr = self.r_edges[1] - self.r_edges[0]
        self.inv_dr = 1.0 / self.dr
        self.z_edges = np.linspace(self.z_lim[0], self.z_lim[1], self.Nz + 1)
        self.dz = self.z_edges[1] - self.z_edges[0]
        self.inv_dz = 1.0 / self.dz
        self.get_area_bins()
        
        if shift_theta is None:
            self._shift_theta = 0
        else:
            self._shift_theta = shift_theta
        
        self.is_inside = np.vectorize(self._is_inside)
        self.data = None
        
        
    def rebin(self, N_radial, N_z):
        """
        Rebins the data into a new binning scheme.

        Args:
            N_radial (int): The new number of radial bins.
            N_z (int): The new number of z bins.
        """        
        self.Nr = N_radial
        self.Nz = N_z
        self.r_edges = np.linspace(0, self.radius, self.Nr + 1)
        if not self.data is None: 
            self.calculate_indices(self.x, self.y, self.z, True)
        
        
    def _is_inside(self, x, y, z):
        """
        Checks whether a position is inside the volume.
        
        Args:
            x (float): The x coordinate of the point.
            y (float): The y coordinate of the point.
            z (float): The z coordinate of the point.
            
        Returns:
            bool: Boolean returning whether the point is inside the volume or not.
        """
        r2 = x**2 + y**2
        return (r2 <= self.radius**2)&(z >= self.z_lim[0])&(z <= self.z_lim[1])
    
    
    def rotate_2D(self, xy, angle):
        """
        Rotate xy points by the indicated angle in the 2D space.
        
        Args:
            xy (array): The N points to be rotated as (2,N)-array
            angle (float): Angle of rotation in radiants
        """
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        return rotation_matrix @ xy
    
    
    def generate_random_points(self, N):
        """
        Generates random points inside the volume.
        
        Args:
            N (int): The number of random points to be generated.
            
        Returns:
            numpy.array: (N, 3)-dimensional array of random positions inside volume.        
        """
        z = np.random.uniform(self.z_lim[0], self.z_lim[1], N)
        r = np.random.uniform(0, self.radius**2, N)**0.5
        theta = np.random.uniform(0, 2*np.pi, N)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y, z])
    
        
    def set_data(self, x, y, z, data):
        """
        Set the data for the binning and calculate indices.

        Args:
            x (array): X-coordinates of the data points.
            y (array): Y-coordinates of the data points.
            z (array): Z-coordinates of the data points.
            data (array): Data values.
            shift_theta (float, optional): Shift angle for the theta dimension.
        """        
        self.x = x
        self.y = y
        self.z = z
        self.data = data
        self.calculate_indices(x, y, z, True)
        
    
    def calculate_indices(self, x, y, z, save_indices=False):
        """
        Calculate the bin indices for the data points.

        Args:
            x (array): X-coordinates of the data points.
            y (array): Y-coordinates of the data points.
            z (array): Z-coordinates of the data points.
            save_indices (bool, optional): Whether to save the indices as attributes. Defaults to False.

        Returns:
            tuple: Indices for radial, theta, and z dimensions.
        """
        theta = np.mod(np.arctan2(y, x), 2*np.pi)
        r = np.sqrt(x**2 + y**2)
        if save_indices:
            self.r = r
            self.theta = theta
        
        indices_r = (r * self.inv_dr).astype(int)
        indices_z = ((z - self.z_lim[0]) * self.inv_dz).astype(int)
               
        np.maximum(indices_r, 0, out=indices_r)
        np.minimum(indices_r, self.Nr - 1, out=indices_r)

        np.maximum(indices_z, 0, out=indices_z)
        np.minimum(indices_z, self.Nz - 1, out=indices_z)

        dtheta = 2 * np.pi / (2 * indices_r + 1)
        index_theta = np.asarray(theta // dtheta, dtype=int)
        np.maximum(index_theta, 0, out=index_theta)
        np.minimum(index_theta, 2 * indices_r, out=index_theta)
        indices_rt = indices_r**2 + index_theta 
        
        if save_indices:
            self.indices_r = indices_r
            self.index_theta = index_theta
            self.indices_z = indices_z
            self._indices_rt = indices_rt
        else:
            return indices_r, index_theta, indices_z
    
    
    def calculate_distribution(self, distribution, scale=1.):
        """
        Calculate the statistical distribution of the data.

        Args:
            distribution (str): Distribution to compute (e.g., 'mean', 'sum').
            scale (float, optional): Scaling factor for the distribution. Defaults to 1.
        """        
        self._stat = distribution
        self.distribution, _, _, _ = scipy.stats.binned_statistic_2d(self._indices_rt, self.indices_z, self.data,
                                                                     bins=(np.arange(-0.5, self.Nr**2 + 0.5), 
                                                                           np.arange(-0.5, self.Nz + 0.5)),
                                                                     statistic=self._stat)
        self.distribution = self.distribution * scale
        
        
    def set_distribution(self, distribution):
        """
        Set the distribution attribute.

        Args:
            distribution (array): Statistical distribution.
        """        
        self.distribution = distribution
        
    
    def plot_distribution(self, z_index, ax=None, normalize_area=False, **kwargs):
        """
        Plot the distribution of the data for a given z-index.

        Args:
            z_index (int): Index for the z dimension.
            ax (matplotlib.axes.Axes, optional): Matplotlib axis object. Defaults to None.
            normalize_area (bool, optional): Whether to normalize by area. Defaults to False.
            **kwargs: Additional keyword arguments for plotting.

        Returns:
            matplotlib.cm.ScalarMappable: Color map for the plot.
            matplotlib.axes.Axes: Matplotlib axis object.
        """        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define color map
        scale = kwargs.get("scale", 1.)
        if normalize_area and not hasattr(self, 'area_bins'):
            self.get_area_bins()
        sts = self.distribution[:,z_index] * scale / self.area_bins if normalize_area else self.distribution[:,z_index] * scale

        vmin, vmax = kwargs.get("vmin", np.nanmin(sts)), kwargs.get("vmax", np.nanmax(sts))
        cmap = kwargs.get("cmap", "viridis")
        log_scale = kwargs.get("log_scale", False)
        norm = plt.Normalize(vmin=vmin, vmax=vmax) if not log_scale else LogNorm(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap)

        # Plot each bin
        for i_rt in range(self.Nr**2):
            # Extract the distribution value for the bin
            i_r, i_theta = self._from_rt_to_r_t(i_rt)
            dtheta = 2 * np.pi / (2 * i_r + 1)
            stat_value = sts[i_rt]

            if np.isnan(stat_value):
                continue

            # Define bin boundaries
            theta_start = i_theta * dtheta + self._shift_theta
            theta_end = (i_theta + 1) * dtheta + self._shift_theta
            r_start = self.r_edges[i_r]
            r_end = self.r_edges[i_r + 1]
            width = r_end - r_start
            
            # Draw the polygon
            patch = patches.Wedge((0., 0.), r_end, np.rad2deg(theta_start), np.rad2deg(theta_end), 
                                  width=width, edgecolor=cmap(norm(stat_value)), facecolor=cmap(norm(stat_value)))
            ax.add_patch(patch)

        ax.set_xlim(-self.radius*1.2, self.radius*1.2)
        ax.set_ylim(-self.radius*1.2, self.radius*1.2)
        
        return plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax
        
    
    def get_area_bins(self):
        """
        Calculate the area of each bin and store them as an attribute.
        """        
        self.area_bins = self.area_circle(self.r_edges[1])
        
        
    def get_center_bins(self):
        """
        Returns the center of the bins.
        """
        centers = np.zeros(self.Nr**2 * self.Nz)
        z_centers = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])
        r_centers = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])
        r_centers[0] = 0.
        r_centers = np.repeat(r_centers, 2 * np.arange(self.Nr) + 1)
        theta_centers = np.hstack([np.arange(2 * n + 1) * 2 * np.pi / (2 * n + 1) + np.pi / (2 * n + 1) + self._shift_theta
                                   for n in np.arange(self.Nr)])
        
        z_centers = np.repeat(z_centers, self.Nr**2)
        r_centers = np.tile(r_centers, self.Nz)
        theta_centers = np.tile(theta_centers, self.Nz)
        
        return np.column_stack([r_centers * np.cos(theta_centers),
                                r_centers * np.sin(theta_centers),
                                z_centers])
                        
        
    def area_circle(self, radius):
        """
        Calculate the area of a cricle of given radius.

        Args:
            radius (float): Radius of the circle.

        Returns:
            float: Area of the circle.
        """
        return np.pi * radius**2
    
    
    def get_value(self, x, y, z):
        """
        Get the value from the distribution array for given coordinates.

        Args:
            x (float or array): X-coordinate(s).
            y (float or array): Y-coordinate(s).
            z (float or array): Z-coordinate(s).

        Returns:
            array: Distribution value(s) for the given coordinates.
        """
        x = x if hasattr(x, '__len__') else np.array([x])
        y = y if hasattr(y, '__len__') else np.array([y])
        z = z if hasattr(z, '__len__') else np.array([z])
        
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        
        i_r, i_t, i_z = self.calculate_indices(x, y, z)
        i_rt = self._from_r_t_to_rt(i_r, i_t)
        
        return self.distribution[i_rt, i_z]
    
    
    def _from_rt_to_r_t(self, index):
        """
        Given the radius-theta index (internal use), it returns the couple (radius, theta) index.
        
        Args:
            index (int): Radius-theta index.
            
        Returns:
            (index_r, index_t) (array): Radius and theta indeces
        """
        index_r = int(np.floor(np.sqrt(index)))
        index_t = int(index - index_r**2)
        return index_r, index_t
    
    
    def _from_r_t_to_rt(self, index_r, index_t):
        """
        Given the couple (radius, theta) index, it returns the radius-theta index (internal use).
        
        Args:
            (index_r, index_t) (array): Radius and theta indeces
            
        Returns:
            index (int): Radius-theta index.
        """
        index = index_r**2 + index_t
        return index

    
    def __call__(self, x, y, z):
        """
        Callable method to get the value from the distribution array for given coordinates.

        Args:
            x (float or array): X-coordinate(s).
            y (float or array): Y-coordinate(s).
            z (float or array): Z-coordinate(s).

        Returns:
            array: Distribution value(s) for the given coordinates.
        """
        return self.get_value(x, y, z)


class CylindricalBinningDistribution(CylindricalBinning):
    """
    A class to perform cylindrical binning and manage distributions for 3D data.

    Attributes:
        r (float): The radius of the cylinder.
        Nr (int): The number of radial bins.
        Nz (int): The number of z bins.
        Nd (int): Number of distribution bins.
        z_lim (list): The limits of the z-axis.
        _shift_theta (float): Internal variable for theta shifting.
        data (numpy.ndarray, optional): The data array (x, y, z, data).
        distribution (numpy.ndarray, optional): The distribution calculated over the bins.
        area_bins (numpy.ndarray, optional): The area of each bin.
    """
    
    def __init__(self, r, N_radial, N_z, N_dist, z_lim, shift_theta=None):
        """
        Initializes the CylindricalBinningDistribution object with given parameters.

        Args:
            h (float): Height of the binning region.
            N_radial (int): Number of radial bins.
            N_z (int): Number of z bins.
            N_dist (int): Number of distribution bins.
            z_lim (array): Limits for the z-dimension.
            shift_theta (float, optional): Shift angle for the theta dimension. Defaults to pi/2 + dtheta/2.
        """      
        self.radius = r
        self.z_lim = np.array(z_lim)
        
        self.Nr = N_radial
        self.Nz = N_z
        self.dtheta = 2 * np.pi / (2 * np.arange(self.Nr) + 1)
        self.r_edges = np.linspace(0, self.radius, self.Nr + 1)
        self.dr = self.r_edges[1] - self.r_edges[0]
        self.inv_dr = 1.0 / self.dr
        self.z_edges = np.linspace(self.z_lim[0], self.z_lim[1], self.Nz + 1)
        self.dz = self.z_edges[1] - self.z_edges[0]
        self.inv_dz = 1.0 / self.dz
        self.get_area_bins()

        if shift_theta is None:
            self._shift_theta = 0.
        else:
            self._shift_theta = shift_theta
       
        self.is_inside = np.vectorize(self._is_inside)
        self.data = None

            
    def set_data(self, stat):
        """
        Set the statistical data for the binning. 
        Not to be confused with CylindricalBinning::set_data.

        Args:
            stat (array): Statistical distribution.
        """        
        self.set_distribution(stat)
                        
    
    def plot_distribution(self, r_index, theta_index, z_index, ax=None, **kwargs):
        """
        Plot the distribution of the data for given indices.

        Args:
            r_index (int): Index for the radial dimension.
            theta_index (int): Index for the theta dimension.
            z_index (int): Index for the z dimension.
            ax (matplotlib.axes.Axes, optional): Matplotlib axis object. Defaults to None.
            **kwargs: Additional keyword arguments for plotting.

        Returns:
            matplotlib.axes.Axes: Matplotlib axis object.
        """        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define color map
        scale = kwargs.get("scale", 1.)
        rt_index = self._from_r_t_to_rt(r_index, theta_index)
        sts = self.distribution[rt_index, z_index].copy() * scale
        
        tmp = np.concatenate([[-np.inf], sts])
        widths = np.diff(tmp[1:])
        widths = np.concatenate([[np.inf], widths, [np.inf]])
        ax.step(tmp, 1./widths, where='post', **kwargs)
        
        return ax
    
    
    def sample(self, x, y, z, size):
        """
        Sample data from the statistical distribution.

        Args:
            x (float or array): X-coordinate(s).
            y (float or array): Y-coordinate(s).
            z (float or array): Z-coordinate(s).
            size (int): Number of samples to generate.

        Returns:
            array: Sampled data.
        """        
        stat = self.get_value(x, y, z)
        indices = np.random.randint(stat.shape[1] - 1, size=(stat.shape[0], size))
        samples = np.random.uniform(stat[np.unravel_index(indices, stat.shape)], 
                                    stat[np.unravel_index(indices+1, stat.shape)])
        return samples
