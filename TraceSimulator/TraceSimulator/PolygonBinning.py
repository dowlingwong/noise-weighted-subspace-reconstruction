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

class PolygonBinning(object):
    """
    A class for binning data in a 3D polygonal coordinate system.

    This class allows you to define a binning scheme based on 
    polygonal regions in a cylindrical coordinate system. It 
    provides functionalities for calculating bin indices for 
    given data points, calculating distributions over the bins, 
    and plotting the distributions.

    Attributes:
        h (float): The height of the apothem.
        Ns (int): The number of sides of the polygon.
        Nr (int): The number of radial bins.
        Nt (int): The number of theta bins.
        Nz (int): The number of z bins.
        z_lim (list): The limits of the z-axis (default: [0, h]).
        _shift_theta (float): Internal variable for theta shifting.
        data (numpy.ndarray, optional): The data array (x, y, z, data).
        distribution (numpy.ndarray): The distribution calculated over the bins.
        area_bins (numpy.ndarray): The area of each bin.
        corners (numpy.ndarray): The corners of each bin.
        centers (numpy.ndarray): The centers of each bin.
        vertices (numpy.ndarray): The (x,y) vertices of the polygon.
    """
    
    def __init__(self, h, N_side, N_radial, N_theta, N_z, z_lim=None, shift_theta=None):
        """
        Initializes a PolygonBinning object.

        Args:
            h (float): The height of the apothem.
            N_side (int): The number of sides of the polygon.
            N_radial (int): The number of radial bins.
            N_theta (int): The number of theta bins.
            N_z (int): The number of z bins.
            z_lim (list, optional): The limits of the z-axis (default: [0, h]).
            shift_theta (float, optional): A shift applied to the theta coordinate (default: None).
        """
    
        self.h = h
        if z_lim is None:
            self.z_lim = np.array([0, self.h])
        else:
            self.z_lim = np.array(z_lim)
        self.Ns = N_side
        
        self.Nr = N_radial
        self.Nt = N_theta
        self.Nz = N_z
        self.dtheta = 2 * np.pi / self.Ns
        self.hrot_edges = np.linspace(0, self.h, self.Nr + 1)
        if shift_theta is None:
            self._shift_theta = 0
        else:
            self._shift_theta = shift_theta
        thetas = np.arange(0, 2 * np.pi, self.dtheta) - self._shift_theta
        r = self.h / np.cos(self.dtheta / 2.)
        self.vertices = np.vstack([r * np.cos(thetas), r * np.sin(thetas)])
        self.is_inside = np.vectorize(self._is_inside)
        self.data = None
        
        
    def rebin(self, N_radial, N_theta, N_z):
        """
        Rebins the data into a new binning scheme.

        Args:
            N_radial (int): The new number of radial bins.
            N_theta (int): The new number of theta bins.
            N_z (int): The new number of z bins.
        """        
        self.Nr = N_radial
        self.Nt = N_theta
        self.Nz = N_z
        self.hrot_edges = np.linspace(0, self.h, self.Nr + 1)
        if not self.data is None: 
            self.calculate_indices(self.x, self.y, self.z, True)
        thetas = np.arange(0, 2 * np.pi, self.dtheta) - self._shift_theta
        r = self.h / np.cos(self.dtheta / 2.)
        self.vertices = np.vstack([r * np.cos(thetas), r * np.sin(thetas)])
        
        
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
        wn = 0
        for i in range(self.Ns - 1):
            x0, y0 = self.vertices.T[i]
            x1, y1 = self.vertices.T[i + 1]
            if y0 <= y:
                if y1 > y and (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0) > 0:
                    wn += 1
            else:
                if y1 <= y and (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0) < 0:
                    wn -= 1
        return (wn != 0) & (z < self.z_lim[1]) & (z > self.z_lim[0])
    
    
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
        Generates random points inside the volume usin triangle point picking algorithm.
        
        Args:
            N (int): The number of random points to be generated.
            
        Returns:
            numpy.array: (N, 3)-dimensional array of random positions inside volume.        
        """
        side_n = np.random.randint(0, self.Ns, N)
        angles = side_n * self.dtheta
        rot_matrices = np.array([[np.cos(angles), np.sin(angles)],
                                 [-np.sin(angles), np.cos(angles)]]).T
        s = np.random.uniform(0, 1, N)
        t = np.random.uniform(0, 1, N)
        z = np.random.uniform(*self.z_lim, N)
        u = self.vertices[:2,0].copy()
        v = self.vertices[:2,1].copy()
        is_in_triangle = s + t <= 1.
        s[~is_in_triangle] = 1. - s[~is_in_triangle]
        t[~is_in_triangle] = 1. - t[~is_in_triangle]
        xy = s[:,None] * u + t[:,None] * v
        xy = np.einsum('ijk,ik->ij', rot_matrices, xy)
        return np.column_stack([xy, z])    
    
        
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
            tuple: Indices for slice, theta, radial, and z dimensions.
        """
        x_rot, y_rot = self.rotate_2D(np.vstack([x, y]), self._shift_theta)
        # calculate the angle removing the angular offset
        theta = np.mod(np.arctan2(y_rot, x_rot), 2*np.pi)
        r = np.sqrt(x**2 + y**2)
        if save_indices:
            self.theta_rot = theta
            self.r = r
        
        index_slice = np.clip(np.asarray(theta // self.dtheta, dtype=int), 0, self.Ns - 1)
        index_theta = np.clip(np.array((theta % self.dtheta) // (self.dtheta / self.Nt), dtype=int), 0, self.Nt - 1)
        rotation_matrix = np.array([[np.cos(self.dtheta * index_slice), -np.sin(self.dtheta * index_slice)],
                                    [np.sin(self.dtheta * index_slice), np.cos(self.dtheta * index_slice)]])
        xp, yp = (rotation_matrix.T @ np.vstack([x_rot, y_rot]).T[:,:, None]).T[0]
        thetap = self.dtheta / 2 - np.arctan2(yp, xp)
        h_rot = r * np.cos(thetap)
        indices_r = np.clip(np.digitize(h_rot, self.hrot_edges) - 1, 0, self.Nr - 1)
        indices_z = np.clip(np.digitize(z, np.linspace(self.z_lim[0], self.z_lim[1], self.Nz + 1)) - 1, 0, self.Nz - 1)
        if save_indices:
            self.index_slice = index_slice
            self.index_theta = index_theta
            self.h_rot = h_rot
            self.indices_r = indices_r
            self.indices_z = indices_z
        else:
            return index_slice, index_theta, indices_r, indices_z
    
    
    def calculate_distribution(self, statistic, scale=1.):
        """
        Calculate the statistical distribution of the data.

        Args:
            statistic (str): Statistic to compute (e.g., 'mean', 'sum').
            scale (float, optional): Scaling factor for the distribution. Defaults to 1.
        """        
        self._stat = statistic
        self.distribution, _, _ = scipy.stats.binned_statistic_dd((self.index_slice, self.index_theta, self.indices_r, self.indices_z), self.data,
                                                               bins=(np.arange(-0.5, self.Ns + 0.5), np.arange(-0.5, self.Nt + 0.5), 
                                                                     np.arange(-0.5, self.Nr + 0.5), np.arange(-0.5, self.Nz + 0.5)),
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
        sts = self.distribution[:,:,:,z_index] * scale / self.area_bins[:,:,:,z_index] if normalize_area else self.distribution[:,:,:,z_index] * scale

        vmin, vmax = kwargs.get("vmin", np.nanmin(sts)), kwargs.get("vmax", np.nanmax(sts))
        cmap = kwargs.get("cmap", "viridis")
        log_scale = kwargs.get("log_scale", False)
        norm = plt.Normalize(vmin=vmin, vmax=vmax) if not log_scale else LogNorm(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap)

        # Plot each bin
        for i_side in range(self.Ns):
            for i_theta in range(self.Nt):
                for i_r in range(self.Nr):
                    # Extract the distribution value for the bin
                    stat_value = sts[i_side, i_theta, i_r].copy()

                    if np.isnan(stat_value):
                        continue

                    # Define bin boundaries
                    theta_start = i_theta * self.dtheta / self.Nt
                    theta_end = (i_theta + 1) * self.dtheta / self.Nt
                    r_start = self.hrot_edges[i_r]
                    r_end = self.hrot_edges[i_r + 1]
                    rotation_matrix = lambda angle: np.array([[np.cos(angle), -np.sin(angle)], 
                                                              [np.sin(angle), np.cos(angle)]])

                    # Calculate polygon vertices
                    corners = [
                        (r_start * np.cos(theta_start) / np.cos(self.dtheta / 2 - theta_start), 
                         r_start * np.sin(theta_start) / np.cos(self.dtheta / 2 - theta_start)),
                        (r_start * np.cos(theta_end) / np.cos(self.dtheta / 2 - theta_end),
                         r_start * np.sin(theta_end) / np.cos(self.dtheta / 2 - theta_end)),
                        (r_end * np.cos(theta_end) / np.cos(self.dtheta / 2 - theta_end),
                         r_end * np.sin(theta_end) / np.cos(self.dtheta / 2 - theta_end)),
                        (r_end * np.cos(theta_start) / np.cos(self.dtheta / 2 - theta_start),
                         r_end * np.sin(theta_start) / np.cos(self.dtheta / 2 - theta_start))
                    ]
                    corners = [rotation_matrix(-self._shift_theta + np.pi) @ v for v in corners]
                    corners = [rotation_matrix(i_side * self.dtheta) @ v for v in corners]
                    
                    # Close the polygon path
                    polygon = mpltPath.Path(corners)

                    # Draw the polygon
                    patch = patches.Polygon(corners, closed=True, edgecolor=cmap(norm(stat_value)), facecolor=cmap(norm(stat_value)))
                    ax.add_patch(patch)

        ax.set_xlim(-self.h*1.2, self.h*1.2)
        ax.set_ylim(-self.h*1.2, self.h*1.2)
        
        return plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax
        
    
    def get_area_bins(self):
        """
        Calculate the area of each bin and store them as an attribute.
        """        
        self.area_bins = np.zeros((self.Ns, self.Nt, self.Nr, self.Nz))
        self.corners = np.zeros((self.Ns, self.Nt, self.Nr, self.Nz, 4, 2))
        self.centers = np.zeros((self.Ns, self.Nt, self.Nr, self.Nz, 3))
        z_centers = np.linspace(self.z_lim[0], self.z_lim[1], self.Nz + 1)
        z_centers = 0.5 * (z_centers[:-1] + z_centers[1:])
        for i_side in range(self.Ns):
            for i_theta in range(self.Nt):
                for i_r in range(self.Nr):

                    # Define bin boundaries
                    theta_start = i_theta * self.dtheta / self.Nt
                    theta_end = (i_theta + 1) * self.dtheta / self.Nt
                    r_start = self.hrot_edges[i_r]
                    r_end = self.hrot_edges[i_r + 1]
                    rotation_matrix = lambda angle: np.array([[np.cos(angle), -np.sin(angle)], 
                                                              [np.sin(angle), np.cos(angle)]])

                    # Calculate polygon vertices
                    corners = [
                        (r_start * np.cos(theta_start) / np.cos(self.dtheta / 2 - theta_start), 
                         r_start * np.sin(theta_start) / np.cos(self.dtheta / 2 - theta_start)),
                        (r_start * np.cos(theta_end) / np.cos(self.dtheta / 2 - theta_end),
                         r_start * np.sin(theta_end) / np.cos(self.dtheta / 2 - theta_end)),
                        (r_end * np.cos(theta_end) / np.cos(self.dtheta / 2 - theta_end),
                         r_end * np.sin(theta_end) / np.cos(self.dtheta / 2 - theta_end)),
                        (r_end * np.cos(theta_start) / np.cos(self.dtheta / 2 - theta_start),
                         r_end * np.sin(theta_start) / np.cos(self.dtheta / 2 - theta_start))
                    ]
                    corners = [rotation_matrix(-self._shift_theta + np.pi) @ v for v in corners]
                    corners = [rotation_matrix(i_side * self.dtheta) @ v for v in corners]
                    corners = np.array(corners)
                    for i_z in range(self.Nz):
                        self.corners[i_side][i_theta][i_r][i_z] = corners.copy()
                        self.area_bins[i_side][i_theta][i_r][i_z] = self.area_polygon(corners[:,0], corners[:,1])
                        self.centers[i_side][i_theta][i_r][i_z] = np.array([np.mean(corners[:,0]), 
                                                                            np.mean(corners[:,1]),
                                                                            z_centers[i_z]])
                        
        
    def area_polygon(self, x, y):
        """
        Calculate the area of a polygon given its vertices.

        Args:
            x (array): X-coordinates of the vertices.
            y (array): Y-coordinates of the vertices.

        Returns:
            float: Area of the polygon.
        """
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    
    
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
        
        i_s, i_t, i_r, i_z = self.calculate_indices(x, y, z)
        
        return self.distribution[i_s, i_t, i_r, i_z]
    
    
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


class PolygonBinningDistribution(PolygonBinning):
    """
    A class to perform polygonal binning and manage distributions for 3D data.

    Attributes:
        h (float): The height of the cylinder.
        Ns (int): The number of sides of the polygon.
        Nr (int): The number of radial bins.
        Nt (int): The number of theta bins.
        Nz (int): The number of z bins.
        Nd (int): Number of distribution bins.
        z_lim (list): The limits of the z-axis (default: [0, h]).
        _shift_theta (float): Internal variable for theta shifting.
        data (numpy.ndarray, optional): The data array (x, y, z, data).
        distribution (numpy.ndarray, optional): The distribution calculated over the bins.
        area_bins (numpy.ndarray, optional): The area of each bin.
        corners (numpy.ndarray, optional): The corners of each bin.
        centers (numpy.ndarray, optional): The centers of each bin.
    """
    
    def __init__(self, h, N_side, N_radial, N_theta, N_z, N_dist, z_lim=None, shift_theta=None):
        """
        Initializes the PolygonBinningDistribution object with given parameters.

        Args:
            h (float): Height of the binning region.
            N_side (int): Number of sides for the polygon.
            N_radial (int): Number of radial bins.
            N_theta (int): Number of theta bins.
            N_z (int): Number of z bins.
            N_dist (int): Number of distribution bins.
            z_lim (array, optional): Limits for the z-dimension. Defaults to [0, h].
            shift_theta (float, optional): Shift angle for the theta dimension. Defaults to pi/2 + dtheta/2.
        """        
        self.h = h
        if z_lim is None:
            self.z_lim = np.array([0, self.h])
        else:
            self.z_lim = np.array(z_lim)
        self.Ns = N_side
        
        self.Nr = N_radial
        self.Nt = N_theta
        self.Nz = N_z
        self.Nd = N_dist
        self.dtheta = 2 * np.pi / self.Ns
        self.hrot_edges = np.linspace(0, self.h, self.Nr + 1)
        if shift_theta is None:
            self._shift_theta = 0.
        else:
            self._shift_theta = shift_theta

            
    def set_data(self, stat):
        """
        Set the statistical data for the binning. 
        Not to be confused with PolygonBinning::set_data.

        Args:
            stat (array): Statistical distirbution.
        """        
        self.set_distribution(stat)
                        
    
    def plot_distribution(self, s_index, theta_index, r_index, z_index, ax=None, **kwargs):
        """
        Plot the distribution of the data for given indices.

        Args:
            s_index (int): Index for the side dimension.
            theta_index (int): Index for the theta dimension.
            r_index (int): Index for the radial dimension.
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
        sts = self.distribution[s_index, theta_index, r_index, z_index].copy() * scale
        
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
