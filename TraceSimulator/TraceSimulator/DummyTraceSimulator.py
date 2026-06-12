import os
import sys
import tqdm

import numpy as np
import pandas as pd
from scipy.fft import rfftfreq

class DummyTraceSimulator(object):
    """
    A dummy trace simulator for generating synthetic traces based on a given configuration.
    This class is intended for testing and development purposes and simulates the trace generation process
    by creating noise and template-based signals.
    """
    
    def __init__(self, config):
        """
        Initialize the DummyTraceSimulator object with the provided configuration.

        Parameters:
        config (dict): Configuration dictionary containing necessary parameters and file paths.
        """
        self.config = config
        self.set_config()
        
        
    def set_config(self, config=None):
        """
        Set or update the configuration for the simulation. Also initializes various simulation parameters.

        Parameters:
        config (dict, optional): Configuration dictionary to update the existing configuration.
        
        Raises:
        RuntimeError: If required configuration fields are not found in the configuration dictionary.
        """        
        if not config is None:
            self.config = config
        
        self._expected_files = ['LCE', 'optArrivalTime', 'PCE', 'phonArrivalTime']
        for k in self._expected_files:
            if not k in self.config.keys():
                raise RuntimeError(f"Configuration field {k} not found in configuration dictionary.")
                
        self._set_digitizer_config()
        self._get_template()
        
        
    def _set_digitizer_config(self):
        """
        Set the digitizer configuration parameters like sampling frequency, time step, trace samples, etc.
        """        
        self.frequency = self.config.get("sampling_frequency", 2.5e5) # Hz
        self.dt = 1. / self.frequency * 1e9 # ns
        self.trace_samples = self.config.get("trace_samples", 16_384) # num. of samples
        self.trace_duration = self.dt * self.trace_samples # ns
        self.t_edges = np.arange(0., self.trace_duration + self.dt / 2., self.dt)
        
        self.n_MMC = 54
        self.n_submerged_MMC = 45
        #self.n_vacuum_MMC = 9
        
        self.noise_type = self.config.get("noise_type", "pink")
        self.noise_std = self.config.get("noise_std", 5.)
        self.energy_to_ADC = self.config.get("E_to_ADC", 2.)
        self.gain_QE = self.config.get("gain_QE", 15.)
        
        
    def _get_template(self):
        """
        Generate the template trace based on the configured rise and decay times.
        """        
        self.tau_rise = self.config.get("tau_rise", 50e3)
        self.tau_decay = self.config.get("tau_decay", 3e6)
        self.trigger_time = self.config.get("trigger_time",
                                            0.1 * self.trace_samples * self.dt)
        xs = np.arange(0, self.trace_duration * 2.5, self.dt)
        self.template = np.concatenate([(np.exp((xs - self.trigger_time) / self.tau_rise))[xs <= self.trigger_time], (np.exp(-(xs - self.trigger_time) / self.tau_decay))[xs > self.trigger_time]])
                
        
    def get_noise(self, noise_std=None, noise_type=None):
        """
        Generate noise based on the specified noise type and standard deviation.

        Parameters:
        noise_std (float, optional): Standard deviation of the noise.
        noise_type (str, optional): Type of noise to generate ('white', 'pink', 'blue', 'violet', 'brown').

        Returns:
        numpy.ndarray: Generated noise array.
        """        
        if not noise_type is None:
            self.config["noise_type"] = noise_type
            self.noise_type = noise_type
        if not noise_std is None:
            self.config["noise_std"] = noise_std
            self.noise_std = noise_std
            
        _noises = {'white': white_noise,
                   'pink': pink_noise,
                   'blue': blue_noise,
                   'violet': violet_noise,
                   'brown': brownian_noise}
        if self.noise_type in _noises.keys():
            return self.noise_std * _noises[self.noise_type](self.trace_samples)
        else:
            return self.noise_std * _noises["pink"](self.trace_samples)
            
            
    def _make_array(self, x):
        """
        Convert the input to a numpy array if it is not already an array.

        Parameters:
        x (list or any): Input to be converted to a numpy array.

        Returns:
        numpy.ndarray: Converted numpy array.
        """
        if isinstance(x, list):
            return np.array(x)
        elif not hasattr(x, '__len__'):
            return np.array([x])
        else:
            return x
        
    def generate(self, E, x, y, z, type_recoil='ER'):
        """
        Generate traces for given energies and optionally positions.

        Parameters:
        E (float or list of floats): Energy or list of energies.
        x (float or list of floats, optional): x-coordinate(s).
        y (float or list of floats, optional): y-coordinate(s).
        z (float or list of floats, optional): z-coordinate(s).
        type_recoil (str, optional): Type of recoil ('ER' or other types).

        Returns:
        numpy.ndarray: Generated traces.
        tuple: Generated positions if x, y, or z are not provided.
        """        
        x, y, z = self._make_array(x), self._make_array(y), self._make_array(z)
        E = self._make_array(E)
        mus = {'ER': np.array([0.3, 0.25, 0.3]), 'NR': np.array([0.7, 0.15, 0.1])}
        E_ph, E_tr, E_uv = [np.abs(np.random.normal(mus[type_recoil][i], 0.05, size=len(E))) for i in range(3)]
        E_ir = 1. - E_ph - E_tr - E_uv
        norm = np.abs(E_ph) + np.abs(E_tr) + np.abs(E_uv) + np.abs(E_ir)
        E_ph, E_tr, E_uv, E_ir = E_ph/norm, E_tr/norm, E_uv/norm, E_ir/norm
        E_ph, E_tr, E_uv, E_ir = E*E_ph, E*E_tr, E*E_uv, E*E_ir
        N_uv = np.asarray(E_uv / 15., dtype=int)
        N_ph = np.asarray(E_ph * 1e3, dtype=int) # average energy phonon: 1.0 meV

        ts = np.zeros((len(E), self.n_MMC, self.trace_samples))
        for i in range(self.n_MMC):
            N_uv_obs = np.random.poisson(np.random.normal(0.05, 0.01, len(E)) * N_uv)
            t_uv_obs = np.random.exponential(1, len(E)) # ns
            ts[:, i] += self.energy_to_ADC * N_uv_obs[:,None] * 15. * \
                       np.apply_along_axis(lambda x: np.histogram(x, bins=self.t_edges)[0], 0, t_uv_obs)

            if i > self.n_MMC - self.n_submerged_MMC:
                i_sub = i - (self.n_MMC - self.n_submerged_MMC)
                N_ph_obs = np.random.poisson(np.random.normal(0.03, 0.007, len(E)) * N_ph)
                for j, n in enumerate(N_ph_obs):
                    t_ph_obs = np.random.noncentral_chisquare(np.random.poisson(4) + 1, np.random.poisson(1) + 1, n)*5e4 + 1e5 # ns
                    ts[j, i] += self.gain_QE * self.energy_to_ADC * 1e-3 * np.histogram(t_ph_obs, bins=self.t_edges)[0]            
        for i in range(len(E)):
            for j in range(self.n_MMC):
                ts[i, j] = self._sum_template(ts[i, j]) + self.get_noise()
        return ts
        
            
    def _sum_template(self, counts):
        """
        Sum the template trace shifted by the counts.

        Parameters:
        counts (numpy.ndarray): Counts to be added to the template.

        Returns:
        numpy.ndarray: Summed trace.
        """
        # need to add shift within time bin
        t_tmp = np.zeros_like(counts)
        for i in np.where(counts > 0.)[0]:
            t_tmp[i:] += counts[i] * self.template[:len(counts) - i]
        return t_tmp
    
    
# from https://stackoverflow.com/questions/67085963/generate-colors-of-noise-in-python 
def noise_psd(N, psd = lambda f: 1):
    """
    Generate noise with a given power spectral density (PSD).

    Parameters:
    N (int): Number of samples.
    psd (function): Function defining the PSD.

    Returns:
    numpy.ndarray: Generated noise.
    """
    X_white = np.fft.rfft(np.random.randn(N));
    S = psd(np.fft.rfftfreq(N))
    # Normalize S
    S = S / np.sqrt(np.mean(S**2))
    X_shaped = X_white * S;
    return np.fft.irfft(X_shaped);

def PSDGenerator(f):
    """
    Decorator to create a PSD generator function.

    Parameters:
    f (function): Function defining the PSD.

    Returns:
    function: PSD generator function.
    """
    return lambda N: noise_psd(N, f)

@PSDGenerator
def white_noise(f):
    return 1;

@PSDGenerator
def blue_noise(f):
    return np.sqrt(f);

@PSDGenerator
def violet_noise(f):
    return f;

@PSDGenerator
def brownian_noise(f):
    return 1/np.where(f == 0, float('inf'), f)

@PSDGenerator
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))
        
