import os
import sys
import tqdm

import numpy as np
import pandas as pd
from scipy.fft import rfftfreq

from .DELightSignalFormation import DELightSignalFormation
from .SimulationMap import *
from .NoiseGenerator import NoiseGenerator

class TraceSimulator(object):
    
    def __init__(self, config):
        """
        Initialize the TraceSimulator object with the provided configuration.

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
        self.disable_trace = self.config.get("disable_trace", False)
        
        self._expected_files = ['LCE', 'PCE', 'TCE']
        if not self.disable_trace:
            self._expected_files += ['optArrivalTime', 'phonArrivalTime', 'trArrivalTime']
        for k in self._expected_files:
            if not k in self.config.keys():
                raise RuntimeError(f"Configuration field {k} not found in configuration dictionary.")
                
        self._set_digitizer_config()
        self._get_template()

        self._load_position_LAMCAL(self.config['positionLAMCAL'])
        self._set_LCE(self.config['LCE'])
        self._set_PCE(self.config['PCE'])
        self._set_TCE(self.config['TCE'])
        if not self.disable_trace:
            self._set_optArrivalTime(self.config['optArrivalTime'])
            self._set_phonArrivalTime(self.config['phonArrivalTime'])
            self._set_trArrivalTime(self.config['trArrivalTime'])
        self._set_signalPartition(self.config.get('signalPartition', None))
        self._set_noiseGenerator()
        
        
    def _load_position_LAMCAL(self, file_path):
        """
        Load indices and positions of the LAMCALs from a given text file.

        Parameters:
        file_path (str): Path to the text file containing the information about LAMCALs.
        """
        _tmp = np.loadtxt(file_path)
        self.LAMCAL_position = _tmp[:,1:]
        self.LAMCAL_indices =np.asarray(_tmp[:,0], dtype=int)
        
        
    def _set_LCE(self, file_path):
        """
        Set the Light Collection Efficiency (LCE) from the provided file path.

        Parameters:
        file_path (str): Path to the file containing LCE data.
        """
        self.lce_per_LAMCAL = SimulationMap(file_path, num_LAMCAL=self.n_LAMCAL, is_cyl=self.is_cyl)
        self.lce = SimulationMap(file_path, per_LAMCAL=False, num_LAMCAL=self.n_LAMCAL, is_cyl=self.is_cyl)
        
        
    def _set_optArrivalTime(self, file_path):
        """
        Set the optical arrival time from the provided file path.

        Parameters:
        file_path (str): Path to the file containing optical arrival time data.
        """
        self.optArrTime = SimulationMap(file_path, num_LAMCAL=self.n_LAMCAL, is_cyl=self.is_cyl)
        
        
    def _set_PCE(self, file_path):
        """
        Set the Phonon Collection Efficiency (PCE) from the provided file path.

        Parameters:
        file_path (str): Path to the file containing PCE data.
        """
        self.pce_per_LAMCAL = SimulationMap(file_path, num_LAMCAL=self.n_vacuum_LAMCAL, is_cyl=self.is_cyl)
        self.pce = SimulationMap(file_path, num_LAMCAL=self.n_vacuum_LAMCAL, per_LAMCAL=False, is_cyl=self.is_cyl)
        
        
    def _set_phonArrivalTime(self, file_path):
        """
        Set the phonon arrival time from the provided file path.

        Parameters:
        file_path (str): Path to the file containing phonon arrival time data.
        """
        self.phonArrTime = SimulationMap(file_path, num_LAMCAL=self.n_vacuum_LAMCAL, is_distribution=True, is_cyl=self.is_cyl)


    def _set_TCE(self, file_path):
        """
        Set the Triplet Collection Efficiency (TCE) from the provided file path.

        Parameters:
        file_path (str): Path to the file containing TCE data.
        """
        self.tce_per_LAMCAL = SimulationMap(file_path, num_LAMCAL=self.n_LAMCAL, is_cyl=self.is_cyl)
        self.tce = SimulationMap(file_path, num_LAMCAL=self.n_LAMCAL, per_LAMCAL=False, is_cyl=self.is_cyl)
        
    
    def _set_trArrivalTime(self, file_path):
        """
        Set the triplet arrival time from the provided file path.

        Parameters:
        file_path (str): Path to the file containing triplet arrival time data.
        """
        self.trArrTime = SimulationMap(file_path, num_LAMCAL=self.n_LAMCAL, is_distribution=True, is_cyl=self.is_cyl)
        
        
    def _set_signalPartition(self, file_path=None):
        """
        Initialize the signal partition using DELightSignalFormation.

        Parameters:
        - file_path: str or None
            Can be a custom path or None. 
            If None or invalid, it will attempt to use default paths.
        """
        _paths = {
            'IAP': '/kalinka/storage/darkmatter/DELight/share/TraceSimulator/signal_partition/',
            'UHD': '/data/silo02/users/ftoschi/share/TraceSimulator/signal_partition/',
            'ETP': '/ceph/bmaier/delight/share/TraceSimulator/signal_partition/'
        }

        if file_path is not None and os.path.exists(file_path):
            _template_path = file_path
            print(f"Using customized partition path: {_template_path}")
        else:
            for key in ['IAP', 'UHD', 'ETP']:
                fallback_path = self.config.get("signalPartitionDirectory", _paths[key])
                if os.path.exists(fallback_path):
                    _template_path = fallback_path
                    #print(f"Using default signal partition path: {_template_path}")
                    break
            else:
                raise FileNotFoundError("No valid signal partition path found in provided or default locations.")

        _load_interaction = self.config.get("load_interaction", "both")
        self.dsf = DELightSignalFormation(template_path=_template_path, load=_load_interaction)
        
        
        
    def _set_noiseGenerator(self):
        """
        Initialize the noise generator.
        """
        self.ng = NoiseGenerator(self.config)
        
        
    def _set_digitizer_config(self):
        """
        Set the digitizer configuration parameters like sampling frequency, time step, trace samples, etc.
        """
        
        self.is_cyl = self.config.get("cylindrical", True)
        self.frequency = self.config.get("sampling_frequency", 2.5e5) # Hz
        self.dt = 1. / self.frequency * 1e9 # ns
        self.trace_samples = self.config.get("trace_samples", 16_384) # num. of samples
        self.trace_duration = self.dt * self.trace_samples # ns
        self.t_edges = np.arange(0., self.trace_duration + self.dt / 2., self.dt)
        
        self.n_LAMCAL = self.config.get("total_LAMCALs", 56)
        self.n_vacuum_LAMCAL = self.config.get("vacuum_LAMCALs", 37)
        
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
        
        
    def set_noise_type(self, noise_type):
        """
        Change noise type and re-initialize the noise generator.
        """
        config_updated = self.config.copy()
        config_updated['noise_type'] = noise_type
        try:
            self.ng.set_noise_type(config_updated['noise_type'])
        except:
            raise
        else:
            self.config = config_updated
                

    def set_noise_power(self, noise_power):
        """
        Change noise power and re-initialize the noise generator.
        """
        config_updated = self.config.copy()
        config_updated['noise_power'] = noise_power
        try:
            self._set_noiseGenerator(config_updated)
        except:
            raise RuntimeError("Failed to update the noise generator with the new noise power.")
        else:
            self.config = config_updated
    
    
    def get_noise(self):
        """
        Generate noise based on configuration.

        Returns:
        numpy.ndarray: Generated noise array.
        """        
            
        return self.ng.generate_noise(self.trace_samples)
            
            
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
        
        
    def get_LAMCAL_distribution(self, E, x, y, z, type_recoil, to_energy=False, progress_bar=False):
        """
        Compute LAMCAL-distributed quanta for given energies and positions.

        Parameters:
        E (float or list of floats): Recoil energy or list of energies.
        x (float or list of floats): x-coordinate(s).
        y (float or list of floats): y-coordinate(s).
        z (float or list of floats): z-coordinate(s).
        type_recoil (str): Type of recoil ('ER' or others).
        to_energy (bool, optional): If True, convert output counts to energies.
        progress_bar (bool, optional): If True, show a progress bar.

        Returns:
        tuple:
            If to_energy=False:
                numpy.ndarray: Observed phonon counts.
                numpy.ndarray: Observed UV photon counts.
                numpy.ndarray: Observed IR photon counts.
                numpy.ndarray: Observed triplet counts.
            If to_energy=True:
                numpy.ndarray: Phonon energies.
                numpy.ndarray: UV photon energies.
                numpy.ndarray: IR photon energies.
                numpy.ndarray: Triplet energies.
        """        
        E = self._make_array(E)
        E_ph, E_tr, E_uv, E_ir = self.dsf.get_partition(E, int_type=type_recoil)
        
        N_uv = np.asarray(E_uv / self.dsf.E_UV, dtype=int)
        N_tr = np.asarray(E_tr / self.dsf.E_triplet, dtype=int)
        N_ir = np.asarray(np.ceil(E_ir / self.dsf.E_IR_avg), dtype=int)
        N_ph = np.asarray(E_ph * 1e3, dtype=int) # average energy phonon: 1.0 meV
        
        with np.errstate(divide='ignore', invalid='ignore'):#safe division, and avoid warning
            self._E_ir_avg = np.divide(E_ir, N_ir, out=np.zeros_like(E_ir), where=N_ir > 0)

        N_uv_obs = np.zeros((len(E), self.n_LAMCAL))
        N_ir_obs = np.zeros((len(E), self.n_LAMCAL))
        N_tr_obs = np.zeros((len(E), self.n_LAMCAL))
        N_ph_obs = np.zeros((len(E), self.n_vacuum_LAMCAL))
        iteration = tqdm.tqdm(range(len(E))) if progress_bar else range(len(E))
        for iE in iteration:
            lceff = np.zeros(self.n_LAMCAL+1)
            tceff = np.zeros(self.n_LAMCAL+1)
            pceff = np.zeros(self.n_vacuum_LAMCAL+1)
            for i in range(self.n_LAMCAL):
                lceff[i] = self.lce_per_LAMCAL(x[iE],y[iE],z[iE],i)
                tceff[i] = self.tce_per_LAMCAL(x[iE],y[iE],z[iE],i)
                if i >= self.n_LAMCAL - self.n_vacuum_LAMCAL:
                    i_sub = i - (self.n_LAMCAL - self.n_vacuum_LAMCAL)
                    pceff[i_sub] = self.pce_per_LAMCAL(x[iE],y[iE],z[iE],i_sub)
            lceff[-1] = 1 - np.sum(lceff[:-1])
            tceff[-1] = 1 - np.sum(tceff[:-1])
            pceff[-1] = 1 - np.sum(pceff[:-1])
            N_uv_obs[iE] = np.random.multinomial(N_uv[iE],lceff)[:-1]
            N_ir_obs[iE] = np.random.multinomial(N_ir[iE],lceff)[:-1]
            N_tr_obs[iE] = np.random.multinomial(N_tr[iE],tceff)[:-1]
            N_ph_obs[iE] = np.random.multinomial(N_ph[iE],pceff)[:-1]
            
        if not to_energy:
            return N_ph_obs.astype(int), N_uv_obs.astype(int), N_ir_obs.astype(int), N_tr_obs.astype(int)
        else:
            return N_ph_obs*1e-3*self.gain_QE, N_uv_obs*self.dsf.E_UV, N_ir_obs*self._E_ir_avg[:,None], N_tr_obs*self.dsf.E_UV

        
    def generate(self, E, x=None, y=None, z=None, type_recoil='ER', phonon_only=False, no_noise=False, quantize=True):
        """
        Generate traces for given energies and optionally positions.

        Parameters:
        E (float or list of floats): Energy or list of energies.
        x (float or list of floats, optional): x-coordinate(s).
        y (float or list of floats, optional): y-coordinate(s).
        z (float or list of floats, optional): z-coordinate(s).
        type_recoil (str, optional): Type of recoil ('ER' or other types).

        Returns:
        numpy.ndarray: Generated traces in ADC counts.
        tuple: Generated positions if x, y, or z are not provided.
        """    
        
        if self.disable_trace:
            raise ValueError("Trace simulation is disabled")
            
        return_position = False
        E = self._make_array(E)
        if x is None or y is None or z is None:
            return_position = True
            x, y, z = np.squeeze(self.generate_random_points(len(E))).T
        else:
            if not self.is_inside(x, y, z):
                raise ValueError("Chosen point is outside of target volume")
            
        x, y, z = self._make_array(x), self._make_array(y), self._make_array(z)
        ts = np.zeros((len(E), self.n_LAMCAL, self.trace_samples))
        N_ph_obs, N_uv_obs, N_ir_obs, N_tr_obs = self.get_LAMCAL_distribution(E, x, y, z, type_recoil)
        for i in range(self.n_LAMCAL):
            if not phonon_only:
                t_uv_obs = self.optArrTime(x, y, z, i) # ns
                ts[:, i] += self.energy_to_ADC * N_uv_obs[:,i,None] * self.dsf.E_UV * \
                        np.array([np.histogram(x, bins=self.t_edges)[0] for x in t_uv_obs])
                ts[:, i] += self.energy_to_ADC * N_ir_obs[:,i,None] * self._E_ir_avg[:,None] * \
                        np.array([np.histogram(x, bins=self.t_edges)[0] for x in t_uv_obs])
                for j, n in enumerate(N_tr_obs):
                    t_tr_obs = self.trArrTime.map[i].sample(x[j], y[j], z[j], int(n[i]))[0] # ns
                    # triplets decay into UV, in this case we are neglecting the remaining 17.82 eV - 15.396 eV = 2.424 eV
                    # as it is not clear the nature of this energy deposition (quasiparticles, IR?)
                    ts[j, i] += self.dsf.E_UV * self.energy_to_ADC * np.histogram(t_tr_obs, bins=self.t_edges)[0]            

            if i >= self.n_LAMCAL - self.n_vacuum_LAMCAL:
                i_sub = i - (self.n_LAMCAL - self.n_vacuum_LAMCAL)
                for j, n in enumerate(N_ph_obs[:,i_sub]):
                    t_ph_obs = self.phonArrTime.map[i_sub].sample(x[j], y[j], z[j], n)[0] # ns
                    ts[j, i] += self.gain_QE * self.energy_to_ADC * 1e-3 * np.histogram(t_ph_obs, bins=self.t_edges)[0]            
        for i in range(len(E)):
            for j in range(self.n_LAMCAL):
                if no_noise:
                    ts[i, j] = self._sum_template(ts[i, j])
                else:
                    ts[i, j] = self._sum_template(ts[i, j]) + self.get_noise()
        if quantize:
            ts = self.quantize_trace(ts)

        if return_position:
            return ts, (x, y, z)
        else:
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
    
    
    def generate_random_points(self, N):
        """
        Generate N random points inside the simulation volume.

        Parameters:
        N (int): Number of points to generate.

        Returns:
        numpy.ndarray: Generated random points.
        """        
        return self.lce.map.generate_random_points(N)


    def is_inside(self, x, y, z):
        """
        Check if the given positions are inside the simulation volume.

        Parameters:
        x (float or numpy.ndarray): x-coordinate(s).
        y (float or numpy.ndarray): y-coordinate(s).
        z (float or numpy.ndarray): z-coordinate(s).

        Returns:
        bool or numpy.ndarray: True if the positions are inside, False otherwise.
        """        
        return self.lce.map.is_inside(x, y, z)
    
    
    def quantize_trace(self, x):
        """
        Quantize trace to ADC counts.
        
        Parameters:
        x (numpy.array): Trace in float ADC counts.
        
        Returns:
        numpy.array: Array of integer ADC counts.
        """
        x_int = np.array(x, dtype=int)
        return x_int

