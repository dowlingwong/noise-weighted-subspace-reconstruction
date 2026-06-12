import os
import numpy as np
import h5py
import tqdm

class DELightSignalFormation:
    
    def __init__(self, template_path=None, load='both'):
        if template_path is None:
            self._template_path = os.path.dirname(os.path.abspath(__file__)) + '/templates'
        else:
            self._template_path = template_path

        # Energy constants
        self.E_UV = 15.396
        self.E_triplet = 17.82
        self.E_excitation = 19.82
        self.E_IR_avg = 2
        
        # Open HDF5 files for ER and/or NR data
        load = load.lower()
        self.load_type = load if load in ['both', 'nr', 'er'] else 'both'
        if self.load_type in ['both', 'er']:
            self._load_ER()
        if self.load_type in ['both', 'nr']:
            self._load_NR()

    def _load_ER(self):
        """
        Open the HDF5 files for ER data and keep file handles open for efficient access.
        """
        self.ER_pcumsum = h5py.File(os.path.join(self._template_path, 'ER_signal_partition_p_cumsum.h5'), 'r')['arr_0']
        self.ER_split = h5py.File(os.path.join(self._template_path, 'ER_signal_partition_p_split.h5'), 'r')['arr_0']
        self.ER_edges = h5py.File(os.path.join(self._template_path, 'ER_signal_partition_edges.h5'), 'r')['arr_0']
        self.ER_energies = np.load(os.path.join(self._template_path, 'ER_signal_partition_energies.npy'))
        
    def _load_NR(self):
        """
        Open the HDF5 files for NR data and keep file handles open for efficient access.
        """
        self.NR_pcumsum = h5py.File(os.path.join(self._template_path, 'NR_signal_partition_p_cumsum.h5'), 'r')['arr_0']
        self.NR_split = h5py.File(os.path.join(self._template_path, 'NR_signal_partition_p_split.h5'), 'r')['arr_0']
        self.NR_edges = h5py.File(os.path.join(self._template_path, 'NR_signal_partition_edges.h5'), 'r')['arr_0']
        self.NR_energies = np.load(os.path.join(self._template_path, 'NR_signal_partition_energies.npy'))
        
    def get_partition(self, E0, int_type='ER', progress_bar=False):
        """
        Returns a random distribution of energy partition for superfluid He-4,
        based on the interaction energy and interaction type.

        Args:
            E0 (float, array): Deposited energy in eV.
            int_type (str): Interaction type ('ER' or 'NR'). Default set to 'ER'.
            
        Returns:
            list of arrays: Energy going into each channel in the form (E_QP, E_tr, E_UV, E_IR).
        """
        E0 = np.atleast_1d(E0)
        tot = len(E0)
        E_QP, E_tr, E_UV, E_IR = np.zeros((4, tot))
        if progress_bar:
            for i in tqdm.tqdm(range(tot), total=tot, desc="Partitioning"):
                E_QP[i], E_tr[i], E_UV[i], E_IR[i] = self._get_partition(E0[i], int_type)
        else:
            for i in range(tot):
                E_QP[i], E_tr[i], E_UV[i], E_IR[i] = self._get_partition(E0[i], int_type)
        return E_QP, E_tr, E_UV, E_IR
        
    def _get_partition(self, E0, int_type='ER'):
        int_type = int_type.lower()
        if self.load_type != 'both':
            if int_type != self.load_type:
                error_msg = 'Requested {0} event, but only {1} is possible.'
                raise ValueError(error_msg.format(int_type.upper(), self.load_type.upper()))
                int_type = self.load_type
        
        if int_type == 'er':
            en = self.ER_energies
            pcumsum = self.ER_pcumsum
            split = self.ER_split
            edges = self.ER_edges
        else:
            en = self.NR_energies
            pcumsum = self.NR_pcumsum
            split = self.NR_split
            edges = self.NR_edges

        iE = np.digitize(E0, en[:])
        if E0 < self.E_excitation:
            return E0, 0.0, 0.0, 0.0

        if iE == 0:
            pcumsum_slice = pcumsum[:split[0]]
            edge = edges[0]
        elif iE == len(en):
            pcumsum_slice = pcumsum[np.cumsum(split[:-1])[-1]:]
            iE -= 1
            edge = edges[-1]
        else:
            k_log = (np.log(E0) - np.log(en[iE - 1])) / (np.log(en[iE]) - np.log(en[iE - 1]))
            iE = iE - 1 if k_log < 0.5 else iE
            start = np.sum(split[:iE])
            stop = start + split[iE]
            pcumsum_slice = pcumsum[start:stop]
            edge = edges[iE]

        dim_1 = int(edge[1][1] - edge[1][0])
        dim_2 = int(edge[2][1] - edge[2][0])
        dim_0 = int(len(pcumsum_slice) / dim_1 / dim_2)

        if dim_0 == dim_1 == dim_2 == 1:
            return E0, 0.0, 0.0, 0.0

        bins = (
            np.linspace(edge[0][0], edge[0][-1], dim_0 + 1),
            np.arange(edge[1][0], edge[1][-1] + 0.5),
            np.arange(edge[2][0], edge[2][-1] + 0.5),
        )

        iS = np.random.uniform(0, 1)
        iS = np.digitize(iS, pcumsum_slice)
        res_indices = np.array(np.unravel_index(iS, (dim_0, dim_1, dim_2))).T
        res = np.zeros(3, dtype=float)
        res[0] = bins[0][res_indices[0]] * E0
        res[1] = bins[1][res_indices[1]]
        res[2] = bins[2][res_indices[2]]

        res[1] = res[1] / en[iE] * E0
        dE1 = res[1] - round(res[1])
        res[1] = round(res[1])
        res[2] = res[2] / en[iE] * E0
        dE2 = res[2] - round(res[2])
        res[2] = round(res[2])
        
        res[1] = res[1] * self.E_triplet
        res[2] = res[2] * self.E_UV
        res[0] += dE1 + dE2

        if res[0] + res[1] + res[2] > E0:
            res[0] = E0 - res[1] - res[2]
        if res[1] == res[2] == 0:
            res[0] = E0

        return res[0], res[1], res[2], E0 - res[0] - res[1] - res[2]

    def close_files(self):
        """
        Close the HDF5 files to release resources.
        """
        self._ER_file.close()
        self._NR_file.close()
