import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset

def normalize_distribution(x, alpha=1.0, beta=7.0):
    """
    Applies logarithmic normalization with sign preservation for distribution-based data.
    Used for datasets like PLIF2D where data follows a distribution rather than bounded range.

    Formula: sign(x) * log(1 + |x/alpha|) / beta

    Parameters:
    x (np.ndarray or torch.Tensor): Input array or tensor with shape (C, H, W) or (num, C, H, W).
    alpha (float): Scale parameter for log1p normalization (controls compression of large values).
    beta (float): Division factor to bring normalized values to [-1, 1] range.

    Returns:
    np.ndarray or torch.Tensor: Normalized array or tensor in range approximately [-1, 1].
    """
    is_torch = isinstance(x, torch.Tensor)

    if is_torch:
        sign_x = torch.sign(x)
        abs_x = torch.abs(x)
        normalized = sign_x * torch.log1p(abs_x / alpha) / beta
    else:
        sign_x = np.sign(x)
        abs_x = np.abs(x)
        normalized = sign_x * np.log1p(abs_x / alpha) / beta

    return normalized


def denormalize_distribution(x_norm, alpha=1.0, beta=7.0):
    """
    Inverts the logarithmic normalization to recover original distribution-based data.

    Inverse formula: x = alpha * sign(x_norm) * (exp(|x_norm * beta|) - 1)

    Parameters:
    x_norm (np.ndarray or torch.Tensor): Normalized array or tensor with shape (C, H, W) or (num, C, H, W).
    alpha (float): Scale parameter used in normalization.
    beta (float): Division factor used in normalization.

    Returns:
    np.ndarray or torch.Tensor: Denormalized array or tensor in original range.
    """
    is_torch = isinstance(x_norm, torch.Tensor)

    if is_torch:
        sign_x = torch.sign(x_norm)
        abs_x = torch.abs(x_norm)
        denormalized = alpha * sign_x * (torch.exp(abs_x * beta) - 1)
    else:
        sign_x = np.sign(x_norm)
        abs_x = np.abs(x_norm)
        denormalized = alpha * sign_x * (np.exp(abs_x * beta) - 1)

    return denormalized


def read_hdf5(data_file):
    import h5py
    data_arr = h5py.File(data_file, 'r', locking=False)
    return data_arr

def get_2Dtest_data(config=None):
    import h5py
    
    test_data_path = config.dataset.test_data_file
    test_data = h5py.File(test_data_path)
    u = test_data['u_fluc'][:]
    v = test_data['v_fluc'][:]

    x, y, t  = test_data['x'][:], test_data['y'][:], test_data['t'][:]

    if config.dataset.ds_ratio == 5 :
        u = u[:, :-1, :-1]
        v = v[:, :-1, :-1]
        
        x = x[:-1]
        y = y[:-1]
        nx, ny = u.shape[1:]
        assert (nx, ny) == (300, 100)

    elif config.dataset.ds_ratio == 2 or config.dataset.ds_ratio == 1:
        u = u[:, :-13, :-5]
        v = v[:, :-13, :-5] 
                      
        x = x[:-13]
        y = y[:-5]
        nx, ny = u.shape[1:]
        assert (nx, ny) == (288, 96)
        # shape = (288,96) -> (144,48) # only use upto 4 layers

    u = u[:, ::config.dataset.ds_ratio, ::config.dataset.ds_ratio]
    v = v[:, ::config.dataset.ds_ratio, ::config.dataset.ds_ratio]
    x = x[::config.dataset.ds_ratio]
    y = y[::config.dataset.ds_ratio]

    assert u.shape[1:] == v.shape[1:] == config.model.image_size
    
    u = np.expand_dims(u, axis=1)  
    v = np.expand_dims(v, axis=1)  
    
    # Concatenate along the new axis
    test_data = np.concatenate((u, v), axis=1)
    assert test_data.shape[2:] == u.shape[2:] ==config.model.image_size,  f"Shape mismatch: test_data.shape[2:] = {test_data.shape[2:]}, u.shape[2:] = {u.shape[2:]}"
    print(f"Shape of test data is {test_data.shape}")
    
    return test_data, x, y, t

def remove_indices_from_array(array=None, indices=None, axis=0):
    return np.delete(array, indices, axis=axis)

class OneObs2D(Dataset):
    
    def __init__(self, data_file=None, filetype="hdf5", transform=None, ds_ratio=1, normalize=True, image_size=None):
        
        assert data_file is not None
        self.normalize = normalize
        self.transform = transform

        if filetype == "hdf5":
            
            #------------------------------------------------------------------
            data_arr = read_hdf5(data_file)
            u = np.asarray(data_arr["u_fluc"][:], dtype = np.float32)   # time, nx, ny
            v = np.asarray(data_arr["v_fluc"][:], dtype = np.float32)   # time, nx, ny
            x = np.asarray(data_arr["x"][:])
            y = np.asarray(data_arr["y"][:])
            t = np.asarray(data_arr["t"][:])
            means = np.asarray(data_arr["means"][:], dtype = np.float32)
            
            nx, ny = 301, 101
            assert u.shape[1:] == (nx, ny)  and  v.shape[1:] == (nx, ny)
            assert x.shape == (nx,) and y.shape == (ny,) and t.shape == (u.shape[0], 1 ) and t.shape== (v.shape[0], 1) 
            assert means.shape[1:] == (nx, ny)
            #------------------------------------------------------------------

            if ds_ratio == 5 :
                u = u[:, :-1, :-1]
                v = v[:, :-1, :-1]
                means = means[:, :-1, :-1]
                x = x[:-1]
                y = y[:-1]
                nx, ny = u.shape[1:]
                assert (nx, ny) == (300, 100)
                # shape = (300,100) -> (60,20) # only use 2 layers
                
            elif ds_ratio == 2 or ds_ratio == 1:
                u = u[:, :-13, :-5]
                v = v[:, :-13, :-5] 
                means = means[:, :-13, :-5]              
                x = x[:-13]
                y = y[:-5]
                nx, ny = u.shape[1:]
                assert (nx, ny) == (288, 96)
                # shape = (288,96) -> (144,48) # only use upto 4 layers

            else:
                print(f"ds_ratio {ds_ratio} not supported")
                raise NotImplementedError
            
            #------------------------------------------------------------------
            
            #downsampling by ds_ratio
            u = u[:, ::ds_ratio, ::ds_ratio]
            v = v[:, ::ds_ratio, ::ds_ratio]

            assert u.shape[1:] == v.shape[1:] == image_size

            self.x = x[::ds_ratio]
            self.y = y[::ds_ratio]
            self.t = t
            self.means = means[:, ::ds_ratio, ::ds_ratio]
            self.data = np.stack((u,v), axis = 1 )   # time, nc, nx, ny

            self.u_min, self.u_max = np.min(u), np.max(u)
            self.v_min, self.v_max = np.min(v), np.max(v)

            #------------------------------------------------------------------------------
            assert self.data.shape[1:] == (2, nx//ds_ratio, ny//ds_ratio) and self.data.dtype == np.float32
            
        else:

            raise NotImplementedError
        
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.normalize:
            image = self.__normalize(image)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __normalize(self, x):
        
        # x shape = (2, h, w)
        eps = 1e-9
        center = np.array([self.u_min, self.v_min]).reshape((2,1,1))
        scale = np.array([self.u_max - self.u_min, self.v_max - self.v_min]).reshape((2,1,1))
        x_scaled =  (x - center) / (scale + eps)
        return ( 2 * x_scaled ) - 1
            

        

class PLIF2D(Dataset):
    """
    Dataset class for PLIF2D fluctuation data with logarithmic normalization.

    Data structure:
        - data: (N, 1, 128, 32) - pre-combined single-channel fluctuation field
        - mask: (128, 32) - boolean mask for valid regions
        - x_coords: (128,) - spatial x coordinates
        - y_coords: (32,) - spatial y coordinates
        - offset_patterns: (N, 2) - offset information per sample
        - original_file_indices: (N,) - source file tracking
    """

    def __init__(self, data_file=None, filetype="hdf5", transform=None,
                 ds_ratio=1, normalize=True, image_size=None,
                 alpha=1.0, beta=7.0):
        """
        Args:
            data_file: Path to HDF5 file
            filetype: File format (only 'hdf5' supported)
            transform: Optional transform to apply
            ds_ratio: Downsampling ratio (should be 1, data already at target resolution)
            normalize: Whether to apply log1p normalization
            image_size: Expected image size tuple (H, W)
            alpha: Scale parameter for log1p normalization (controls compression)
            beta: Division factor to bring normalized values to [-1, 1] range
        """
        assert data_file is not None
        assert filetype == "hdf5", "Only HDF5 format supported"

        self.normalize = normalize
        self.transform = transform
        self.alpha = alpha
        self.beta = beta

        if filetype == "hdf5":
            # ============================================================
            # 1. Load data from HDF5
            # ============================================================
            data_arr = read_hdf5(data_file)

            # Load main data field (already in N, C, H, W format)
            data = np.asarray(data_arr["data"][:], dtype=np.float32)  # (N, 1, 128, 32)

            # Load coordinates
            x_coords = np.asarray(data_arr["x_coords"][:])  # (128,)
            y_coords = np.asarray(data_arr["y_coords"][:])  # (32,)

            # Load mask (optional usage)
            mask = np.asarray(data_arr["mask"][:]).astype(bool)  # (128, 32)

            # Load metadata (for tracking)
            offset_patterns = np.asarray(data_arr["offset_patterns"][:])  # (N, 2)
            original_indices = np.asarray(data_arr["original_file_indices"][:])  # (N,)

            # ============================================================
            # 2. Validate shapes
            # ============================================================
            N, C, H, W = data.shape
            assert (H, W) == image_size, f"Data shape {(H, W)} != expected {image_size}"
            assert C == 1, f"Expected single channel, got {C}"
            assert x_coords.shape == (H,), f"x_coords shape mismatch"
            assert y_coords.shape == (W,), f"y_coords shape mismatch"
            assert mask.shape == (H, W), f"mask shape mismatch"

            # ============================================================
            # 3. Handle downsampling (if needed, though ds_ratio should be 1)
            # ============================================================
            if ds_ratio != 1:
                # Apply downsampling if requested
                data = data[:, :, ::ds_ratio, ::ds_ratio]
                x_coords = x_coords[::ds_ratio]
                y_coords = y_coords[::ds_ratio]
                mask = mask[::ds_ratio, ::ds_ratio]

                # Update dimensions
                H, W = H // ds_ratio, W // ds_ratio
                assert (H, W) == image_size, f"Downsampled shape {(H, W)} != expected {image_size}"

            # ============================================================
            # 4. Store attributes
            # ============================================================
            self.data = data  # (N, 1, H, W)
            self.x = x_coords
            self.y = y_coords
            self.mask = mask
            self.offset_patterns = offset_patterns
            self.original_indices = original_indices

            # Verify final shape and dtype
            assert self.data.shape == (N, 1, H, W), f"Final data shape mismatch"
            assert self.data.dtype == np.float32, f"Data dtype should be float32"

        else:
            raise NotImplementedError(f"Filetype {filetype} not supported")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns normalized fluctuation field for given index.

        Returns:
            image: (1, H, W) normalized fluctuation field in range [-1, 1]
        """
        image = self.data[idx]  # (1, H, W)

        if self.normalize:
            image = self.__normalize(image)

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __normalize(self, fluctuation):
        """
        Apply logarithmic normalization with sign preservation using centralized function.

        Args:
            fluctuation: (1, H, W) raw fluctuation field

        Returns:
            normalized: (1, H, W) normalized field in range [-1, 1]
        """
        return normalize_distribution(fluctuation, alpha=self.alpha, beta=self.beta)


class OneObs3D(Dataset):
    
    def __init__(self, data_root=None, data_file=None, filetype="hdf5", transform=None, ds_ratio=1):
        
        assert data_file is not None
        self.data_root = data_root
        self.data_file = data_file
        self.transform = transform
        self.ds_ratio = ds_ratio

        if filetype == "hdf5":
            
            data_arr = read_hdf5(osp.join(self.data_root, self.data_file))
            print(f"{data_file} opened")

            self.data = (data_arr['u_fluc'], data_arr['v_fluc'], data_arr['w_fluc'])
            assert len(self.data[0]) == len(self.data[1]) and  len(self.data[1]) == len(self.data[2])
            
            self.data_len = len(self.data[0])

        else:
            raise NotImplementedError
        
            
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        sample = torch.stack([torch.as_tensor(self.data[j][idx], dtype=torch.float32) for j in range(len(self.data))], dim=0)
        # shape = (num_channels, dim1, dim2, dim3, ..,)
        assert sample.shape == (3,144,63,26)

        if torch.any(torch.isnan(sample)):
           sample[torch.isnan(sample)] = 0
           print(f"NaNs found in data samples at idx={idx} in {self.data_file}, replaced NaNs with 0")     

        if self.ds_ratio == 1:
            padded_sample = torch.zeros((3,144,64,32), dtype=torch.float32)  #TODO: efficient way to do this using torch.nn.functional.pad
            padded_sample[:,:,:63,:26] = sample
            sample = padded_sample
            assert sample.shape == (3,144,64,32)

        elif self.ds_ratio == 2:
            sample = sample[:, ::2, ::2, ::2]
            #print(f"sample shape is {sample.shape}")
            assert sample.shape == (3, 72, 32, 13)
            padded_sample = torch.zeros((3,72,32,16), dtype=torch.float32)  #TODO: efficient way to do this using torch.nn.functional.pad
            padded_sample[:, :, :, :13] = sample
            sample = padded_sample
            assert sample.shape == (3,72,32,16)

        else:
            raise NotImplementedError

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def get_transform(transform):
    if transform is None:
        return None
    else:
        raise NotImplementedError


def get_dataset(config):

    if config.dataset.name.lower() == "oneobs2d":
        
        return OneObs2D(data_file = config.dataset.data_file, filetype = config.dataset.filetype.lower(), 
                        transform = get_transform(config.dataset.transform), ds_ratio = config.dataset.ds_ratio,
                        normalize = config.dataset.normalize, image_size = config.model.image_size)

    elif config.dataset.name.lower() == "plif2d":

        return PLIF2D(data_file = config.dataset.data_file,
                      filetype = config.dataset.filetype.lower(),
                      transform = get_transform(config.dataset.transform),
                      ds_ratio = config.dataset.ds_ratio,
                      normalize = config.dataset.normalize,
                      image_size = config.model.image_size,
                      alpha = getattr(config.dataset, "alpha", 1.0),
                      beta = getattr(config.dataset, "beta", 7.0))

    elif config.dataset.name.lower() == "oneobs3d":
        
        if isinstance(config.dataset.data_file, list):

            dsets = [OneObs3D(data_root=config.dataset.data_root, data_file=dset, 
                            filetype = config.dataset.filetype.lower(), transform = get_transform(config.dataset.transform), ds_ratio = config.dataset.ds_ratio) for dset in config.dataset.data_file]
                
            return torch.utils.data.Subset(torch.utils.data.ConcatDataset(dsets), indices=torch.arange(50000))

        
        elif isinstance(config.dataset.data_file, str):
            
            return OneObs3D(data_root=config.dataset.data_root, data_file=config.dataset.data_file, 
                            filetype = config.dataset.filetype.lower(), transform = get_transform(config.dataset.transform), ds_ratio = config.dataset.ds_ratio)   
        
        else:
            raise NotImplementedError
    
    else:
        raise NotImplementedError