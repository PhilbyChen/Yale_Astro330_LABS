from astropy.io import fits
import os

def load_fits(filepath, extension = 0, *args):
    '''
    Write a function which takes as its argument a string filepath to a FITS file, 
    and should have an optional argument to set the extension (default 0). 
    It should then load the given extension of that fits file using a context manager, 
    and return a tuple containing the header and data of that extension.
    '''
    '''
    Load a specific extension from a FITS file.
    
    Parameters:
    filepath : str
        Path to the FITS file.
        
    extension : int, optional
        Extension number to load (default 0).
        
    Returns:
    header : astropy.io.fits.header.Header
        Header object from the specified extension.
        
    data : numpy.ndarray or None
        Data array from the specified extension.
        Returns None if extension has no data.
    '''
    with fits.open(filepath) as hdul:
        hdu = hdul[extension]
        return hdul.header.copy(), hdu.data
    pass