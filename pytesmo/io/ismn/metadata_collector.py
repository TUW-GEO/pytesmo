#Copyright (c) 2013,Vienna University of Technology, Department of Geodesy and Geoinformation
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology, Department of Geodesy and Geoinformation nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL Vienna University of Technology, 
#Department of Geodesy and Geoinformation BE LIABLE FOR ANY
#DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on Aug 1, 2013

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''


import os
import readers
import numpy as np

def collect_from_folder(rootdir):
    """
    function walks the rootdir directory and looks for network
    folders and ISMN datafiles. It collects metadata for every
    file found and returns a numpy.ndarray of metadata
    
    Parameters
    ----------
    rootdir : string
        root directory on filesystem where the ISMN data was unzipped to
    
    Returns
    -------
    metadata : numpy.ndarray
        structured numpy array which contains the metadata for one file per row
    """
    
    
    metadata_catalog=[]
    for root, subFolders, files in os.walk(rootdir):
        #print root,subFolders,files
        for filename in files:
            fullfilename = os.path.join(root,filename)
            try:
                metadata = readers.get_metadata(fullfilename)
            except (readers.ReaderException,IOError) as e:
                continue    
            
            for i, variable in enumerate(metadata['variable']):
                
                metadata_catalog.append((metadata['network'],metadata['station'],
                                         variable,metadata['depth_from'][i],metadata['depth_to'][i],
                                         metadata['sensor'],metadata['longitude'],metadata['latitude'],
                                         metadata['elevation'],fullfilename))
                
     
    return np.array(metadata_catalog,dtype=np.dtype([('network',object),('station',object),('variable',object),
                                                       ('depth_from',np.float),('depth_to',np.float),
                                                       ('sensor',object),('longitude',np.float),('latitude',np.float),
                                                       ('elevation',np.float),('filename',object)]))
        
        


