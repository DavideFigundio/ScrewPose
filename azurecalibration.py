'''
Script that displays the intrinsic matrix and distortion
coefficents of an Azure Kinect camera.
'''

import pykinect_azure as pykinect

if __name__ == "__main__":
    pykinect.initialize_libraries()
    device_config = pykinect.default_configuration
    
    device = pykinect.start_device(config=device_config)

    print(device.calibration)