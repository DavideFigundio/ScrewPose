'''
Simple script to display color video from an Azure Kinect camera.
'''

import pykinect_azure as pykinect
import cv2

def main():
    print("Setting up Azure Kinect...")
    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

	# Modify camera configuration
    device_config = pykinect.default_configuration

	# Start device
    device = pykinect.start_device(config=device_config)

    cv2.namedWindow('Image',cv2.WINDOW_NORMAL)

    i = 0
    max = 500
    while i < max:
        # Get capture
        capture = device.update()

		# Get the color image from the capture
        ret, image = capture.get_color_image()
        
        if not ret:
            continue
        
        print(str(i) + "/" + str(max), end = "\r")
        filename = "./images/" + str(i) + ".png"        
        
        cv2.imshow('Image', image)
        #cv2.imwrite(filename, image.copy())
        i += 1

if __name__ == "__main__":
    main()