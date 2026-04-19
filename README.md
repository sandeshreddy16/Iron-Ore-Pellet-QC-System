# Iron Ore Pellet Quality Control System
Automated pellet classification using Computer Vision and YOLOv8

## About
This system automatically detects and classifies iron ore 
pellets from images using AI. It measures each pellet diameter 
in mm and classifies it as OK, Small, Large or Shape Reject.

## Result
- Images Processed : 150
- Total Pellets    : 11523
- OK               : 10882  
- Small            : 179     
- Large            : 384    
- Shape Reject     : 78     
- Pass Rate        : 94.4%  

## Classification
- Green  = OK (8–18mm)
- Orange = Small (below 8mm)
- Red    = Large (above 18mm)
- Pink   = Shape Reject

## Technologies
- Python, OpenCV, YOLOv8

## How To Run
pip install opencv-python ultralytics scikit-image scipy pandas matplotlib

python final_pellet_system.py

## Developer
Sandesh Reddy 
