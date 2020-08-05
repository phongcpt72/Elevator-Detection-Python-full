# Elevator-Detection-Python-full

1. Target 
+ Detect when users using elevator with different holding styles. 
+ Output includes: start time and stop time, predict number of transition floors and going down-up when users using elevator.  

2. Data Collection 
+ Z axis earth acceleration and pressure sensors with sampling rate at 25Hz. 
+ 4 holding styles: Calling (CA), Texting (TE), Swinging (SW) and Pocket (PO) 
+ Apply Low Pass Filter (butter worth filter: order 2, sample rate 25Hz and cut-off frequency 1.5Hz)  

3. Features Extraction, Algorithm and Method 

3.1 Feature Extraction 
+ Mean 
+ Difference timestamp between Min – Max of Z axis Acceleration of Earth’s gravity 

3.2 Algorithm 
+ Dynamic time warping 
+ Linear Regression
