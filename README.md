# Cascaded_PF_Tracking
Cascaded Particle Filter with 2D Color-based Tracking( hue and saturation color histogram) and 1D Depth-based Tracking (geodesic distance)


## I/O Specification
- Input: real-time kinect camera video
- Output: Cascaded particle filtering tracking results

## Usage
- Select a target (area) and its extremety point for tracking 
	- using mouse input in window "Color Probabilistic Tracking Samples"
- Propogated samples are shown in window "Color Probabilistic Tracking - Samples" after selection.
- Results are shown in window "Color Probabilistic Tracking - Result".
- Change the values for amplifiers "VAR_U", "VAR_V", ..., "VAR_HPRIME" to track targets with different mobility. 
	- Use large values for person with high mobility.
shitty
