# Cascaded_PF_Tracking
// Experiment with color based PF.cpp : Cascaded Particle Filter with 2D Color-based Tracking( hue and saturation color histogram) and 1D Depth-based Tracking (geodesic distance)

// Input: real-time kinect camera video
// Output: Cascaded particle filtering tracking results

// Select a target (area) and its extremety point for tracking using mouse input on window "Color Probabilistic Tracking
// - Samples", and after selection, samples propagated are shown in window "Color Probabilistic 
// Tracking - Samples" and results are shown in window "Color Probabilistic Tracking - Result".

// Change the values for amplifiers "VAR_U", "VAR_V", ..., "VAR_HPRIME" to track targets with
// different mobility. 
//Typically, values for these amplifiers are set larger for tracking a 
// person with high mobility than for tracking a person with low mobility.
