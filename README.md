# Trajectory_estimation_on_sound
Main codes for Master's Thesis of Eike Bergen at Tsinghua University (Professor Shuguang Li).
Trajectory prediction of micro vibrating robots using multimodal learning.
Helper scripts create wav files from raw csv data and from them heat plot visualizations (spectrograms).
A CNN is trained on these images (containing 1 second of acoustic data) with a camera image (merged trajectory of 1 second) as label.
CNN Architechture based on Chen et al. 2021: "The Boombox: Visual Reconstruction from Acoustic Vibrations" (https://arxiv.org/abs/2105.08052)
