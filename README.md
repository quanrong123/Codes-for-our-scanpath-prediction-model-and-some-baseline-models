# Codes-for-the-baseline-models
This project contains the codes for the baseline models compared in previous rebuttal. 

=======================================================

To run the codes, the reviewer should first run the four saliency detection models in "Candidate_saliency_maps_generation" folder to obtain the candidate saliency maps, and then fuse the candidate saliency maps into one. In our setting, we fuse the candidate saliency maps on superpixel level. The concrete whole operation process is as follows:
1. Segment each input image into 2000 superpixels by running "main.m" in "01-superpixel segmentation" folder.
2. Use BayesDGC and GLAD models (in "02-saliency fusion") to fuse the candidate saliency maps, respectively. 
3. Exploit SMFC and WTA-IOR models (in "03-scanpath prediction") to generate scanpaths from the fused saliency maps. 

=======================================================

We run the baseline models on the MIT and OSIE datasets. The original images, as well as the candidate saliency maps of these two datasets can be found in Baidu Wangpan:

MIT: https://pan.baidu.com/s/1rm2_16bl3P5x4mV-1FUePg Fetch Code: dhc3

OSIE: https://pan.baidu.com/s/1RA3wMnoc9u3bdFiyC1HipQ Fetch Code: 2z4p
