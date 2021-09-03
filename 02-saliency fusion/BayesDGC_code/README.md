
========================================================================

1. run ./generate_features/01-generate_features.py to obtain each superpixel's feature.
2. run 02-generate_saliency_label.m to get each superpixel several saliency labels from the candidate saliency maps.
3. run 03-BayesDGC.py to fuse each superpixel's saliency labels into one.
4. run 04-get_final_map.m  to obtain the fused saliency map.

========================================================================

