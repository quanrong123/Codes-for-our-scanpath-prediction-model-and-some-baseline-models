clear all; clc;
directoryOrigPict = 'F:\scanpath prediction\datasets\images1\';
directoryOrigSal = 'F:\scanpath prediction\comparison\saliency_maps_pgm1\';
directoryOutput = 'F:\scanpath prediction\comparison\SMFC1\';
mkdir(directoryOutput)
scriptForGeneratingScanpaths(directoryOrigPict, directoryOrigSal, directoryOutput,1, 10, 8, 5, 22, 'naturalScenes')

















