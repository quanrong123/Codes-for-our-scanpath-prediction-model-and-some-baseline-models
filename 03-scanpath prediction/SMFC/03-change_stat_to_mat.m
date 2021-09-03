clear all; clc;
statspath = 'F:\scanpath prediction\comparison\SMFC\';
statspathDir = dir([statspath, '*.stat']);
num_stats = length(statspathDir);
resultspath =  'F:\scanpath prediction\comparison\SMFC1\';
mkdir(resultspath)
for i = 1:num_stats
    stat_name = [statspath, statspathDir(i).name];
    res_name = [resultspath, statspathDir(i).name(1:end-5), '.mat'];
    a = importdata(stat_name);
    %predicted_fixations = cell(1,1);
    %for j = 1:1
    fixations = zeros(1,2);
    b = a(1,:);
    for k = 1:10
        c = b(3*(k-1)+1:3*(k-1)+2);
        fixations(k,:) = [c(2), c(1)];
    end
    predicted_fixations = fixations;
    %end
    save(res_name, 'predicted_fixations')


end














