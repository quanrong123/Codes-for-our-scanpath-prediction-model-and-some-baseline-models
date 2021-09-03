clear all;
addpath(genpath('./Walther_Koch_Saliency/'))
imgspath = '/disk3/quanrong/scanpath_prediction/saliency_maps/';
resultspath = '/disk3/quanrong/scanpath_prediction/seqs/';
mkdir(resultspath)
imgspathDir = dir([imgspath, '*.jpg']);
num_images = length(imgspathDir);
for i = 1:num_images
    image_name = imgspathDir(i).name;    
    map_path = [imgspath, image_name];
    save_path = [resultspath, image_name(1:end-4), '.mat'];
    saliency_map_blur = imread(map_path);   
    sm = double(saliency_map_blur);
    nf = 15;
    smSize = size(sm);
    sm = imresize(sm, [128, 171]);
    sm = (sm - min(sm(:))) / (max(sm(:)) - min(sm(:)));
   %% inhibitation of return and fov processing
    % first initialize parameters
    params = defaultSaliencyParams;
    
    saliencyData.origImage.size = [128 171 1];
    saliencyData.data = imresize(sm, [48 64]);
    saliencyData.parameters = params;

    wta = initializeWTA(saliencyData, params);
    fixation_sequences = zeros(nf, 2);
    for fix = 1:nf
        % evolve WTA until we have the next winner
        winner = [-1,-1];    
        while(winner(1) == -1)
            [wta,winner] = evolveWTA(wta);
        end
        %fprintf('.');    
        % get shape data and apply inhibition of return    
        wta = applyIOR(wta, winner, params, []);   
        % convert the winner to image coordinates
        fixation_sequences(fix,:) = winnerToImgCoords(winner, params);
    end
    %fprintf('\n'); 
    sm = imresize(sm, smSize);
    fixation_sequences(:,1) = ceil(fixation_sequences(:,1) - 1e-6)*smSize(1)/768;
    fixation_sequences(:,2) = ceil(fixation_sequences(:,2) - 1e-6)*smSize(2)/1024;
    fixation_sequences = round(fixation_sequences);
  
    save(save_path, 'fixation_sequences')  
end































