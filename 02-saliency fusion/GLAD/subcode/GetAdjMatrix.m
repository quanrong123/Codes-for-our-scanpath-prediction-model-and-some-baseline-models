function [adjMatrixSecond bd]= GetAdjMatrix(idxImg, spNum)
% Get adjacent matrix of super-pixels
% idxImg is an integer image, values in [1..spNum]

% Code Author: Wangjiang Zhu
% Email: wangjiang88119@gmail.com
% Date: 3/24/2014

[h, w] = size(idxImg);

%Get edge pixel locations (4-neighbor)
topbotDiff = diff(idxImg, 1, 1) ~= 0;
topEdgeIdx = find( padarray(topbotDiff, [1 0], false, 'post') ); %those pixels on the top of an edge
botEdgeIdx = topEdgeIdx + 1;

leftrightDiff = diff(idxImg, 1, 2) ~= 0;
leftEdgeIdx = find( padarray(leftrightDiff, [0 1], false, 'post') ); %those pixels on the left of an edge
rightEdgeIdx = leftEdgeIdx + h;

%Get adjacent matrix of super-pixels
adjMatrix = zeros(spNum, spNum);
adjMatrix( sub2ind([spNum, spNum], idxImg(topEdgeIdx), idxImg(botEdgeIdx)) ) = 1;
adjMatrix( sub2ind([spNum, spNum], idxImg(leftEdgeIdx), idxImg(rightEdgeIdx)) ) = 1;
adjMatrix = adjMatrix + adjMatrix';
adjMatrix(1:spNum+1:end) = 1;%set diagonal elements to 1


% superpixels on all the four image boundaries are connected 
bd=unique([idxImg(1,:),idxImg(h,:),idxImg(:,1)',idxImg(:,w)']);
for i=1:length(bd)
    for j=i+1:length(bd)
        adjMatrix(bd(i),bd(j))=1;
        adjMatrix(bd(j),bd(i))=1;
    end
end

%{
% superpixels on each image boundary are connected
boundry_t = unique(idxImg(1,:));
for i=1:length(boundry_t)
    for j=i+1:length(boundry_t)
        adjMatrix(boundry_t(i),boundry_t(j))=1;
        adjMatrix(boundry_t(j),boundry_t(i))=1;
    end
end
boundry_d = unique(idxImg(h,:));
for i=1:length(boundry_d)
    for j=i+1:length(boundry_d)
        adjMatrix(boundry_d(i),boundry_d(j))=1;
        adjMatrix(boundry_d(j),boundry_d(i))=1;
    end
end
boundry_r = unique(idxImg(:,1));
for i=1:length(boundry_r)
    for j=i+1:length(boundry_r)
        adjMatrix(boundry_r(i),boundry_r(j))=1;
        adjMatrix(boundry_r(j),boundry_r(i))=1;
    end
end
boundry_l = unique(idxImg(:,w));
for i=1:length(boundry_l)
    for j=i+1:length(boundry_l)
        adjMatrix(boundry_l(i),boundry_l(j))=1;
        adjMatrix(boundry_l(j),boundry_l(i))=1;
    end
end
%}

% find the second outerboundary superpixel
 adjMatrixSecond = adjMatrix; 
for i=1:spNum
    siteline=find( adjMatrix(i,:)>0 );
    lenthsiteline=length(siteline); 
    for j=1:lenthsiteline
        adjMatrixSecond(i,:)= adjMatrixSecond(i,:)+ adjMatrix( siteline( j ), :);
    end
end

adjMatrixSecond = sparse(adjMatrixSecond);
adjMatrix = sparse(adjMatrix);








