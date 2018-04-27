function [label] = untitled(I)
%uses function of googlenet to classify and image 
%  label = ballplayer/person means player
I = imresize(I,[224 224]);
net = googlenet;
inputSize = net.Layers(1).InputSize;

[label,scores] = classify(net,I);
label

end

