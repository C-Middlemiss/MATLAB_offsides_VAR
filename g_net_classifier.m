function [label] = g_net_classifier(I)
%uses function of googlenet to classify and image 
%  label = ballplayer/person means player
I = imresize(I,[224 224]);
imshow(I)
net = googlenet;
inputSize = net.Layers(1).InputSize;

[label,scores] = classify(net,I);
label

end

