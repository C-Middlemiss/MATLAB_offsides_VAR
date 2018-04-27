I = imread('clip2_still.png');
I = imresize(I,[224 224]);
%imshow(I)
net = googlenet;
inputSize = net.Layers(1).InputSize;

[label,scores] = classify(net,I);
label
