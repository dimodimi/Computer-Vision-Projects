function [ features ] = extractSurfFeatures(I, intPoints)
%Returns the SURF descriptors for each point of interest in an image I (Nx64 matrix)

features = zeros(size(intPoints, 1), 64);

%Zero pad the image in case a point lies close to the edges of I
[M, N] = size(I);
I2 = zeros(M+20, N+20);
I2(11:end-10, 11:end-10) = I;

%Haar wavelets
haarX = [-1 1];
haarY = [-1;1];

%20x20 gaussian
sigma = 3.3;
len = 10;
n = [-len:-1 1:len];
[n1, n2] = meshgrid(n, n);
gaussian = exp(-(n1.^2 + n2.^2)/(2*(sigma^2)));


for i = 1:size(intPoints, 1)
    %20x20 region around a point of interest
    reg = I2(intPoints(i, 1) +1 : intPoints(i,1) + 20, intPoints(i, 2) +1 : intPoints(i, 2) + 20);
    
    %Split the region into 16 subregions
    for s1 = 0:3
        for s2 = 0:3
            %For each subregion find the Haar wavelet response in both the
            %x and y axis
            subreg = reg(5*s1+1:5*s1+5, 5*s2+1:5*s2+5);
            dx = conv2(subreg, haarX, 'same');
            dy = conv2(subreg, haarY, 'same');
            
            %Gaussian weigting
            g = gaussian(5*s1+1:5*s1+5, 5*s2+1:5*s2+5);
            
            dx = dx .* g;
            dy = dy .* g;
            
            features(i, 16*s1 + 4*s2 + 1) = sum(sum(dx));
            features(i, 16*s1 + 4*s2 + 2) = sum(sum(abs(dx)));
            features(i, 16*s1 + 4*s2 + 3) = sum(sum(dy));
            features(i, 16*s1 + 4*s2 + 4) = sum(sum(abs(dy)));
        end
    end
end

%Normalize vectors - invariance to scaling
features = normr(features);

end

