%% import training data
tic
Gamma = zeros(31266,190);

for i = 1:100
    s = "./frontalimages_spatiallynormalized_cropped_equalized_part1/"+int2str(i)+"a.jpg";
    Gamma(:,i) = reshape(imread(s),[31266,1]);
end
for i = 101:190
    s = "./frontalimages_spatiallynormalized_cropped_equalized_part2/"+int2str(i)+"a.jpg";
    Gamma(:,i) = reshape(imread(s),[31266,1]);
end
%%
Psi = sum(Gamma,2)/190;
A = Gamma - Psi;
C = A*A';
% [V,D] = eig(A'*A); % A'A has same eigenvalues as C
[U,S,V] = svd(A'*A);
V = normc(A*V); % eigenvectors of C are A*V
%% part a) plotting eigenvalues
fa = figure;
plot(1:190,diag(S));
ylabel('\sigma_i');
xlabel('index')
title('singular values of data matrix')

%% part b) determining number of components
% PCs are eigenvectors with the largest eigenvalues
s = "./frontalimages_spatiallynormalized_cropped_equalized_part1/43a.jpg";
Phi = double(reshape(imread(s),[31266,1]))-Psi;
error_n = zeros(190,1);
for i=1:190 % i number of PCs
    eigfaces = V(:,1:i);
    Phi_hat = eigfaces*(eigfaces'*Phi); % project onto eigenvectors
    error_n(i) = immse(Phi_hat,Phi);
end
fb = figure;
plot(1:190,error_n)
title('MSE vs Number of PCs (43a.jpg)');
ylabel('Mean Squared Error')
xlabel('Number of Principle Components')

figure
imshow(reshape(uint8(Phi_hat + Psi),[193,162]))
%% part c) 
s = "./frontalimages_spatiallynormalized_cropped_equalized_part1/43b.jpg";
smile = double(reshape(imread(s),[31266,1]));
Phi_s = smile - Psi;
error_s = zeros(190,1);
for i=1:190 % i number of PCs
    eigfaces = V(:,1:i);
    Phi_hat = eigfaces*(eigfaces'*Phi_s); % project onto eigenvectors
    error_s(i) = immse(Phi_hat,Phi_s);
end
fc = figure;
plot(1:190,error_s)
title('MSE vs Number of PCs (43b.jpg)');
ylabel('Mean Squared Error')
xlabel('Number of Principle Components')

figure
imshow(reshape(uint8(Phi_hat + Psi),[193,162]))
%% part d) 
s = "./frontalimages_spatiallynormalized_cropped_equalized_part2/193a.jpg";
newface = double(reshape(imread(s),[31266,1]));
Phi_nf = newface - Psi;
error_nf = zeros(190,1);
for i=1:190 % i number of PCs
    eigfaces = V(:,1:i);
    Phi_hat = eigfaces*(eigfaces'*Phi_nf); % project onto eigenvectors
    error_nf(i) = immse(Phi_hat,Phi_nf);
end
fd = figure;
plot(1:190,error_nf)
title('MSE vs Number of PCs (193a.jpg)');
ylabel('Mean Squared Error')
xlabel('Number of Principle Components')

figure
imshow(reshape(uint8(Phi_hat + Psi),[193,162]))
%% part e)
s = "./pizza.jpeg";
pizza = rgb2gray(imread(s));
pizza = imresize(pizza, [193,162]);
Phi_p = double(reshape(pizza, [31266,1])) - Psi;
Phi_hat = V*(V'*Phi_p); % project onto eigenvectors
error_p = immse(Phi_hat,Phi_p);
pizza_reconst = reshape(uint8(Phi_hat + Psi),[193,162]);

fe = figure;
imshow([pizza pizza_reconst])
%% part f)
s = "./frontalimages_spatiallynormalized_cropped_equalized_part1/43a.jpg";
face = imread(s);
figure
imshow(face);
error_r = zeros(73,1);
count = 1;

ff2 = figure;
for i = 0:72
    face_r = imrotate(face,i*5,'bilinear','crop');
    phi_r = double(reshape(face_r,[31266,1]))-Psi;
    phi_hat = V*(V'*phi_r); % project onto eigenvectors
    error_r(i+1) = immse(phi_hat,phi_r);
    if isequal(mod(i*5,45),0) && ne(360,i*5)
        subplot(2,8,count)
        imshow(int8(reshape(phi_r,[193,162])))
        subplot(2,8,count+8)
        imshow(int8(reshape(phi_hat,[193,162])))
        count = count + 1;
    end
end

ff1 = figure;
plot(0:5:360,error_r)
title('MSE vs Rotation (43a.jpg)');
ylabel('Mean Squared Error')
xlabel('Rotation Angle (degrees)')

%%
toc