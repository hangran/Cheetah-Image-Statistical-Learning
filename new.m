
load('TrainingSamplesDCT_8.mat');

% (a)reasonable estimates for the prior probilities
feature_num=64;
cheetah_num=250;
grass_num=1053;
P_Cheetah= (cheetah_num/(cheetah_num+grass_num));
P_Grass=grass_num/(grass_num+cheetah_num);



% (b)for each vector, we compute the index (position within the vector)  
% of the coefficient that has the second largest energy value (absolute value)
FG = abs(TrainsampleDCT_FG);
BG = abs(TrainsampleDCT_BG);


 for i=1:grass_num
     p= sort(BG,2,'descend');
     x_index(i)=find (BG (i,:)== p(i,2));
 end
 for j=1:cheetah_num
     q= sort(FG,2,'descend');
     y_index(j)=find (FG (j,:)== q(j,2));
 end


% histogram and conditional probability

figure(1)
FG_hist=histogram(x_index,1:65,'Normalization','probability');
cheetah_con = FG_hist.Values;
title('P(x|cheetah)');
xlabel('i');


figure(2);
BG_hist=histogram(y_index,1:65,'Normalization','probability');
grass_con = BG_hist.Values;
title('P(y|grass)');
xlabel('j')


% (c)
% read image/files
zig=load('Zig-Zag Pattern.txt');
zig=zig+1;
zig=zig(:);

cheetach=imread('cheetah.bmp');
cheetach=im2double(cheetach);
[row, column] = size(cheetach);

% computing the dct and features
for m=1:(row-7)
  for n=1:(column-7)
      temp_row = zeros(1,64);
        index_DCT= abs(dct2(cheetach(m:m+7, n:n+7)));
        index_DCT= index_DCT(:);
     
        for k=1:64
            temp_row(zig(k)) = index_DCT(k);
        end
        
       s = sort(temp_row, 'descend');
       s2= s(2);
       lo= find(temp_row==s2);
       loc(m:m+7,n:n+7)=lo;
 end
end


% classifier
prob1=BG_hist.Values;
prob2=FG_hist.Values;
final = zeros(row,column);
for i3=1:row
    for j3=1:column
        l1=loc(i3,j3);
        PRBG=prob1(l1);
        PRFG=prob2(l1);
        if (PRBG*P_Cheetah > PRFG*P_Grass)
      
            final(i3,j3)=1;
        else
            final(i3,j3)=0;
        end
    end
end


figure(3);
imagesc(final);
colormap(gray(255));


% (d)
% compute the error of the mask

mask=imread('cheetah_mask.bmp');
mask=im2double (mask);
error= mask ~= final;
figure(4);
imagesc(error);
colormap(gray(255));

sm= sum(error);
error= sum(sm)/(row*column)

% compute the probability of error


% compute the probability of error
error_p = 0;
for i4 = 1: feature_num
    if final(i4) == 0
        error_p = error_p + P_Cheetah * prob1(i4);
    end
    if final(i4) == 1
        error_p = error_p + P_Grass* prob2(i4);
    end
end
error_p






