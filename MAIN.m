%%% BRAIN TUMOUR  DETECTION AND CLASSIFICATION)%%% 

clc;

close all;

clear all;

warning('off','all');

%%TO READ THE INPUT IMAGE%%%

[FileName,PathName] = uigetfile('*.jpg;*.png;*.bmp','Pick an MRI Image');
if isequal(FileName,0)||isequal(PathName,0)
    warndlg('User Press Cancel');
else
    
a = imread([PathName,FileName]);
figure,imshow(a),title('INPUT IMAGE');


%%%%%%% READ THE INPUT IMAGE BY USING CAMERA %%%%%%%% 


%%%%%%%TO RESIZE THE INPUT IMAGE%%%%%%
a=imresize(a,[256 256]);
figure,imshow(a),title('RESIZED INPUT IMAGE');

%%%%%%TO CONVERT THE RESIZED INPUT IMAGE INTO GRAYSCALE IMAGE%%%%%%
[m n o]=size(a);
if o==3
    gray=rgb2gray(a);
else 
    gray=a;
end
end
 figure,imshow(gray),title('GRAY SCALE IMAGE');


%%%%%%%REGION BASED SEGMENTATION%%%%%%%
   
%%%%%TO ADJUST THE GRAY IMAGE%%%%%
bw1=imadjust(gray);
figure,imshow(bw1),title('CONTRAST ENHANCED GRAYSCALE IMAGE');
    
%%CONVERT TO BLACK AND WHITE IMAGE%%%
bw=im2bw(bw1,0.7);

figure,imshow(bw),title('BLACK AND WHITE IMAGE');

    
% %%LABEL THE BLACK AND WHITE IMAGE%%%
label=bwlabel(bw);

figure,imshow(label),title('LABELED IMAGE');
 
    %%%%SELECT THE REGION PROPERTY%%%%
      %%%%%TO CALL THE ALL PROPERTIES%%%%%
stats=regionprops(label,'all');
     
%%%%%TAKE  IMPORTANT PROPERTIES%%%%%%
area=[stats.Area];
centroid=[stats.Centroid];
majorAxisLength=[stats.MajorAxisLength];
minorAxisLength=[stats.MinorAxisLength];
eccentricity=[stats.Eccentricity];
orientation=[stats.Orientation];
filledArea=[stats.FilledArea];
equivdiameter=[stats.EquivDiameter];
density=[stats.Solidity];
perimeter=[stats.Perimeter];

%%%%%%FROM THE ALL PROPERTIES WE CHOOSE 'AREA' AND 'DENSITY' %%%%%%
 
%%%%CHOOSE THE HIGH DENSE AREA%%%%
high_dense_area=density>0.3;
  
%%%%CHOOSE THE MAX AREA WITH HIGH DENSITY%%%%
max_area=max(area(high_dense_area));
 
%%%%FIND THE MAXIMUM AREA%%%%
tumour_label=find(area==max_area);
  
%%%%LABEL THAT THE MAXIMUM AREA%%%%
tumour=ismember(label,tumour_label);

%%%%%%%DILATE  THE TUMOR AREA%%%%%%
se=strel('square',8);
tumour=imdilate(tumour,se);

 
%%%%TO SHOW THE TUMOR AREA ALONE%%%%%
figure,imshow(tumour,[]),title('TUMOR ALONE IMAGE');

 %%%%%TO SHOW THE DETECTED TUMOR IN THE INPUT IMAGE%%%%
[B,L]=bwboundaries(tumour,'noholes');
figure,imshow(a,[]);
hold on
for  i=1:length(B)
    plot(B{i}(:,2),B{i}(:,1),'g','linewidth',1.5)
end
title('DETECTED TUMOUR IMAGE')
hold off



%%%TO PLOT THE INPUT IMAGE,TUMOUR ALONE IMAGE ,AND DETECTED TUMOR ON INPUT IMAGE%%%
figure,
subplot(1,3,1),imshow(a,[]),title('INPUT TUMOR IMAGE');
subplot(1,3,2),imshow(tumour,[]),title('TUMOUR ALONE IMAGE');
[B,L]=bwboundaries(tumour,'noholes');
subplot(1,3,3),imshow(a,[]);
hold on
for  i=1:length(B)
    plot(B{i}(:,2),B{i}(:,1),'g','linewidth',1.5)
end
title('TUMOUR DETECTED')
hold off
set(gcf, 'Position', get(0,'Screensize'));

ho=tumour;
%%%% TO SPEED UP THE PROCESS RESIZE THE IMAGE INTO SMALL SIZE  %%%%%%

re=imresize(ho,[200 200]);
figure,imshow(re),title('RESIZED SMALL IMAGE');

%%%%% CONVERT THE DATA TYPE INTO UNSIGNED INTEGER %%%%%
re=im2uint8(re);
%     imwrite(re,'8 .jpg');


%%%%% TRAIN THE DATASET IMAGES %%%%%
 
matlabroot='C:\Users\SPIRO-IMAGE\Desktop\FINISHED CODES\2018a codes\BRAIN_CNN_FINISHED\dataset';

Data=imageDatastore(matlabroot,'IncludeSubfolders',true,'LabelSource','foldernames');

%%%%%% CREATE CONVOLUTIONAL NEURAL NETWORK LAYERS %%%%%%


layers=[imageInputLayer([200 200 1])  
    
convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,256,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options=trainingOptions('sgdm','MaxEpochs',15,'initialLearnRate',0.01,'Plots','training-progress');


convnet=trainNetwork(Data,layers,options);


%%%%% TO  CLASSIFY THE OUTPUT %%%%%%% 

output=classify(convnet,re);
tf1=[];

for ii=1:2
    st=int2str(ii)
    tf=ismember(output,st);
    tf1=[tf1 tf];
end

output1=find(tf1==1);

if output1==2
    
    figure,imshow(a);
    hold on;
    
    stats=regionprops(tumour,'all');
    area=[stats.Area];
    
    if(area<5000)
        
    msgbox('MALIGNANT TUMOR');
    warndlg('normal stage');
    disp('MALIGNANT TUMOR-normal stage');
    
    elseif(area>=5000&&area<10000)
        
        msgbox('MALIGNANT TUMOR');
         warndlg('medium stage');
        disp('MALIGNANT TUMOR-medium stage');
        
    elseif(area>=10000)
        msgbox('MALIGNANT TUMOR');
        warndlg('severe stage');
    disp('MALIGNANT TUMOR- stage');
    
    hold off;
    
    end
   
    ccf=2;
    
elseif output1==1
    
    figure,imshow(a);
    hold on;
    
    stats=regionprops(tumour,'all');
    area=[stats.Area];
    
    if(area<1000)
        
    msgbox('BENIGN TUMOR');
    warndlg('normal stage');
    disp('BENIGN TUMOR-normal stage');
    
    elseif(area>=1000&&area<2000)
        
        msgbox('BENIGN TUMOR');
         warndlg('medium stage');
         disp('BENIGN TUMOR-medium stage');
    
    elseif(area>=2000)
        msgbox('BENIGN TUMOR');
        warndlg('severe stage');
        disp('BENIGN TUMOR- stage');
        
        hold off;
        
    end
   
    ccf=1;
end   



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%end%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%