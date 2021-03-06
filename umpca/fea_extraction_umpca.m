function fea_extraction_umpca
%% read the train matrices
nTrain=14;
T_train=zeros(1520,1628,nTrain);
trainDir = 'matrices/train'; %gets directory
trainPattern = fullfile(trainDir, '*.csv'); 
trainFiles = dir(trainPattern);

for k = 1 : nTrain
  baseFileName = trainFiles(k).name;
  fullFileName = fullfile(trainDir, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  T_train(:,:,k) = csvread(fullFileName,1,1);
end

%% read the test matrices
nTest=6;
T_test=zeros(1520,1628,nTest);
testDir = 'matrices/test'; %gets directory
testPattern = fullfile(testDir, '*.csv'); 
testFiles = dir(testPattern);

for k = 1 : nTest
  baseFileName = testFiles(k).name;
  fullFileName = fullfile(testDir, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  T_test(:,:,k) = csvread(fullFileName,1,1);
end

%% UMPCA on train set
N=ndims(T_train)-1;% Order of the tensor sample
Is=size(T_train);% 1520x1628x16
numSpl=Is(3);%There are 16 samples
numP=4; %set number of projections to 4
[Us_train,TXmean_train,odrIdx]  = UMPCA(T_train,numP); %run UMPCA
T=T_train-repmat(TXmean_train,[ones(1,N), numSpl]);%Centering
numP=length(odrIdx);   

% calculate the new features 
newfea_train = zeros(numSpl,numP);
for iP=1:numP
    projFtr=ttv(tensor(T),Us_train(:,iP),[1 2]);
    newfea_train(:,iP)=projFtr.data;
end
newfea_train = newfea_train(:,odrIdx);   % new features of train
csvwrite(fullfile('newfea_train.csv'),newfea_train);  %export new features of train set
size(newfea_train)

%% TEST UMPCA
N=ndims(T_test)-1;% Order of the tensor sample
Is=size(T_test);% 1520x1628x4
numSpl=Is(3);%There are 4 samples
numP=4;
[Us_test,TXmean,odrIdx]  = UMPCA(T_test,numP); % run UMPCA
T=T_test-repmat(TXmean,[ones(1,N), numSpl]);%Centering
numP=length(odrIdx);   

% calculate the new features 
newfea_test = zeros(numSpl,numP);
for iP=1:numP
    projFtr=ttv(tensor(T),Us_test(:,iP),[1 2]);
    newfea_test(:,iP)=projFtr.data;
end
newfea_test = newfea_test(:,odrIdx);   % new features of test
csvwrite(fullfile('newfea_test.csv'),newfea_test); 
size(newfea_test)

%% reconstruct the projections
ncomp=numP;
proj=cell(ncomp,1); % store each projection component
for i = 1:ncomp
u1=Us_train{1,i};
u2=Us_train{2,i};
U=u1*transpose(u2);
proj{i}=U;    
end
%% reconstruct the first sample using the four EMPs
export_dir = 'reconstruction';
tHat=zeros(1520,1628,ncomp);

% export the reconstructed matrix using the 4 components
for j = 1:ncomp
    recons=newfea_train(1,j)*proj{j};%+TXmean_train;
    tHat(:,:,j)=recons;
    csvwrite(fullfile(export_dir,sprintf('%s_%d%s','hat',j,'.csv')),recons); % export reconstructed matrices
end

%%
contourf(tHat(:,:,4))

%% RECONSTRUCTION USING ALL SAMPLES
T=zeros(1520,1628,20);
T(:,:,1:14)=T_train;
T(:,:,15:20)=T_test;

N=ndims(T)-1;% Order of the tensor sample
Is=size(T);% 1520x1628x20
numSpl=Is(3);
numP=4;
[Us,TXmean,odrIdx]  = UMPCA(T,numP); % run UMPCA
T=T-repmat(TXmean,[ones(1,N), numSpl]);%Centering
numP=length(odrIdx);   

% calculate the new features 
newfea = zeros(numSpl,numP);
for iP=1:numP
    projFtr=ttv(tensor(T),Us(:,iP),[1 2]);
    newfea(:,iP)=projFtr.data;
end
newfea = newfea(:,odrIdx);   

ncomp=numP;
proj=cell(ncomp,1); % store each projection component
for i = 1:ncomp
u1=Us{1,i};
u2=Us{2,i};
U=u1*transpose(u2);
proj{i}=U;    
end

export_dir = 'reconstruction';
tHat=zeros(1520,1628,ncomp);

% export the reconstructed matrix using the 4 components
for j = 1:ncomp
    recons=newfea(1,j)*proj{j};%+TXmean;   % the reconstructions look very similar after adding the mean
    tHat(:,:,j)=recons;
    csvwrite(fullfile(export_dir,sprintf('%s_%d%s','hat',j,'full.csv')),recons);
end

end

