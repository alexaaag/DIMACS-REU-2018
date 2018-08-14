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
Is=size(T_train);% 1520x1628x20
numSpl=Is(3);%There are 20 samples
numP=4;
[Us,TXmean,odrIdx]  = UMPCA(T_train,numP); % run MPCA
T=T_train-repmat(TXmean,[ones(1,N), numSpl]);%Centering
numP=length(odrIdx);   

% calculate the new features 
newfea_train = zeros(numSpl,numP);
for iP=1:numP
    projFtr=ttv(tensor(T),Us(:,iP),[1 2]);
    newfea_train(:,iP)=projFtr.data;
end
newfea_train = newfea_train(:,odrIdx);   % new features of train
%export_dir = '/Users/alexa712/Documents/School/DIMACS REU/DIMACS-REU-2018/umpca_reconstruction';
csvwrite(fullfile('newfea_train.csv'),newfea_train);
size(newfea_train)

%% TEST UMPCA
N=ndims(T_test)-1;% Order of the tensor sample
Is=size(T_test);% 1520x1628x20
numSpl=Is(3);%There are 20 samples
numP=4;
[Us,TXmean,odrIdx]  = UMPCA(T_test,numP); % run UMPCA
T=T_test-repmat(TXmean,[ones(1,N), numSpl]);%Centering
numP=length(odrIdx);   

% calculate the new features 
newfea_test = zeros(numSpl,numP);
for iP=1:numP
    projFtr=ttv(tensor(T),Us(:,iP),[1 2]);
    newfea_test(:,iP)=projFtr.data;
end
newfea_test = newfea_test(:,odrIdx);   % new features of test
csvwrite(fullfile(export_dir,'newfea_test.csv'),newfea_test);
size(newfea_test)

%% reconstruction
% create the U matrices
ncomp=numP;
proj=cell(ncomp,1);
for i = 1:ncomp
u1=Us{1,i};
u2=Us{2,i};
U=u1*transpose(u2);
proj{i}=U;    
end
%%
export_dir = '/Users/alexa712/Documents/School/DIMACS REU/DIMACS-REU-2018/umpca_reconstruction/reconstruction';
tHat=zeros(1520,1628,14,ncomp);
for i = 1
    for j = 1:ncomp
      recons=newfea(i,j)*proj{j};
      tHat(:,:,i,j)=recons;
      csvwrite(fullfile(export_dir,sprintf('%s_%d_%d%s','hat',i,j,'.csv')),recons);
    end
end

%%
contourf(tHat(:,:,1,2))
end

