function tensorDecomp
%%
T=zeros(1520,1628,6);
myDir = 'umpca_reconstruction/matrices'; %gets directory
filePattern = fullfile(myDir, '*.csv'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
n=length(theFiles);
add
for k = 1 : n
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(myDir, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  T(:,:,k) = csvread(fullFileName,1,1);
end

%% 
N=ndims(T)-1;% Order of the tensor sample
Is=size(T);% 1520x1628x20
numSpl=Is(3);%There are 20 samples
numP=10;
[Us,TXmean,odrIdx]  = UMPCA(T,numP); % run MPCA
T=T-repmat(TXmean,[ones(1,N), numSpl]);%Centering
numP=length(odrIdx);   

% calculate the new features 
newfea = zeros(numSpl,numP);
for iP=1:numP
    projFtr=ttv(tensor(T),Us(:,iP),[1 2]);
    newfea(:,iP)=projFtr.data;
end
newfea = newfea(:,odrIdx);


%% 
%newfea(5,1)
export_dir = '/Users/alexa712/Documents/School/DIMACS REU/analysis/data-fusion-3d-printing/umpca_reconstruction';
csvwrite(fullfile(export_dir,'newfea.csv'),newfea);
size(newfea)

%% reconstruction
% create the U matrices
ncomp=10;
proj=cell(ncomp,1);
for i = 1:ncomp
u1=Us{1,i};
u2=Us{2,i};
U=u1*transpose(u2);
proj{i}=U;    
end
%%

tHat=zeros(1520,1628,20,4);
for i = 1:20
    for j = 1:2
      recons=newfea(i,j)*proj{j};
      tHat(:,:,i,j)=recons;
      csvwrite(fullfile(export_dir,sprintf('%s_%d_%d%s','hat',i,j,'.csv')),recons);
    end
end

%%
contourf(tHat(:,:,2,3))
end

