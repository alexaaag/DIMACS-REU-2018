function [Us,TXmean,odrIdx]  = UMPCA(TX,numP)
% UMPCA: Uncorrelated Multilinear Principle Component Analysis
%
% %[Prototype]%
% function [Us,TXmean,odrIdx]  = UMPCA(TX,numP)
%
% %[Author Notes]%
% Author: Haiping LU
% Email : hplu@ieee.org   or   eehplu@gmail.com
% Release date: February 28, 2012 (Version 1.0)
% Please email me if you have any problem, question or suggestion
%
% %[Algorithm]%:
% This function implements the Uncorrelated Multilinear Principal Component
% Analysis (UMPCA) algorithm presented in the follwing paper:
%    Haiping Lu, K.N. Plataniotis, and A.N. Venetsanopoulos,
%    "Uncorrelated Multilinear Principal Component Analysis for Unsupervised Multilinear Subspace Learning",
%    IEEE Transactions on Neural Networks,
%    Vol. 20, No. 11, Page: 1820-1836, Nov. 2009.
% Please reference this paper when reporting work done using this code.
%
% %[Toolbox needed]%:
% Matlab Tensor Toolbox (included in this package)
% source: http://csmr.ca.sandia.gov/~tgkolda/TensorToolbox/
%
% %[Syntax]%: [Us,TXmean,odrIdx]  = UMPCA(TX,numP)
%
% %[Inputs]%:
%    TX: the input training data in tensorial representation, the last mode
%        is the sample mode. For Nth-order tensor data, TX is of 
%        (N+1)th-order with the (N+1)-mode to be the sample mode.
%        E.g., 30x20x10x100 for 100 samples of size 30x20x10
%        If your training data is too big, resulting in the "out of memory"
%        error, you could work around this problem by reading samples one 
%        by one from the harddisk, or you could email me for help.
%
%    numP: the dimension of the projected vector, denoted as P in the
%          paper. It is the number of elementary multilinear projections 
%          (EMPs) in tensor-to-vector projection.
%
%
% %[Outputs]%:
%    Us: the multilinear projection, consiting of numP (P in the paper) 
%        elementary multilinear projections (EMPs), each EMP is consisted
%        of N vectors, one in each mode 
%
%    TXmean: the mean of the input training samples TX
%
%    odrIdx: the ordering index of projected features in decreasing  
%            variance 
%
%
% %[Supported tensor order]%
% This function supports N=2,3,4, for other order N, please modify the
% codes accordingly or email hplu@ieee.org or eehplu@gmail.com for help
%
% %[Examples]%
%%%%%%%%%%%%%%%%%%%%%%%%%%Example on 2D face data%%%%%%%%%%%%%%%%%%%%%%%%%%
%       load FERETC70A15S8/FERETC70A15S8_80x80%each sample is a second-order tensor of size 80x80
%       N=ndims(fea2D)-1;%Order of the tensor sample
%       Is=size(fea2D);%80x80x721
%       numSpl=Is(3);%There are 721 face samples
%       numP=80;
%       load('FERETC70A15S8/3Train/1');%load partition for 3 samples per class
%       fea2D_Train = fea2D(:,:,trainIdx);
%       [Us,TXmean,odrIdx]  = UMPCA(fea2D,numP);
%       fea2D=fea2D-repmat(TXmean,[ones(1,N), numSpl]);%Centering
%       numP=length(odrIdx);
%       newfea = zeros(numSpl,numP);
%       for iP=1:numP
%           projFtr=ttv(tensor(fea2D),Us(:,iP),[1 2]);
%           newfea(:,iP)=projFtr.data;
%       end
%       newfea = newfea(:,odrIdx);%newfea is the final feature vector to be 
%       %fed into a standard classifier (e.g., nearest neighbor classifier)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% %[Notes]%:
% A. Developed using Matlab R2006a & Matlab Tensor Toolbox 2.1
% B. Revision history:
%       Version 1.0 released on February 28, 2012
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%TX: (N+1)-dimensional tensor Tensor Sample Dimension x NumSamples
N=ndims(TX)-1;%The order of samples.
IsTX=size(TX);
Is=IsTX(1:N);%The dimensions of the tensor

%Please see Corollary 1 in the TNN paper: numP<=min(min(Is),M}
if numP>min(Is), numP=min(Is);end
numSpl=IsTX(N+1);%Number of samples
if numSpl<numP, numP=numSpl-1;end%Centering makes the samples to be dependent, hence rank lowered by 1

%%%%%%%%%%%%%Zero-Mean%%%%%%%%%%
TXmean=mean(TX,N+1);%The mean
TX=TX-repmat(TXmean,[ones(1,N), numSpl]);%Centering

%%%%%%%%%%%%%%%UMPCA parameters%%%%%%%%%%%%%%%%
maxK=10; %maximum number of iterations, you can change this number
Us=cell(N,numP);
Us0=cell(N,1);
for iP=1:numP %Get each EMP one by one
    %Initialization
    for n=1:N
        if iP==1
            Un=ones(Is(n),1);
            Un=Un/norm(Un);
            Us0{n}=Un;
        end
        Us{n,iP}=Us0{n};
    end
    %End Initialization
    
    %Start iterations
    for k=1:maxK
        for n=1:N
            switch N
                case 2
                    switch n
                        case 1
                            Ypn=ttv(tensor(TX),Us(2,iP),2);
                        case 2
                            Ypn=ttv(tensor(TX),Us(1,iP),1);
                    end
                case 3
                    switch n
                        case 1
                            Ypn=ttv(tensor(TX),Us(2:3,iP),[2 3]);
                        case 2
                            Ypn=ttv(tensor(TX),Us([1,3],iP),[1 3]);
                        case 3
                            Ypn=ttv(tensor(TX),Us(1:2,iP),[1 2]);
                    end
                case 4
                    switch n
                        case 1
                            Ypn=ttv(tensor(TX),Us(2:4,iP),[2 3 4]);
                        case 2
                            Ypn=ttv(tensor(TX),Us([1,3,4],iP),[1 3 4]);
                        case 3
                            Ypn=ttv(tensor(TX),Us([1,2,4],iP),[1 2 4]);
                        case 3
                            Ypn=ttv(tensor(TX),Us(1:3,iP),[1 2 3]);
                    end
                otherwise
                    error('Order N not supported. Please modify the code here or email hplu@ieee.org for help.')
            end
            Ypn=Ypn.data;
            ST=zeros(Is(n));
            for i=1:numSpl
                YDiff=Ypn(:,i);
                ST=ST+YDiff*YDiff'; %Within-class Scatter
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if iP>1
                GYYG=Gps'*Ypn'*Ypn*Gps; %equation (16) in the paper
                ThetaST=(eye(Is(n))-Ypn*Gps*inv(GYYG)*Gps'*Ypn'); %equation (15) in the paper
                ST=ThetaST*ST;
                [Lmdn,Un]=maxeig(ST);
            else
                [Lmdn,Un]=maxeig(ST);
            end
            Un=Un/norm(Un);
            Us{n,iP}=Un;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    %%%%%%%%%%%%%%%%%%%%%%%Projection%%%%%%%%%%%%%%%%%%%%%%%%%%%
    gp=ttv(tensor(TX),Us(:,iP),1:N);
    gp=gp.data;
    if iP==1
        Gps=gp;
    else
        Gps=[Gps gp];
    end
    %%%%%%%%%%%%%%%%%%%%%%%End Projection%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%%%%%%%%%%%%%%%%%%%%%%%%Sort by Variance%%%%%%%%%%%%%%%%%%%%%%%%%
vecYps=Gps';%vecYps contains the feature vectors for training data
Ymean=mean(vecYps,2);%Should be zero
TVars=diag(vecYps*vecYps');%Calculate variance
[stTVars,odrIdx]=sort(TVars,'descend');
odrIdx=odrIdx(1:numP); %Take the first numP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%