function A = double(T)
%DOUBLE Convert ttensor to double array.
%
%   A = double(T) converts T to a standard multidimensional array.
%
%   See also TTENSOR.
%
%MATLAB Tensor Toolbox.
%Copyright 2006, Sandia Corporation. 

% This is the MATLAB Tensor Toolbox by Brett Bader and Tamara Kolda. 
% http://csmr.ca.sandia.gov/~tgkolda/TensorToolbox.
% Copyright (2006) Sandia Corporation. Under the terms of Contract
% DE-AC04-94AL85000, there is a non-exclusive license for use of this
% work by or on behalf of the U.S. Government. Export of this data may
% require a license from the United States Government.
% The full license terms can be found in tensor_toolbox/LICENSE.txt
% $Id: double.m,v 1.4 2006/08/21 21:04:40 bwbader Exp $

A = double(full(T));
