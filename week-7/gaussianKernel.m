function sim = gaussianKernel(x1, x2, sigma)
  %RBFKERNEL returns a radial basis function kernel between x1 and x2
  %   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
  %   and returns the value in sim

  sim = e ^ (- (sum((x1(:) - x2(:)) .^ 2)) / (2 * sigma ^ 2));
end
