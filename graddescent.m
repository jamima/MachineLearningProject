function [w, fHist] = graddescent(x, r, w0, stepsize, verbose)
% GRADDESCENT Gradient descent algorithm for logistic discriminant binary classifier.
%     Find the solution for logistic discriminant classifier given the 
%     data, starting solution and step size for gradient direction. The
%     function to optimize is loglikelihood of data
%     
%     INPUTS:
%         x           input variables
%         r           input targets
%         w0          starting solution for the weights
%         stepsize    step size parameter for gradient direction
%         verbose     0/1 value; print info to screen
%
%     OUTPUTS:
%         w           solution found by the algorithm
%         fHist       value of loglikelihood through the iterations
%    

	if (nargin < 5 || isempty(verbose))
		verbose = 0;
	end
	if (nargin < 4 || isempty(stepsize))
		stepsize = 0.01;
	end
	
	w = w0;
	maxIter = 2000;
	normTol = 1e-4;
	funTol = 1e-5;	
	
	if (verbose)
		fprintf('\n%10s%10s%12s\n', 'iteration', 'f(x)', 'step norm');
	end

	f_old = Inf;
	w_old = Inf;
	iter = 0;    
    fHist = [];
	while (true)
		iter = iter + 1;
		y = 1 ./ (1 + exp(-x * w));
		f = -sum(r .* log(y) + (1 - r) .* log(1 - y));       % compute loglike          
        fHist = [fHist; f];
		grad = ((r - y)'*x)';    % compute gradient
		w = w + stepsize * grad;                            % move in gradient direction		
		if (verbose)
			fprintf('%6d%14.2f%12.2f\n', iter, f, norm(w - w_old)); 
		end
		
		% check convergence
		if (iter >= maxIter || norm(w - w_old) < normTol || abs(f - f_old) < funTol)
			break;
		end
		
		f_old = f;
		w_old = w;
	end
	
	if (verbose)
		if (iter >= maxIter)
			fprintf('\nMaximum iterations exceeded (%d).\n\n', maxIter);
		end
		if (norm(w - w_old) < normTol)
			fprintf('\nChange in solution value less than tolerance.\n\n'); 
		end
		if (abs(f - f_old) < funTol)
			fprintf('\nChange in function value less than tolerance.\n\n'); 
		end
	end
end

