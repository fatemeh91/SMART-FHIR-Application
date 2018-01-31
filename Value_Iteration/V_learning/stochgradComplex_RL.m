function [par_vec, history, exitflag] = stochgradComplex_RL(objFun, w0, options, RL_data, funObj_validate, dims)

opt_alg='RMSProp';
Nall=numel(RL_data);
exitflag = [];   history.w = []; history.fval = []; history.fval_validation = [];
verbose = true; MaxEpoch = options.MaxEpoch; w  = w0; par_vec = w0; history.w = w0'; par_vec_old=par_vec;
Momentum_factor = options.Momentum_factor;

minFuncArgs = []; minFuncArgs.derivativeCheck = 'off'; minFuncArgs.display = 'off';
minFuncArgs.MaxIter = options.maxStochIter; minFuncArgs.TolFun = 1e-4; minFuncArgs.TolX = 1e-4;
minFuncArgs.Method = 'lbfgs'; minFuncArgs.GradObj = options.GradObj;
Delta0 = 1; Deltamin = 0.1; Deltamax = 5.0; eta_minus = 0.5; eta_plus = 1.2;
Delta = ones(length(w0),1)*Delta0;    df_old = zeros(length(w0),1);    MeanSquare = ones(length(w0),1);

eta = 1/(Deltamax+Deltamin);  M = @(t)Momentum_factor^(2/t);  rho=0.9;
rng(1000,'twister');
%% Main loop
for nupdates=1:MaxEpoch
    indices = randperm(Nall);
    batch_ind = indices(1:min(round(min(0.2+0.1*nupdates,0.7)*Nall),12*8));
    batch_indv = indices(max(round(0.7*Nall),Nall-12*8):Nall);
    bdata = RL_data(batch_ind);                vdata = RL_data(batch_indv);
    % evaluate value of all actions at state S_t+1
    st=1;WL=cell(dims.num_layers,1);
    for l=1:dims.num_layers
        WL{l} = reshape(par_vec(st:st-1+prod(dims.sizeWL{l})), dims.sizeWL{l}(1), dims.sizeWL{l}(2));
        st=st+prod(dims.sizeWL{l});
    end
    VStp1=cell(numel(batch_ind),1);
    parfor iindx=1:numel(batch_ind)
        VStp1{iindx} = evalVNetwork(bdata{iindx}.state_tp1, WL); %weights at previous iteration
    end
    VStp1_v=cell(numel(batch_indv),1);
    parfor iindx=1:numel(batch_indv)
        VStp1_v{iindx} = evalVNetwork(vdata{iindx}.state_tp1, WL);
    end
    
    if strcmp(opt_alg,'minFunc')
        try    [wnew, ~, exitflag] = minFunc(objFun, par_vec, minFuncArgs, bdata, VStp1); df = par_vec - wnew; % implicit gradient
        catch, disp('stochgradComplex: minFunc did not evaluate ...!'); [~, df] = objFun(w, bdata, VStp1);
        end
        % Update params
        w = w - eta*df;
    elseif strcmp(opt_alg,'RMSProp')
        try    wnew = minFunc(objFun, par_vec, minFuncArgs, bdata, VStp1); df = par_vec - wnew; % implicit gradient
        catch, disp('stochgradComplex: minFunc did not evaluate ...!'); [~, df] = objFun(w, bdata, VStp1);
        end
        if sum(abs(df))<eps, [~, df] = objFun(w, bdata, VStp1); end
        MeanSquare = rho * MeanSquare + (1-rho) * df.^2;
        positive = df.*df_old > 0; % gradient sign did not change
        negative = df.*df_old < 0; % gradient sign did change
        % update of step lengths
        Delta(positive) = min(Delta(positive)*eta_plus, Deltamax);
        Delta(negative) = max(Delta(negative)*eta_minus, Deltamin);
        % rprop update
        w = par_vec + M(nupdates) * (par_vec - par_vec_old) + (1-M(nupdates)) * (- eta * Delta .* df ./ sqrt(MeanSquare+1e-6));
        df_old = df;
    end
    % Parameter averaging
    if ~any(isnan(w)), par_vec_old=par_vec; par_vec = w;  end
    history.w = [history.w ; par_vec'];
    %training performance
    try    ftr= funObj_validate(par_vec, batch_ind, VStp1);
    catch, disp('stochgradComplex: funObj_validate did not evaluate on tr ...!'); ftr = NaN;
    end
    if nupdates==1, ftrAvg = ftr; history.fval = [history.fval ; ftrAvg];
    elseif ~isnan(ftr), ftrAvg = ftrAvg - 0.3 * (ftrAvg - ftr);   history.fval = [history.fval ; ftrAvg];
    else,               history.fval = [history.fval ; NaN];
    end
    %validation performance
    try    fvl= funObj_validate(par_vec, batch_indv, VStp1_v);
    catch, disp('stochgradComplex: funObj_validate did not evaluate on vl ...!'); fvl = NaN;
    end
    if nupdates==1, fvlAvg = fvl; history.fval_validation = [history.fval_validation ; fvlAvg];
    elseif ~isnan(fvl), fvlAvg = fvlAvg - 0.3 * (fvlAvg - fvl);  history.fval_validation = [history.fval_validation ; fvlAvg];
    else,               history.fval_validation = [history.fval_validation ; NaN];
    end
    if verbose,	  fprintf('nupdates so far %d : Tr_Perf %f, Val_Perf %f \n', nupdates, history.fval(end) ,history.fval_validation(end)); end
    
    if nupdates>50 && (history.fval_validation(end)  -history.fval_validation(end-1)) / history.fval_validation(end-1)< 1e-3 && ...
            (history.fval_validation(end-1)-history.fval_validation(end-2)) / history.fval_validation(end-2) < 1e-3 && ...
            (history.fval_validation(end-2)-history.fval_validation(end-3)) / history.fval_validation(end-3) < 1e-3 && ...
            (history.fval_validation(end-3)-history.fval_validation(end-4)) / history.fval_validation(end-4) < 1e-3
        break;
    end
end % next iteration
prf=history.fval_validation; prf(1:10)=NaN;
mx=nanmax(prf); ind=find(history.fval_validation(1:end)==mx,1,'last');
par_vec = history.w(ind,:);
if verbose, fprintf('finished after %d updates\n', nupdates); end

end
% http://www.willamette.edu/~gorr/classes/cs449/intro.html
% http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf