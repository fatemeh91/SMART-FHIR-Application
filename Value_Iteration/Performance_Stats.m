function stats = Performance_Stats(labels,scores,posclass,target_sensitivity,AUC_threshold,pltflg)

if nargin<4 || isempty(target_sensitivity), stats.target_sensitivity=0.85; else, stats.target_sensitivity=target_sensitivity; end
if nargin<5 || isempty(AUC_threshold), stats.AUC_threshold=NaN; else, stats.AUC_threshold=AUC_threshold; end
if nargin<6, pltflg=1;end
    
[stats.X,stats.Y,stats.T,stats.AUC,OPTROCPT] = perfcurve(labels,scores,posclass);
indSen = find(stats.Y>=stats.target_sensitivity,1,'first');
stats.OPTROCPT(1)=stats.X(indSen); stats.OPTROCPT(2)=stats.Y(indSen);
stats.P = sum(labels==posclass); stats.N = sum(labels~=posclass);
stats.FPR = stats.OPTROCPT(1);  stats.SPC = 1-stats.FPR;  stats.FP = stats.FPR * stats.N; 
stats.TN = stats.N - stats.FP; stats.TPR = stats.OPTROCPT(2);  stats.SEN = stats.TPR;    stats.TP = stats.TPR * stats.P; 
stats.FN = stats.P - stats.TP; 
stats.ACC = (stats.TP + stats.TN) / (stats.P + stats.N); stats.PPV = stats.TP /(stats.TP + stats.FP); 
stats.NPV = stats.TN / (stats.TN + stats.FN); stats.FDR = stats.FP / (stats.FP + stats.TP);

if isnan(stats.AUC_threshold)
    stats.AUC_threshold = stats.T(stats.X==stats.OPTROCPT(1) & stats.Y==stats.OPTROCPT(2));
end
stats.accuracy_assessment=assessment(labels,double([scores>stats.AUC_threshold]+(posclass==2)),'class'); 
[stats.Xpr,stats.Ypr,stats.Tpr,stats.AUCpr,stats.OPTROCPTpr] = perfcurve(labels, scores,posclass, 'xCrit', 'reca', 'yCrit', 'prec'); 

if pltflg
disp(['AUCroc: ' num2str(round(1000*stats.AUC)/1000)])
disp(['Sensitivity: ' num2str(round(stats.SEN*1000)/1000)])
disp(['Specificity: ' num2str(round(stats.SPC*1000)/1000)])
disp(['PPV: ' num2str(round(stats.PPV*1000)/1000)])
disp(['Accuracy: ' num2str(round(stats.ACC*1000)/1000)])
disp(['AUCpr: ' num2str(round(1000*stats.AUCpr)/1000)])
disp('--------------------------------------------------')
end