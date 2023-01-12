% Script for plotting and analysing the precision-recall curve from 
% two sets of confusion matrices loaded from .csv files.

clc
close all
clear

%% Loading Data

CD = readtable("CD_confusions.csv");
AD = readtable("AD_confusions.csv");

%% Computation of Precision, Recall and F-measures

AD_precision = AD.TP./(AD.TP + AD.FP);
AD_recall = AD.TP./(AD.TP + AD.FN);
AD_F_scores = 2*(AD_precision.*AD_recall)./(AD_precision+AD_recall);

CD_precision = CD.TP./(CD.TP + CD.FP);
CD_recall = CD.TP./(CD.TP + CD.FN);
CD_F_scores = 2*(CD_precision.*CD_recall)./(CD_precision+CD_recall);


%% Selection of best F-measures

[AD_max, AD_max_i] = max(AD_F_scores);
best_threshold_AD = AD.threshold(AD_max_i)
best_F_AD = AD_max

[CD_max, CD_max_i] = max(CD_F_scores);
best_threshold_CD = CD.threshold(CD_max_i)
best_F_CD = CD_max

%% Plotting

line_thickness = 3;

subplot 231
hold on
plot(CD.threshold, CD_precision, 'linewidth', line_thickness)
plot(CD.threshold(CD_max_i), CD_precision(CD_max_i), 'o', 'MarkerSize', 10, 'linewidth', 2)
title("Center Distance - Precision")
xlabel("Threshold [mm]")
ylabel("Precision")
grid on

subplot 232
hold on
plot(CD.threshold, CD_recall, 'linewidth', line_thickness)
plot(CD.threshold(CD_max_i), CD_recall(CD_max_i), 'o', 'MarkerSize', 10, 'linewidth', 2)
title("Center Distance - Recall")
xlabel("Threshold [mm]")
ylabel("Recall")
grid on

subplot 233
hold on
plot(CD_recall, CD_precision, 'linewidth', line_thickness)
plot(CD_recall(CD_max_i), CD_precision(CD_max_i), 'o', 'MarkerSize', 10, 'linewidth', 2)
title("Center Distance - Precision-Recall")
xlabel("Recall")
ylabel("Precision")
grid on

subplot 234
hold on
plot(AD.threshold, AD_precision, 'linewidth', line_thickness)
plot(AD.threshold(AD_max_i), AD_precision(AD_max_i), 'o', 'MarkerSize', 10, 'linewidth', 2)
title("Average Distance - Precision")
xlabel("Threshold [mm]")
ylabel("Precision")
grid on

subplot 235
hold on
plot(AD.threshold, AD_recall, 'linewidth', line_thickness)
plot(AD.threshold(AD_max_i), AD_recall(AD_max_i), 'o', 'MarkerSize', 10, 'linewidth', 2)
title("Average Distance - Recall")
xlabel("Threshold [mm]")
ylabel("Recall")
grid on

subplot 236
hold on
plot(AD_recall, AD_precision, 'linewidth', line_thickness)
plot(AD_recall(AD_max_i), AD_precision(AD_max_i), 'o', 'MarkerSize', 10, 'linewidth', 2)
title("Average Distance - Precision-Recall")
xlabel("Recall")
ylabel("Precision")
grid on
