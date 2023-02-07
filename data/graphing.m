clear 
clc
close all

%% LOADING .CSV FILES

% t_loss = readtable("screwpose_transformation_loss.csv");
% ad_s = readtable("screwpose_AD_S.csv");
% add_s = readtable("screwpose_ADD_S.csv");

% t_loss = readtable("screwdataset_transformation_loss.csv");
% ad_s = readtable("screwdataset_AD_S.csv");
% add_s = readtable("screwdataset_ADD_S.csv");

t_loss = readtable("buttonpose_transformation_loss.csv");
ad_s = readtable("buttonpose_AD_S.csv");
add_s = readtable("buttonpose_ADD_S.csv");

%% PLOTTING

line_thickness = 2;

f = figure;
f.Position = [100 100 700 300];

subplot 121

hold on
grid on

plot(ad_s.Step, ad_s.Value, 'linewidth', line_thickness)
plot(t_loss.Step, t_loss.Value, 'linewidth', line_thickness)


% set(gca, "YScale", "log");

xlabel("Epoch")
ylabel("Value [mm]")

legend("AD-S on test set", "Transformation Loss")

subplot 122

plot(add_s.Step, add_s.Value, 'linewidth', line_thickness)
grid on

xlabel("Epoch")

legend("ADD-S on test set")

