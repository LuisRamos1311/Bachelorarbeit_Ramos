# Bachelorarbeit_Ramos
Bachelorarbeit Wirtschaftsinformatik


Data aquired from
https://www.cryptodatadownload.com/data/gemini/
first work will be will daily data from 2015 to 2025-11-18

Installed talib from https://ta-lib.org/install/?utm_source=chatgpt.com#executable-installer-recommended

Created data_pipeline
Test prints:
dataset sizes
shape of the first sample
its label
first 3 timesteps of that sequence
detailed last timestep with feature names
it additionally prints the last 3 timesteps of the same sequence
This just helps you see how the window starts and how it ends 
(how indicators and prices evolve over the 30 days). 
It doesn’t change any data, it’s only for inspection and understanding.

Trying to expand the project to add tools that differentiate tft from others
GRNs, VSNs, known future inputs