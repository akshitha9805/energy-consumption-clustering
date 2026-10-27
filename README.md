Energy Consumption Pattern Clustering using K-Means

Course: Machine Learning II (BAI702), Problem-Based Learning (PBL) activity Problem statement 14: Energy Consumption Pattern Clustering College: Sai Vidya Institute of Technology, Bengaluru, Dept. of CSE (AI & ML) Team members: <add names and USNs here> Guide / Course co-ordinator: <add name here>

1. Problem definition and objectives

Households use electricity differently across the day. This project builds an unsupervised learning system that groups days with similar energy consumption patterns using K-Means clustering.

Objectives:

Group daily load profiles into clusters of similar usage
Analyse the impact of feature selection and number of clusters (k)
Interpret each resulting group (for example morning peak, evening peak, low usage)
2. Dataset

UCI Machine Learning Repository: Individual Household Electric Power Consumption

Link: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
One household near Paris, France, Dec 2006 to Nov 2010
2,075,259 one-minute readings; about 1.25% missing values
Columns: date, time, global active power, global reactive power, voltage, global intensity, sub-metering 1-3
Licence: CC BY 4.0 (open data)
Attribute used for clustering: Global_active_power (kW)

The dataset is not stored in this repository (it is large). The script downloads it automatically.

3. Methodology
Cleaning: remove missing, negative, above-25 kW and duplicate readings
Hourly averaging: hours need at least 45 readings
Daily profiles: one row per day with 24 hourly values
Standardisation: StandardScaler
Feature selection study: 24 hourly values, 4 time-block means, summary statistics, hourly + weekend flag
Choosing k: elbow (inertia), silhouette, Davies-Bouldin, Calinski-Harabasz for k = 2 to 10
Final K-Means: n_init=30, random_state=42, plus stability check over random seeds
PCA (2-D) for visualisation only
Interpretation: cluster names from mean profiles, weekday/weekend share, season mix
4. How to run
bash
pip install -r requirements.txt
python energy_clustering.py

Optional: python energy_clustering.py --data household_power_consumption.txt or python energy_clustering.py --k 4.

Outputs are saved in the results/ folder.

5. Results and interpretation

Fill this section from your own results/report.txt.

Days used after cleaning: <N>
Chosen k: <k> (silhouette = <value>, Davies-Bouldin = <value>, Calinski-Harabasz = <value>)
Stability (mean ARI over random seeds): <value>
Cluster	Name	Days	Share %	Avg daily kWh	Peak hour
0	<name>				
1	<name>				

Cluster profiles Choosing k PCA view Feature selection

6. Conclusion and future scope

K-Means finds recurring daily usage patterns in real smart-meter data. Feature selection and the choice of k affect the grouping, and the stability score shows how reliable it is.

Future scope: Gaussian Mixture / DBSCAN, data from many homes (including Indian households), temperature and holiday features, cluster-based forecasting, and a dashboard with savings alerts.

7. Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn