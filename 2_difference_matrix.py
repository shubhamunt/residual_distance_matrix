import numpy as np

matrix_A1 = np.loadtxt("WT1_average_matrix.txt")
matrix_A2 = np.loadtxt("WT2_average_matrix.txt")
matrix_A3 = np.loadtxt("WT3_average_matrix.txt")

matrix_B1 = np.loadtxt("Mut1_average_matrix.txt")
matrix_B2 = np.loadtxt("Mut2_average_matrix.txt")
matrix_B3 = np.loadtxt("Mut3_average_matrix.txt")

# Step 1: Compute average matrices
avg_matrix_A = np.round(np.mean([matrix_A1, matrix_A2, matrix_A3], axis=0), 1)
avg_matrix_B = np.round(np.mean([matrix_B1, matrix_B2, matrix_B3], axis=0), 1)

# Step 2: Compute standard deviation of the average for each element
SD_avg_A = np.round(np.std([matrix_A1, matrix_A2, matrix_A3], axis=0, ddof=1) / np.sqrt(3), 1)  # Standard error of the mean
SD_avg_B = np.round(np.std([matrix_B1, matrix_B2, matrix_B3], axis=0, ddof=1) / np.sqrt(3), 1)  # Standard error of the mean

# Step 3: Compute the difference between average matrices
diff_B_A = avg_matrix_B - avg_matrix_A

# Step 4: Compute the standard deviation matrix for the difference (error propagation)
SD_diff_B_A = np.round(np.sqrt(SD_avg_A**2 + SD_avg_B**2), 1)

# Print results
print("Average Matrix A:\n", avg_matrix_A)
print("\nStandard Deviation of Average Matrix A:\n", SD_avg_A)
print("\nAverage Matrix B:\n", avg_matrix_B)
print("\nStandard Deviation of Average Matrix B:\n", SD_avg_B)
print("\nDifference Matrix (B - A):\n", diff_B_A)
print("\nStandard Deviation of Difference Matrix (B - A):\n", SD_diff_B_A)

min_diff_B_A = np.min(diff_B_A)
max_diff_B_A = np.max(diff_B_A)

min_SD_diff_B_A = np.min(SD_diff_B_A)
max_SD_diff_B_A = np.max(SD_diff_B_A)

print("\nmax_diff_B_A: ", np.round(max_diff_B_A, 1))
print("\nmin_diff_B_A: ", np.round(min_diff_B_A, 1))
z1 = np.max([np.abs(min_diff_B_A) , np.abs(max_diff_B_A)])
print("z1 = ",np.round (z1, 1))

print("\nmax_diff_SD_B-A: ", np.round(max_SD_diff_B_A, 1))
print("\nmin_diff_SD_B-A: ", np.round(min_SD_diff_B_A, 1))
z2 = np.max([np.abs(min_SD_diff_B_A) , np.abs(max_SD_diff_B_A)])
print("z2 = ",np.round (z2, 1))


# Visualize Average Distance Matrix A
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
output_image_1 = "Avg_matrix_WT"
output_image_2 = "SD_in_Avg_matrix_WT"
output_image_3 = "Avg_matrix_Mut"
output_image_4 = "SD_in_Avg_matrix_Mut"

number_of_residues = 514
offset = 70

plt.imshow(avg_matrix_A, cmap='viridis_r', origin='lower')

cbar = plt.colorbar()  
cbar.set_label('Distance (Å)', fontsize=12)  
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) 
plt.title('Average distance matrix WT', fontsize = 15)

# Generate tick positions and labels
tick_positions = np.arange(0, number_of_residues, 50)
tick_labels = tick_positions + offset

plt.xticks(tick_positions, tick_labels, fontsize=12, rotation=90)
plt.yticks(tick_positions, tick_labels, fontsize=12)

plt.xlabel('Residue Index', fontsize=15)
plt.ylabel('Residue Index', fontsize=15)

plt.savefig(output_image_1, dpi=350,bbox_inches='tight')
plt.show()
#############################################################################
# Visualize SD in Average Distance Matrix A

plt.imshow(SD_avg_A, cmap='viridis_r', origin='lower')

cbar = plt.colorbar()  
cbar.set_label('Distance (Å)', fontsize=12)  
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) 
plt.title('SD in average distance matrix WT', fontsize = 15)

# Generate tick positions and labels
tick_positions = np.arange(0, number_of_residues, 50)
tick_labels = tick_positions + offset

plt.xticks(tick_positions, tick_labels, fontsize=12,rotation=90)
plt.yticks(tick_positions, tick_labels, fontsize=12)

plt.xlabel('Residue Index', fontsize=15)
plt.ylabel('Residue Index', fontsize=15)

plt.savefig(output_image_2, dpi=350,bbox_inches='tight')
plt.show()

#############################################################################
# Visualize Average Distance Matrix B

plt.imshow(avg_matrix_B, cmap='viridis_r', origin='lower')

cbar = plt.colorbar()  
cbar.set_label('Distance (Å)', fontsize=12)  
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

plt.title('Average distance matrix Mut', fontsize = 15)

# Generate tick positions and labels
tick_positions = np.arange(0, number_of_residues, 50)
tick_labels = tick_positions + offset

plt.xticks(tick_positions, tick_labels, fontsize=12,rotation=90)
plt.yticks(tick_positions, tick_labels, fontsize=12)

plt.xlabel('Residue Index', fontsize=15)
plt.ylabel('Residue Index', fontsize=15)

plt.savefig(output_image_3, dpi=350,bbox_inches='tight')
plt.show()

#############################################################################
# Visualize SD in Average Distance Matrix B

plt.imshow(SD_avg_B, cmap='viridis_r', origin='lower')

cbar = plt.colorbar()  
cbar.set_label('Distance (Å)', fontsize=12)  
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) 
plt.title('SD in average distance matrix Mut', fontsize = 15)

# Generate tick positions and labels
tick_positions = np.arange(0, number_of_residues, 50)
tick_labels = tick_positions + offset

plt.xticks(tick_positions, tick_labels, fontsize=12, rotation=90)
plt.yticks(tick_positions, tick_labels, fontsize=12)

plt.xlabel('Residue Index', fontsize=15)
plt.ylabel('Residue Index', fontsize=15)

plt.savefig(output_image_4, dpi=350,bbox_inches='tight')
plt.show()


# Visualize Distance Matrix
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))
#plt.imshow(diff, cmap='Spectral_r', origin='lower', vmax = z1, vmin = (-1 * z1)) 
plt.imshow(diff_B_A, cmap='seismic', origin='lower', vmax = 1.5, vmin = -1.5)

# Create the colorbar
cbar = plt.colorbar(shrink = 0.8)  
cbar.set_label('Distance (Å)', fontsize = 12)  

plt.title('ΔDistance: Mut – WT')

# Generate tick positions and labels
tick_positions = np.arange(0, 514, 30)
tick_labels = tick_positions + offset

plt.xticks(tick_positions, tick_labels, fontsize=10, rotation = 90)
plt.yticks(tick_positions, tick_labels, fontsize=10)

plt.xlabel('Residue Index', fontsize=15)
plt.ylabel('Residue Index', fontsize=15)

#plt.grid(visible=True, color='gray', linewidth=0.2, linestyle=':')

ax = plt.gca()  # gca --> Get the current axes
ax.spines['top'].set_linewidth(1.1)    
ax.spines['right'].set_linewidth(1.1) 
ax.spines['bottom'].set_linewidth(1.1) 
ax.spines['left'].set_linewidth(1.1) 

plt.savefig('Mut-WT.png', dpi=450,bbox_inches='tight')
plt.show()


######################### SD ###############################################################

# Visualize Distance Matrix
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))
#plt.imshow(diff, cmap='Spectral_r', origin='lower', vmax = z1, vmin = (-1 * z1)) 
plt.imshow(SD_diff_B_A, cmap='viridis_r', origin='lower', vmax = 1.0, vmin = 0.0)

# Create the colorbar
cbar = plt.colorbar(shrink=0.8)  
cbar.set_label('Distance (Å)', fontsize=12)  

plt.title('SD in ΔDistance: Mut – WT')

# Generate tick positions and labels
tick_positions = np.arange(0, 514, 30)
tick_labels = tick_positions + offset

plt.xticks(tick_positions, tick_labels, fontsize=10, rotation = 90)
plt.yticks(tick_positions, tick_labels, fontsize=10)

plt.xlabel('Residue Index', fontsize=15)
plt.ylabel('Residue Index', fontsize=15)

#plt.grid(visible=True, color='gray', linewidth=0.2, linestyle=':')

ax = plt.gca()  # gca --> Get the current axes
ax.spines['top'].set_linewidth(1.1)    
ax.spines['right'].set_linewidth(1.1) 
ax.spines['bottom'].set_linewidth(1.1) 
ax.spines['left'].set_linewidth(1.1) 

plt.savefig('SD_Mut-WT.png', dpi=400,bbox_inches='tight')
plt.show()

