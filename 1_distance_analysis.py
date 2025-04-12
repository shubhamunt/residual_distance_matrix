import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from multiprocessing import Pool, cpu_count  # Import multiprocessing
import time

pdb_file = "wt.pdb"
output_image = "WT1.png"
title = "WT1"
output_matrix = "WT1_average_matrix.txt"
protein_atoms = 8239
offset = 70
max_frames=25000
# non-standard residues, if any, "3JD", "PO4", "RPB" are examples here
NS1 = "3JD" 
NS2 = "PO4"
NS3 = "RPB"

atomic_mass_dictionary = {
    'H': 1.0, 'HA':1.0,'HB':1.0,'HC': 1.0,'HD': 1.0,'HE': 1.0, 'HO': 1.0, 'HS': 1.0, 'HW': 1.0, 'H2': 1.0, 'H3': 1.0, 'HB1':1.0, 'HB2':1.0,'HB3':1.0,
    'HG':1.0,'HG1':1.0, 'HG2':1.0,'HG3':1.0,'HG11':1.0, 'HG12':1.0,'HG13':1.0,'HG21':1.0, 'HG22':1.0,'HG23':1.0,'HE1':1.0, 'HE2':1.0,'HE3':1.0,'HD1':1.0, 
    'HD2':1.0,'HD3':1.0,'HH11':1.0, 'HH12':1.0,'HH21':1.0, 'HH22':1.0,'HD21':1.0,'HD22':1.0,'HD23':1.0,
    'HZ':1.0,'HZ1':1.0, 'HZ2':1.0,'HZ3':1.0,
    'C': 12.0, 'CA': 12.0, 'CB': 12.0, 'CC': 12.0, 'CK': 12.0, 'CM': 12.0,
    'CN': 12.0, 'CQ': 12.0, 'CR': 12.0, 'CT': 12.0, 'CV': 12.0, 'CW': 12.0,
    'CD': 12.0,'CD1': 12.0, 'CE': 12.0, 'CF': 12.0, 'CG': 12.0, 'CH': 12.0, 'CI': 12.0,
    'CJ': 12.0, 'CP': 12.0, 'C2': 12.0, 'C3': 12.0, 'CZ':12.0,'CZ1':12.0,'CZ2':12.0,'CZ3':12.0,
    'N': 14.0, 'NA': 14.0, 'NB': 14.0, 'NC': 14.0, 'NT': 14.0, 'N2': 14.0,'NZ':1.0,
    'N3': 14.0, 'N*': 14.0,
    'O': 16.0, 'OH': 16.0, 'OS': 16.0, 'OW': 16.0, 'O2': 16.0,'OG': 16.0,'OE1': 16.0,'OE2': 16.0,
    'S': 32.1, 'SH': 32.1, 'SD': 32.1, 
    'P': 31.0,
    'CU': 63.5, 'CO': 40.1, 'I': 35.4, 'IM': 35.4, 'MG': 24.3,
    'QC': 133.0, 'QK': 39.1, 'QL': 6.9, 'QN': 23.0, 'QR': 85.5
}

residue_map = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H","HIP": "H", "HIE": "H","ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "NS1": "X1", "NS2": "X2", "NS3": "X3"  # Non-standard residues mapped to 'X'
    }


def read_Frame1(pdb_file):
    frame_lines = []  # List to store lines of Frame1
    with open(pdb_file, "r") as file:
        for i, line_content in enumerate(file, start=2):
            if i > protein_atoms:
                break  # Stop after reading the specified number of atoms
            frame_lines.append(line_content)  # Add the line to the list
    return frame_lines  # function exit

Frame1 = read_Frame1(pdb_file)

def generate_fasta_from_Frame1(frame_lines, residue_list):
    fasta_list = []  # Store the sequence
    seen_residue_ids = set()  # Track processed residue IDs

    # Read from Frame1 to generate a FASTA sequence
    for line in frame_lines:  
        if line.startswith("ATOM"): 
            residue_name = line[17:20].strip()  # Extract residue name
            residue_id = line[22:26].strip()  # Extract residue ID (to track uniqueness)

            # Add only if residue has not been processed yet for this ID
            if residue_id not in seen_residue_ids:
                seen_residue_ids.add(residue_id)  # Mark as processed
                if residue_name in residue_list:
                    fasta_list.append(residue_map.get(residue_name, "?"))  # Convert to 1-letter, default as "?"
                else:
                    fasta_list.append("?")  # Handle non-standard residues

    return fasta_list  # Returns the FASTA sequence as a list

residue_list = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS","HIP","HIE" ,"ILE", 
                "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]

# Read Frame1
#Frame1 = read_Frame1(pdb_file)

# Run the function
fasta_sequence = generate_fasta_from_Frame1(Frame1, residue_list)  # Process Frame1
print("FASTA sequence:\n")
print("".join(fasta_sequence))  # Convert list to string for display

# Count the number of residues
number_of_residues = len(fasta_sequence)
print("\nNumber of residues identified : ", number_of_residues )

####################################################################################################################

def residual_com(frame, atomic_mass_dictionary):
    residue_com_dictionary = {}  # Dictionary to store COM data for each residue
    
    # Iterate over each line in the frame
    for line in frame:  
        if line.startswith("ATOM"):  # Process only ATOM lines
            atom_name = line[12:16].strip()  # Extract atom name
            residue_name = line[17:20].strip()  # Extract residue name
            residue_id = int(line[22:26].strip())  # Extract residue ID and convert to integer
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])  # Extract coordinates
            
            # Get atomic mass from dictionary, default to 12.0 if unknown
            atomic_mass = atomic_mass_dictionary.get(atom_name, 12.0)

            # Initialize residue entry if not already present
            if residue_id not in residue_com_dictionary:
                residue_com_dictionary[residue_id] = {"total_mass": 0, "mass_weighted_x": 0, "mass_weighted_y": 0, "mass_weighted_z": 0}

            # Accumulate mass-weighted coordinates and total mass
            residue_com_dictionary[residue_id]["total_mass"] += atomic_mass
            residue_com_dictionary[residue_id]["mass_weighted_x"] += atomic_mass * x  # += --> Summation
            residue_com_dictionary[residue_id]["mass_weighted_y"] += atomic_mass * y
            residue_com_dictionary[residue_id]["mass_weighted_z"] += atomic_mass * z

            # Print the dictionary before computing COM
            #print("Intermediate Dictionary (Before Computing COM):")
             #for res_id, values in residue_com.items():
             #print(f"Residue {res_id}: {values}")

    # Compute final center of mass for each residue
    for res_id, values in residue_com_dictionary.items():
        total_mass = values["total_mass"]           # values[v] --> extracts the value "v" from a dictionary 
        com_x = round(values["mass_weighted_x"] / total_mass, 1) 
        com_y = round(values["mass_weighted_y"] / total_mass, 1)
        com_z = round(values["mass_weighted_z"] / total_mass, 1)
        
        residue_com_dictionary[res_id] = (com_x, com_y, com_z)  # Replace dict with tuple COM

    return residue_com_dictionary

#print(residue_com_dictionary)

# Load the PDB file
#pdb_file = "/Users/shubhamchatterjee/Desktop/DISTANCE_ANALYSIS/md1_1-10frames.pdb"
#with open(pdb_file, "r") as file:
#    frame1 = file.readlines()  # Read all lines into a list

# Call the function (store the returned value)
residue_com_dictionary = residual_com(Frame1, atomic_mass_dictionary)

#print("\nCenter of Mass for Each Residue:\n")
#for key, value in residue_com_dictionary.items():
#    print(f"Residue {key}: COM = {value}")

###############################################################################################################################
def compute_distance_matrix(residue_com_dictionary):
    # Convert residue center-of-mass values to a NumPy array of shape (number_of_residues, 3)
    com_coords = np.array(list(residue_com_dictionary.values()), dtype=np.float16)  # Use float8 to limit precision
    
    # Compute pairwise Euclidean distances between all residues
    # using cdist. It is more efficient than numpy module for distance calculations
    com_distance_matrix = cdist(com_coords, com_coords, metric='euclidean').astype(np.float16)  # Compute in float8
    
    # Round the entire distance matrix to one decimal place
    #matrix_B = np.round(matrix_B, 1)
    com_distance_matrix
    return com_distance_matrix

###########################################################################################################################
# Function to read a single frame from the PDB trajectory file
def read_frame(file, protein_atoms):
    frame_lines = []  # List to store lines of the current frame
    for i in range(protein_atoms):
        line = file.readline()
        if not line:  # End of file
            break
        frame_lines.append(line)
    return frame_lines
###############################################################################################################################

start_time = time.time()

# Main function to process the PDB trajectory file
def process_pdb_trajectory(pdb_file, protein_atoms, max_frames):
    distance_matrices = []  # List to store distance matrices for each frame
    
    i = 0  # i is frame number being processed

    with open(pdb_file, "r") as file:
        
        while i < max_frames:
            
            frame_lines = read_frame(file, protein_atoms)
            
            if len(frame_lines) == 0:  # End of file
                break

            # Process the frame to compute the distance matrix
            residue_com_dictionary = residual_com(frame_lines, atomic_mass_dictionary)
            com_distance_matrix = compute_distance_matrix(residue_com_dictionary)
            distance_matrices.append(com_distance_matrix)
            
            i += 1
            print("\nFrame being processed: ", i)
            print("\nDistance matrix: \n", com_distance_matrix)

    return distance_matrices

# Call the function to process the PDB trajectory file
distance_matrices = process_pdb_trajectory(pdb_file, protein_atoms, max_frames)

sum_matrix = np.zeros((number_of_residues, number_of_residues))
valid_frames = 0

for index, value in enumerate(distance_matrices):             
    # Check if the current matrix has the desired shape
    if np.shape(value) == (number_of_residues, number_of_residues):
        sum_matrix += value
        valid_frames += 1
    else:
        print(f"\nSkipping frame {index + 1}: shape is {np.shape(value)} (expected {(number_of_residues, number_of_residues)})")

if valid_frames > 0:
    print("\nValid frames:" ,valid_frames)
    average_matrix = sum_matrix / valid_frames
    print("\nAverage Distance Matrix:\n")
    print(average_matrix.round(1))
    np.savetxt(output_matrix, average_matrix, fmt="%.1f", delimiter=" ", comments="")
    
else:
    print("No valid frames with the expected shape were found.")

end_time = time.time()    
print("\nReal world time", (end_time - start_time) / 60, "minutes")


# Visualize Distance Matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.imshow(average_matrix, cmap='viridis', origin='lower')

cbar = plt.colorbar()  
cbar.set_label('Distance (Å)', fontsize=12)  
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) # Format color bar to show two decimal places

# Rotate color bar labels by 90 degrees
#for label in cbar.ax.get_yticklabels():
#    label.set_rotation(30) 
    
plt.title(title)

# Generate tick positions and labels
tick_positions = np.arange(0, number_of_residues, 50)
tick_labels = tick_positions + offset

plt.xticks(tick_positions, tick_labels, fontsize=12, rotation=90)
plt.yticks(tick_positions, tick_labels, fontsize=12)

plt.xlabel('Residue Index', fontsize=15)
plt.ylabel('Residue Index', fontsize=15)

plt.savefig(output_image, dpi=350,bbox_inches='tight')
#plt.show()
