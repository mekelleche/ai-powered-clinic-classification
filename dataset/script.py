import os
import matplotlib.pyplot as plt

# 1. Parse your CSV row data into a dictionary
# (Gender 1 usually implies Male or Female depending on dataset; we'll assume Female here)
data = {
    "WBC": 10.63, "NE#": 6.31, "LY#": 2.79, "MO#": 0.91, "EO#": 0.56, "BA#": 0.06,
    "RBC": 4.31, "HGB": 12.7, "HCT": 37.6, "MCV": 87.2, "MCH": 29.5, "MCHC": 33.8,
    "RDW": 12.8, "PLT": 364, "MPV": 9.6, "PCT": 0.35, "PDW": 10.6,
    "FERRITIN": 194.0, "FOLATE": 5.06, "B12": 178.2
}

# 2. Standard clinical reference ranges & units
ref_ranges = {
    "WBC": ("4.5 - 11.0", "10^9/L"),
    "NE#": ("1.8 - 7.0", "10^9/L"),
    "LY#": ("1.0 - 4.8", "10^9/L"),
    "MO#": ("0.1 - 0.8", "10^9/L"),     # 0.91 is High
    "EO#": ("0.0 - 0.4", "10^9/L"),     # 0.56 is High
    "BA#": ("0.0 - 0.1", "10^9/L"),
    "RBC": ("3.8 - 5.1", "10^12/L"),
    "HGB": ("11.7 - 15.5", "g/dL"),
    "HCT": ("35.0 - 45.0", "%"),
    "MCV": ("80.0 - 100.0", "fL"),
    "MCH": ("27.0 - 33.0", "pg"),
    "MCHC": ("32.0 - 36.0", "g/dL"),
    "RDW": ("11.5 - 14.5", "%"),
    "PLT": ("150 - 450", "10^9/L"),
    "MPV": ("9.0 - 12.0", "fL"),
    "PCT": ("0.16 - 0.36", "%"),
    "PDW": ("9.0 - 17.0", "%"),
    "FERRITIN": ("15 - 150", "ng/mL"),   # 194 is High
    "FOLATE": ("4.6 - 18.7", "ng/mL"),
    "B12": ("200 - 900", "pg/mL")        # 178.2 is Low (Anemia flag)
}

# 3. Setup canvas to resemble a standard 8.5" x 11" medical document
fig, ax = plt.subplots(figsize=(8.5, 11), dpi=300)
ax.axis('off')
fig.patch.set_facecolor('#ffffff')

# --- HEADER SECTION ---
plt.text(0.05, 0.95, "APEX DIAGNOSTIC LABORATORIES", fontsize=14, fontweight='bold', color='#1a365d')
plt.text(0.05, 0.92, "742 Medical Center Parkway, Suite 100 | Tel: (555) 019-9381", fontsize=8.5, color='#4a5568')
plt.text(0.05, 0.90, "Laboratory Director: Dr. Alan Grant, MD, FCAP", fontsize=8.5, style='italic', color='#4a5568')

# Header Line
ax.plot([0.05, 0.95], [0.88, 0.88], color='#1a365d', lw=2, transform=ax.transAxes)

# --- PATIENT INFO BOX ---
plt.text(0.05, 0.84, "Patient Name: DOE, JANE", fontsize=9.5, fontweight='bold')
plt.text(0.05, 0.81, "Patient ID:   NX-883412", fontsize=9.5)
plt.text(0.05, 0.78, "Gender/Age:   Female / 31", fontsize=9.5)

plt.text(0.55, 0.84, "Collection Date: 2026-05-10 07:45", fontsize=9.5)
plt.text(0.55, 0.81, "Report Date:     2026-05-10 12:30", fontsize=9.5)
plt.text(0.55, 0.78, "Ordering Phys:   Dr. E. Sattler", fontsize=9.5)

# Middle Line
ax.plot([0.05, 0.95], [0.75, 0.75], color='#cbd5e1', lw=1, transform=ax.transAxes)

# --- TABLE HEADERS ---
plt.text(0.05, 0.71, "TEST DESCRIPTION", fontsize=9.5, fontweight='bold', color='#1a365d')
plt.text(0.42, 0.71, "RESULT", fontsize=9.5, fontweight='bold', color='#1a365d')
plt.text(0.55, 0.71, "FLAG", fontsize=9.5, fontweight='bold', color='#1a365d')
plt.text(0.68, 0.71, "REF. RANGE", fontsize=9.5, fontweight='bold', color='#1a365d')
plt.text(0.85, 0.71, "UNITS", fontsize=9.5, fontweight='bold', color='#1a365d')

ax.plot([0.05, 0.95], [0.69, 0.69], color='#1a365d', lw=1, transform=ax.transAxes)

# --- DATA GENERATION LOOP ---
y_pos = 0.66
line_height = 0.021

test_groups = [
    ("COMPLETE BLOOD COUNT (CBC)", ["WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDW"]),
    ("WBC DIFFERENTIAL", ["NE#", "LY#", "MO#", "EO#", "BA#"]),
    ("PLATELET COUNT", ["PLT", "MPV", "PCT", "PDW"]),
    ("METABOLIC & VITAMIN PANEL (ANEMIA)", ["FERRITIN", "FOLATE", "B12"])
]

for group_name, keys in test_groups:
    # Print Section Header
    plt.text(0.05, y_pos, group_name, fontsize=8.5, fontweight='bold', color='#475569')
    y_pos -= line_height
    
    for key in keys:
        val = data[key]
        ref, unit = ref_ranges[key]
        
        # Calculate flags dynamically based on the ranges
        flag = ""
        if key == "B12" and val < 200: flag = "L"
        elif key == "MO#" and val > 0.8: flag = "H"
        elif key == "EO#" and val > 0.4: flag = "H"
        elif key == "FERRITIN" and val > 150: flag = "H"
        
        # Test Name
        plt.text(0.07, y_pos, key, fontsize=9, color='#1e293b')
        
        # Results & Flags formatting (OCR needs clear text contrast)
        if flag != "":
            plt.text(0.42, y_pos, f"{val}", fontsize=9, fontweight='bold', color='#0f172a')
            plt.text(0.55, y_pos, flag, fontsize=9, fontweight='bold', color='#b91c1c') # Red for flags
        else:
            plt.text(0.42, y_pos, f"{val}", fontsize=9, color='#334155')
            
        plt.text(0.68, y_pos, ref, fontsize=9, color='#475569')
        plt.text(0.85, y_pos, unit, fontsize=9, color='#475569')
        y_pos -= line_height
    y_pos -= 0.008  # Small spacing between groups

# --- FOOTER ---
ax.plot([0.05, 0.95], [0.08, 0.08], color='#cbd5e1', lw=1, transform=ax.transAxes)
plt.text(0.05, 0.05, "* Final Verified Electronic Document. 'H' indicates High, 'L' indicates Low values relative to standard reference intervals.", fontsize=7.5, color='#64748b', style='italic')

# Save as high-res PNG image optimal for OCR training data
plt.savefig('synthetic_blood_report.png', bbox_inches='tight', dpi=300)
plt.close()
print("Synthetic laboratory report generated successfully as 'synthetic_blood_report.png'!")
