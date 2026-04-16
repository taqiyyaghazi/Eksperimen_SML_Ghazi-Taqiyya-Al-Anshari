import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_student_data(df_input):
    """
    Fungsi untuk melakukan seluruh tahapan preprocessing pada dataset student dropout.
    """
    df = df_input.copy()

    # 1. Handling Missing Values
    num_cols = ['Family_Income', 'Study_Hours_per_Day', 'Stress_Index']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols_miss = ['Parental_Education']
    for col in cat_cols_miss:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 2. Handling Outliers (Clipping Family_Income)
    q1 = df['Family_Income'].quantile(0.25)
    q3 = df['Family_Income'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df['Family_Income'] = df['Family_Income'].clip(lower=lower_bound, upper=upper_bound)

    # 3. Encoding Categorical Data
    le = LabelEncoder()
    education_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    semester_map = {'Year 1': 1, 'Year 2': 2, 'Year 3': 3, 'Year 4': 4}

    if df['Parental_Education'].dtype == 'object':
        df['Parental_Education'] = df['Parental_Education'].map(education_map)
    
    if df['Semester'].dtype == 'object':
        df['Semester'] = df['Semester'].map(semester_map)

    cat_features_nominal = ['Gender', 'Internet_Access', 'Part_Time_Job', 'Scholarship', 'Department']
    for col in cat_features_nominal:
        df[col] = le.fit_transform(df[col].astype(str))

    # 4. Binning
    if df['GPA'].max() <= 4.0 and df['GPA'].min() >= 0.0:
        df['GPA_Category_Encoded'] = le.fit_transform(pd.cut(df['GPA'], bins=[0, 2.0, 3.0, 4.0], labels=['Low', 'Mid', 'High'], include_lowest=True))
    
    if df['Age'].max() > 10:
        df['Age_Group_Encoded'] = le.fit_transform(pd.cut(df['Age'], bins=[0, 20, 25, 100], labels=['Young', 'Mature', 'Senior']).astype(str))
    
    if df['Attendance_Rate'].max() > 1:
        df['Attendance_Encoded'] = le.fit_transform(pd.cut(df['Attendance_Rate'], bins=[0, 75, 100], labels=['Low', 'High'], include_lowest=True).astype(str))

    df['Income_Level_Encoded'] = le.fit_transform(pd.qcut(df['Family_Income'], q=3, duplicates='drop').astype(str))

    # 5. Feature Scaling
    scaler = StandardScaler()
    features_to_scale = ['Age', 'Family_Income', 'Study_Hours_per_Day', 'Attendance_Rate', 'Travel_Time_Minutes', 'Stress_Index', 'GPA', 'Semester_GPA', 'CGPA']
    
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # 6. Remove Unused Columns
    columns_to_exclude = ['Student_ID', 'GPA_Category', 'Age_Group', 'Income_Level', 'Attendance_Category']
    ready_features = [col for col in df.columns if col not in columns_to_exclude]
    
    return df[ready_features]

if __name__ == "__main__":
    import os
    
    # File paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "..", "student_dropout_raw.csv")
    output_file = os.path.join(current_dir, "student_dropout_preprocessing.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
    else:
        print(f"Loading data from {input_file}...")
        df_raw = pd.read_csv(input_file)
        
        print("Preprocessing data...")
        df_processed = preprocess_student_data(df_raw)
        
        print(f"Saving preprocessed data to {output_file}...")
        df_processed.to_csv(output_file, index=False)
        print("Done!")
