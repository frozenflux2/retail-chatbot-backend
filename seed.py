import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# CSV file path
CSV_FILE_PATH = "dummy.csv"

# PostgreSQL connection parameters (from environment variables)
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

def create_database_if_not_exists():
    """Create the database if it doesn't exist."""
    try:
        # Connect to PostgreSQL server without specifying a database
        conn = psycopg2.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            dbname="postgres"  # Connect to default postgres database
        )
        conn.autocommit = True  # Required for CREATE DATABASE
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{DB_NAME}'")
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database '{DB_NAME}' created successfully")
        else:
            print(f"Database '{DB_NAME}' already exists")
            
        # Close connection to postgres database
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating database: {e}")
        raise

def connect_to_postgres():
    """Connect to PostgreSQL database and return the connection."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        print(f"Successfully connected to PostgreSQL database: {DB_NAME}")
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        raise

def create_properties_table(conn):
    """Create the properties table in PostgreSQL if it doesn't exist."""
    cursor = conn.cursor()
    
    # Drop the table if it exists to recreate with proper types
    cursor.execute("DROP TABLE IF EXISTS properties;")
    
    # Create a table with all columns as TEXT to avoid type issues
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS properties (
        id TEXT PRIMARY KEY,
        company_id TEXT,
        status TEXT,
        slug TEXT,
        name TEXT,
        name_ja TEXT,
        address TEXT,
        address_ja TEXT,
        street TEXT,
        ward TEXT,
        city TEXT,
        description TEXT,
        type TEXT,
        layout TEXT,
        popularity TEXT,
        is_popular BOOLEAN,
        is_most_popular BOOLEAN,
        bed_rooms TEXT,
        room_number TEXT,
        floors TEXT,
        security_deposit TEXT,
        advertising_fee TEXT,
        size_sqm TEXT,
        build_date DATE,
        build_date_ja TEXT,
        age TEXT,
        age_ja TEXT,
        move_in_condition TEXT,
        move_in_condition_ja TEXT,
        featured_image TEXT,
        rent_currency TEXT,
        rent_amount TEXT,
        management_fee TEXT,
        deposit_fee TEXT,
        latitude TEXT,
        longitude TEXT,
        facing_side TEXT,
        facing_side_ja TEXT,
        source_link TEXT,
        created_at TIMESTAMP,
        updated_at TIMESTAMP
    );
    """)
    
    conn.commit()
    print("Properties table created successfully")

def fetch_data_from_csv():
    """Fetch data from the CSV file."""
    try:
        # Read the CSV file with all columns as strings initially
        df = pd.read_csv(CSV_FILE_PATH, dtype=str)
        
        print(f"Successfully loaded {len(df)} rows of data from CSV file")
        return df
    except Exception as e:
        print(f"Error loading data from CSV file: {e}")
        raise

def clean_data(df):
    """Clean and prepare the data for database insertion."""
    # Select only the columns we need for our table
    essential_columns = [
        'id', 'company_id', 'status', 'slug', 'name', 'name_ja', 
        'address', 'address_ja', 'description', 'type', 'layout',
        'popularity', 'is_popular', 'is_most_popular', 'bed_rooms',
        'room_number', 'floors', 'security_deposit', 'advertising_fee', 'size_sqm',
        'build_date', 'build_date_ja', 'age', 'age_ja', 'move_in_condition',
        'move_in_condition_ja', 'featured_image', 'rent_currency', 'rent_amount',
        'management_fee', 'deposit_fee', 'latitude', 'longitude',
        'facing_side', 'facing_side_ja', 'source_link', 'created_at', 'updated_at'
    ]
    
    # Filter columns that exist in the dataframe
    available_columns = [col for col in essential_columns if col in df.columns]
    df_clean = df[available_columns].copy()
    
    # Clean up floors field to extract only the number
    if 'floors' in df_clean.columns:
        # Extract only the numeric part from floors field
        floor_pattern = r'(\d+)'
        for idx, floors in df_clean['floors'].items():
            if pd.isna(floors) or floors is None:
                continue
                
            match = re.search(floor_pattern, str(floors))
            if match:
                df_clean.at[idx, 'floors'] = match.group(1)  # Extract just the number
    
    # Split address into components
    if 'address' in df_clean.columns:
        # Create new columns for address components
        df_clean['street'] = None
        df_clean['ward'] = None
        df_clean['city'] = None
        
        # Process each address
        for idx, address in df_clean['address'].items():
            if pd.isna(address) or address is None:
                continue
                
            # Split by commas and clean up whitespace
            parts = [p.strip() for p in address.split(',')]
            
            # Check if address is in reverse order (Tokyo, Shibuya, Higashi 4-7-4)
            is_reverse_order = False
            if len(parts) >= 3:
                # Check if the first part is likely a city (Tokyo, Osaka, etc.)
                common_cities = ['Tokyo', 'Osaka', 'Yokohama', 'Nagoya', 'Sapporo', 'Fukuoka', 'Kyoto']
                if any(city.lower() in parts[0].lower() for city in common_cities):
                    is_reverse_order = True
            
            # Handle different address formats
            if is_reverse_order and len(parts) >= 3:
                # Format: "Tokyo, Shibuya, Higashi 4-7-4"
                df_clean.at[idx, 'city'] = parts[0]
                df_clean.at[idx, 'ward'] = parts[1]
                # If there are more parts, combine them as the street
                if len(parts) > 3:
                    df_clean.at[idx, 'street'] = ', '.join(parts[2:])
                else:
                    df_clean.at[idx, 'street'] = parts[2]
            elif len(parts) >= 4:  # Format: "7-14, Shirogane 1-chome, Minato-ku, Tokyo"
                # Combine the first two parts as the street address
                df_clean.at[idx, 'street'] = f"{parts[0]}, {parts[1]}"
                df_clean.at[idx, 'ward'] = parts[2]
                df_clean.at[idx, 'city'] = parts[3]
            elif len(parts) == 3:  # Format: "8-12-1 Machiya, Arakawa-ku, Tokyo"
                df_clean.at[idx, 'street'] = parts[0]
                df_clean.at[idx, 'ward'] = parts[1]
                df_clean.at[idx, 'city'] = parts[2]
            elif len(parts) == 2:  # Missing city
                df_clean.at[idx, 'street'] = parts[0]
                df_clean.at[idx, 'ward'] = parts[1]
            elif len(parts) == 1:  # Only street
                df_clean.at[idx, 'street'] = parts[0]
    
    # Convert text fields to lowercase and remove special characters
    text_fields = ['street', 'ward', 'city', 'name', 'address', 'slug']
    for field in text_fields:
        if field in df_clean.columns:
            # Convert to lowercase and remove special characters
            df_clean[field] = df_clean[field].apply(
                lambda x: re.sub(r'[^\w\s]', '', str(x).lower()) if not pd.isna(x) and x is not None else x
            )
    
    # Keep all potentially problematic columns as text
    # No conversion to numeric types for these columns
    
    # Convert boolean columns
    boolean_columns = ['is_popular', 'is_most_popular']
    for col in boolean_columns:
        if col in df_clean.columns:
            # Handle various boolean representations
            df_clean[col] = df_clean[col].apply(
                lambda x: True if str(x).lower() in ('true', '1', 'yes', 'y', '100') 
                else (False if str(x).lower() in ('false', '0', 'no', 'n') else None)
            )
    
    # Convert date columns
    date_columns = ['build_date', 'created_at', 'updated_at']
    for col in date_columns:
        if col in df_clean.columns:
            # Convert to datetime and handle NaT (Not a Time)
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            # Replace NaT with None
            df_clean[col] = df_clean[col].apply(lambda x: None if pd.isna(x) else x)
    
    # Replace 'nan' strings with None
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(lambda x: None if pd.isna(x) or str(x).lower() == 'nan' else x)
    
    print("Data cleaned and prepared for database insertion")
    return df_clean

def insert_data_to_postgres(conn, df):
    """Insert data into the PostgreSQL database."""
    cursor = conn.cursor()
    
    # Get column names
    columns = df.columns.tolist()
    column_str = ', '.join([f'"{col}"' for col in columns])  # Quote column names
    
    # Insert data row by row to identify problematic rows
    successful_inserts = 0
    for i, row in df.iterrows():
        try:
            # Convert row to list of values
            values = [None if pd.isna(val) else val for val in row]
            
            # Create placeholders for the SQL query
            placeholders = ', '.join(['%s'] * len(values))
            
            # Build the query
            query = f'INSERT INTO properties ({column_str}) VALUES ({placeholders})'
            
            # Execute the query
            cursor.execute(query, values)
            successful_inserts += 1
            
            # Commit after each row
            conn.commit()
            
            # Print progress
            if successful_inserts % 10 == 0:
                print(f"Inserted {successful_inserts} rows so far")
                
        except Exception as e:
            print(f"Error inserting row {i}: {e}")
            print(f"Problematic row data: {row.to_dict()}")
            # Continue with the next row
            continue
    
    print(f"Successfully inserted {successful_inserts} out of {len(df)} rows into the database")

def main():
    """Main function to orchestrate the data transfer process."""
    try:
        # Create database if it doesn't exist
        create_database_if_not_exists()
        
        # Fetch data from CSV
        df = fetch_data_from_csv()
        
        # Clean data
        df_clean = clean_data(df)
        
        # Connect to PostgreSQL
        conn = connect_to_postgres()
        
        # Create table if it doesn't exist
        create_properties_table(conn)
        
        # Insert data
        insert_data_to_postgres(conn, df_clean)
        
        # Close connection
        conn.close()
        
        print("Data transfer completed successfully")
    except Exception as e:
        print(f"Error in data transfer process: {e}")

if __name__ == "__main__":
    main()
