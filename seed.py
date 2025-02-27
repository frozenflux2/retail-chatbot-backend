import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import re
from openai import OpenAI
import time
import numpy as np
import concurrent.futures
from tqdm import tqdm

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

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Number of parallel workers for embedding generation
MAX_WORKERS = 5

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

def create_pgvector_extension(conn):
    """Create the pgvector extension if it doesn't exist."""
    cursor = conn.cursor()
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        print("pgvector extension created or already exists")
    except Exception as e:
        print(f"Error creating pgvector extension: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()

def create_properties_table(conn):
    """Create the properties table in PostgreSQL if it doesn't exist."""
    cursor = conn.cursor()
    
    # Drop the table if it exists to recreate with proper types
    cursor.execute("DROP TABLE IF EXISTS properties;")
    
    # Create a table with all columns as TEXT to avoid type issues
    # Add description_embedding column for vector search
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
        updated_at TIMESTAMP,
        description_embedding VECTOR(1536)
    );
    """)
    
    conn.commit()
    print("Properties table created successfully")
    cursor.close()

def fetch_data_from_csv():
    """Fetch data from CSV file."""
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Successfully loaded {len(df)} rows from CSV file")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise

def clean_data(df):
    """Clean and prepare data for database insertion."""
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()
    
    # Convert all numeric columns to string to avoid type issues
    for col in df_clean.columns:
        if col not in ['is_popular', 'is_most_popular', 'created_at', 'updated_at', 'build_date']:
            df_clean[col] = df_clean[col].astype(str)
            # Replace 'nan' strings with None
            df_clean[col] = df_clean[col].apply(lambda x: None if x == 'nan' else x)
    
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
    
    print("Data cleaned and prepared for database insertion")
    return df_clean

def get_embedding(text):
    """Get embedding vector for text using OpenAI's embedding model."""
    if not text or pd.isna(text) or text == 'None':
        return None
    
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Add exponential backoff retry logic for rate limits
        if "rate limit" in str(e).lower():
            retry_time = 2
            print(f"Rate limit hit. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return get_embedding(text)  # Retry once
        return None

def insert_data_to_postgres(conn, df):
    """Insert data into the PostgreSQL database."""
    cursor = conn.cursor()
    
    # Get only the columns that exist in the table
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'properties'")
    table_columns = [row[0] for row in cursor.fetchall()]
    
    # Filter dataframe to only include columns that exist in the table
    df_filtered = df[[col for col in df.columns if col.lower() in [c.lower() for c in table_columns]]]
    
    # Get column names from filtered dataframe
    columns = df_filtered.columns.tolist()
    column_str = ', '.join([f'"{col}"' for col in columns])  # Quote column names
    
    # Insert data row by row to identify problematic rows
    successful_inserts = 0
    for i, row in df_filtered.iterrows():
        try:
            # Convert row to list of values
            values = [None if pd.isna(val) or val == 'None' or val == 'nan' else val for val in row]
            
            # Create placeholders for the SQL query
            placeholders = ', '.join(['%s'] * len(values))
            
            # Build the query
            query = f'INSERT INTO properties ({column_str}) VALUES ({placeholders})'
            
            # Execute the query
            cursor.execute(query, values)
            conn.commit()  # Commit after each successful insert
            
            successful_inserts += 1
            
            # Print progress
            if successful_inserts % 10 == 0:
                print(f"Inserted {successful_inserts} rows so far")
                
        except Exception as e:
            print(f"Error inserting row {i}: {e}")
            print(f"Problematic row data: {row.to_dict()}")
            conn.rollback()  # Rollback the transaction for this row
            # Continue with the next row
            continue
    
    print(f"Successfully inserted {successful_inserts} out of {len(df)} rows into the database")

def process_embedding_batch(property_batch):
    """Process a batch of properties to generate embeddings."""
    results = []
    
    for prop_id, description in property_batch:
        try:
            # Get embedding from OpenAI
            embedding = get_embedding(description)
            
            if embedding:
                results.append((prop_id, embedding))
            else:
                print(f"Failed to generate embedding for property {prop_id}")
        except Exception as e:
            print(f"Error processing property {prop_id}: {e}")
    
    return results

def update_embeddings_in_db(conn, embedding_results):
    """Update the database with generated embeddings."""
    cursor = conn.cursor()
    updated_count = 0
    
    for prop_id, embedding in embedding_results:
        try:
            cursor.execute(
                "UPDATE properties SET description_embedding = %s WHERE id = %s",
                (embedding, prop_id)
            )
            conn.commit()
            updated_count += 1
        except Exception as e:
            print(f"Error updating embedding for property {prop_id}: {e}")
            conn.rollback()
    
    return updated_count

def generate_and_store_embeddings(conn):
    """Generate embeddings for property descriptions and store them in the database using parallel processing."""
    cursor = conn.cursor()
    
    try:
        # Get all properties with descriptions but no embeddings
        cursor.execute("""
            SELECT id, description 
            FROM properties 
            WHERE description IS NOT NULL 
            AND description != '' 
            AND description_embedding IS NULL
        """)
        properties = cursor.fetchall()
        
        if not properties:
            print("No properties found that need embeddings")
            return
        
        print(f"Generating embeddings for {len(properties)} properties using parallel processing...")
        
        # Process properties in batches to avoid overwhelming the system
        batch_size = 20
        total_processed = 0
        
        # Create batches
        property_batches = [properties[i:i+batch_size] for i in range(0, len(properties), batch_size)]
        
        # Process batches with progress bar
        with tqdm(total=len(properties), desc="Generating embeddings") as pbar:
            for batch in property_batches:
                # Process this batch in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    # Split the batch into smaller chunks for each worker
                    worker_batch_size = max(1, len(batch) // MAX_WORKERS)
                    worker_batches = [batch[i:i+worker_batch_size] for i in range(0, len(batch), worker_batch_size)]
                    
                    # Submit tasks to the executor
                    future_to_batch = {executor.submit(process_embedding_batch, worker_batch): worker_batch 
                                      for worker_batch in worker_batches}
                    
                    # Process results as they complete
                    all_results = []
                    for future in concurrent.futures.as_completed(future_to_batch):
                        results = future.result()
                        all_results.extend(results)
                        
                # Update database with all results from this batch
                updated = update_embeddings_in_db(conn, all_results)
                total_processed += updated
                pbar.update(updated)
                
                # Brief pause between batches to avoid rate limits
                time.sleep(0.5)
        
        print(f"Successfully generated and stored embeddings for {total_processed} properties")
    
    except Exception as e:
        print(f"Error in generate_and_store_embeddings: {e}")
        conn.rollback()
    finally:
        cursor.close()

def main():
    """Main function to orchestrate the data transfer process."""
    try:
        # Create database if it doesn't exist
        create_database_if_not_exists()
        
        # Connect to PostgreSQL
        conn = connect_to_postgres()
        
        # Create pgvector extension
        create_pgvector_extension(conn)
        
        # Create table if it doesn't exist
        create_properties_table(conn)
        
        # Fetch data from CSV
        df = fetch_data_from_csv()
        
        # Clean data
        df_clean = clean_data(df)
        
        # Insert data
        insert_data_to_postgres(conn, df_clean)
        
        # Generate and store embeddings
        generate_and_store_embeddings(conn)
        
        # Close connection
        conn.close()
        
        print("Data transfer and embedding generation completed successfully")
    except Exception as e:
        print(f"Error in data transfer process: {e}")

if __name__ == "__main__":
    main()
