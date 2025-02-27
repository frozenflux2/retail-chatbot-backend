import os
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import json

# Load environment variables from .env file (for API keys and DB credentials)
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the structured output format using Pydantic
class AtHearthResponse(BaseModel):
    response: str
    generate_sql: bool
    sql_query: Optional[str] = None

def execute_sql_query(query):
    """Execute SQL query on PostgreSQL database and return results"""
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "athearth"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            port=os.getenv("DB_PORT", "5432")
        )
        
        # Create a cursor that returns results as dictionaries
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Convert results to a list of dictionaries
            results_list = [dict(row) for row in results]
            
        conn.close()
        return results_list, None
    
    except Exception as e:
        return None, f"Database error: {str(e)}"

def chat_with_gpt4o(prompt, conversation_history=None):
    if conversation_history is None:
        conversation_history = []
    
    # Add the user's message to the conversation history
    conversation_history.append({"role": "user", "content": prompt})
    
    # Create a system message specific to AtHearth rental platform
    system_message = """You are AtHearth's AI rental assistant for Tokyo properties. 
    Help users find suitable apartments in Tokyo based on their preferences.
    
    Key information about AtHearth:
    - Tokyo's rental platform since 2015
    - Specializes in foreigner-friendly properties
    - Offers properties across Tokyo's wards including Minato-ku, Shibuya-ku, Shinjuku-ku, etc.
    - Provides support for the entire rental process from search to move-in
    - Only about 4% of Tokyo listings are actually available for foreigners
    
    Popular Tokyo wards include:
    - Minato-ku: Most English-friendly facilities, many international schools, embassies
    - Shibuya-ku: Trendy area with shopping and nightlife
    - Shinjuku-ku: Major transportation hub with many properties
    
    Typical rental process timeline:
    - 2-3 months before: Start planning
    - 1-1.5 months before: Search properties
    - 3 weeks before: Property viewings
    - 1-2 weeks before: Complete payment & sign lease
    - Day before: Key pickup
    
    Database schema for properties:
    ```
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
    ```
    
    IMPORTANT: Only the following fields are queryable and must be used for filtering in SQL queries:
    
    1. street (TEXT):
       - Contains street name in English
       - Use LOWER() and LIKE with wildcards for partial matching
       - Example: WHERE LOWER(street) LIKE LOWER('%roppongi%')
    
    2. ward (TEXT):
       - Contains Tokyo ward name (e.g., "Minato-ku", "Shibuya-ku")
       - Common wards: Minato-ku, Shibuya-ku, Shinjuku-ku, Setagaya-ku, Meguro-ku
       - Use LOWER() and LIKE with wildcards for partial matching
       - Example: WHERE LOWER(ward) LIKE LOWER('%minato%')
    
    3. city (TEXT):
       - Contains city name, usually "Tokyo"
       - Use LOWER() and LIKE with wildcards for partial matching
       - Example: WHERE LOWER(city) LIKE LOWER('%tokyo%')
    
    4. bed_rooms (TEXT):
       - Contains number of bedrooms as text (e.g., "1", "2", "3")
       - For exact matching: WHERE bed_rooms = '2'
       - For range queries: WHERE CAST(bed_rooms AS INTEGER) >= 2
    
    5. floors (TEXT):
       - Contains floor number as text (e.g., "1", "2", "10")
       - For exact matching: WHERE floors = '3'
       - For range queries: WHERE CAST(floors AS INTEGER) >= 5
    
    6. age (TEXT):
       - Contains building age in years as text (e.g., "5", "10", "20")
       - For exact matching: WHERE age = '5'
       - For range queries: WHERE CAST(age AS INTEGER) <= 10
    
    7. rent_amount (TEXT):
       - Contains monthly rent amount as text (e.g., "100000", "150000")
       - For exact matching: WHERE rent_amount = '100000'
       - For range queries: WHERE CAST(rent_amount AS INTEGER) BETWEEN 100000 AND 200000
    
    Be helpful, informative, and guide users through the Tokyo rental process.
    
    If the user asks for data that would require a database query:
    1. Set generate_sql to true
    2. Provide an appropriate SQL query using ONLY the queryable fields listed above for filtering (WHERE clauses)
    3. You can SELECT other fields from the schema to display information, but only filter on the queryable fields
    4. Make sure all SQL syntax is valid
    5. Always use CAST() for numeric comparisons since fields are stored as TEXT
    6. Limit results to 10 by default to avoid overwhelming responses
    
    Otherwise, set generate_sql to false and leave sql_query empty."""
    
    try:
        # Make API call to OpenAI with structured output format
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",  # Using the GPT-4o model
            messages=[
                {"role": "system", "content": system_message},
                *conversation_history
            ],
            temperature=0.7,
            max_tokens=1000,
            response_format=AtHearthResponse,
        )
        
        # Extract the structured response
        structured_response = completion.choices[0].message.parsed
        
        # If SQL query is generated, execute it and get results
        if structured_response.generate_sql and structured_response.sql_query:
            results, error = execute_sql_query(structured_response.sql_query)
            
            if error:
                # If there's an error, update the response
                structured_response.response += f"\n\nI tried to query our database, but encountered an error: {error}"
            elif results is not None:
                # If we have results, generate a new response based on them
                result_prompt = f"""
                I executed the following SQL query:
                {structured_response.sql_query}
                
                And got these results:
                {json.dumps(results, default=str)}
                
                Based on these actual database results, please provide a comprehensive and accurate response to the user's query:
                "{prompt}"
                
                Include specific details from the results where relevant.
                """
                
                # Get a new response based on the query results
                result_completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are AtHearth's AI rental assistant. Provide accurate information based on the database results."},
                        {"role": "user", "content": result_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                # Update the response with the result-based answer
                structured_response.response = result_completion.choices[0].message.content
        
        # Add the assistant's response to the conversation history (as string for history)
        response_for_history = {
            "role": "assistant", 
            "content": f"Response: {structured_response.response}" + 
                      (f"\nSQL Query: {structured_response.sql_query}" if structured_response.generate_sql else "")
        }
        conversation_history.append(response_for_history)
        
        return structured_response, conversation_history
    
    except Exception as e:
        error_response = AtHearthResponse(
            response=f"An error occurred: {str(e)}",
            generate_sql=False
        )
        return error_response, conversation_history

def main():
    print("Welcome to AtHearth's Tokyo Rental Assistant!")
    print("I can help you find the perfect apartment in Tokyo and guide you through the rental process.")
    print("Type 'exit' to end the conversation.")
    
    conversation_history = []
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("AtHearth Assistant: Thank you for using AtHearth's rental service. We hope to help you find your perfect Tokyo home soon!")
            break
        
        response_data, conversation_history = chat_with_gpt4o(user_input, conversation_history)
        
        # Display the text response
        print(f"AtHearth Assistant: {response_data.response}")
        
        # If SQL query was generated, display it
        if response_data.generate_sql and response_data.sql_query:
            print("\nGenerated SQL Query:")
            print(response_data.sql_query)

if __name__ == "__main__":
    main()
