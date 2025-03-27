import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY not found in environment variables")
else:
    genai.configure(api_key=api_key)

def generate_retention_strategies(employee_data: dict) -> list:
    """
    Generate personalized retention strategies based on employee data using Gemini AI.
    
    Args:
        employee_data (dict): Dictionary containing employee attributes
        
    Returns:
        list: List of retention strategy recommendations
    """
    # Convert input data to a more readable format
    formatted_data = "\n".join([f"{k}: {v}" for k, v in employee_data.items()])
    
    prompt = f"""
    Based on the following employee details, suggest the best retention strategies to reduce attrition.
    Employee Data:
    {formatted_data}
    
    Format your response with these requirements:
    1. Create categories of strategies (like Work-Life Balance, Job Satisfaction, etc.)
    2. Format each category heading in bold and end it with a semicolon
    3. Under each category, list specific actionable strategies as clean bullet points
    4. Do not use stars or any special symbols in the bullet points
    5. Keep recommendations concise and directly actionable
    
    Consider factors such as salary, work-life balance, job satisfaction, performance, and career growth.
    """
    
    try:
        if not api_key:
            return ["API key not configured. Unable to generate retention strategies."]
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        # Process the response to ensure it matches the desired format
        raw_strategies = response.text.split("\n")
        formatted_strategies = []
        
        current_line = ""
        for line in raw_strategies:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a heading line (contains a semicolon)
            if ":" in line:
                # Format as bold with semicolon
                heading_parts = line.split(":")
                formatted_heading = f"**{heading_parts[0].strip()}:**"
                formatted_strategies.append(formatted_heading)
            else:
                # Clean the bullet point line
                if line.startswith("•") or line.startswith("-") or line.startswith("*"):
                    line = line[1:].strip()
                formatted_strategies.append(f"* {line}")
        print(formatted_strategies)
        return formatted_strategies
    except Exception as e:
        return [f"Error generating retention strategies: {str(e)}"]