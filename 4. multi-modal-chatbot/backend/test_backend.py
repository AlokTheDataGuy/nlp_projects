import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the necessary modules
from models.text_processor import process_text
import asyncio

async def test_text_processor():
    """Test the text processor with a simple message."""
    print("Testing text processor...")
    response = await process_text("Hello, how are you today?")
    print(f"Response: {response}")
    return response

if __name__ == "__main__":
    print("Running backend test...")
    
    # Run the test
    asyncio.run(test_text_processor())
    
    print("Test completed successfully!")
