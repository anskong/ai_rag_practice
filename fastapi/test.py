import os
from dotenv import load_dotenv

load_dotenv()

test_key = os.getenv("TEST_KEY")

print(test_key) 
