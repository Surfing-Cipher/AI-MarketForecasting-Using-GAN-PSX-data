from models_engine import GANGenerator
import traceback

print("1. Initializing GANGenerator...")
try:
    gan = GANGenerator()
    print("GANGenerator Initialized.")
except Exception as e:
    print(f"Failed to initialize GANGenerator: {e}")
    traceback.print_exc()
    exit()

print("2. Generating Data...")
try:
    data = gan.generate_synthetic_data()
    print(f"Data Generated. Type: {type(data)}, Length: {len(data)}")
    print(f"First 5 values: {data[:5]}")
except Exception as e:
    print(f"Failed to generate data: {e}")
    traceback.print_exc()
