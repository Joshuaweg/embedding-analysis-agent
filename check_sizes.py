import os

def get_file_sizes():
    """Print sizes of all files in current directory"""
    files = []
    for file in os.listdir('.'):
        if os.path.isfile(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            files.append((file, size_mb))
    
    # Sort by size
    files.sort(key=lambda x: x[1], reverse=True)
    
    print("\nFile sizes:")
    print("-" * 50)
    for file, size in files:
        print(f"{file}: {size:.2f} MB")
        
    # Print files to ignore
    print("\nFiles to add to .gitignore (>100MB):")
    print("-" * 50)
    for file, size in files:
        if size > 100:
            print(file)

if __name__ == "__main__":
    get_file_sizes() 