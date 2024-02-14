def read_file_into_string(file_path):
    try:
        with open(file_path, 'r') as file:
            # Read the entire content of the file into a single string
            file_contents = file.read()
            return file_contents
    except FileNotFoundError:
        return "File not found. Please provide a valid file path."

# Example usage:
file_path = 'C:\\Users\\Philip Caldarella\\Desktop\\CSC494\\test.txt'
file_text = read_file_into_string(file_path)
print("Contents of the file as a single string:")
print(file_text) 
