import csv

# Open the CSV file
with open('colour_data.txt', 'r') as input_file:
    # Read the CSV file
    reader = csv.reader(input_file)
    # Remove the third column
    new_rows = [[row[1], row[2], row[4]] for row in reader]

# Open a new CSV file
with open('output.csv', 'w', newline='') as output_file:
    # Write the new rows to the output file
    writer = csv.writer(output_file)
    writer.writerows(new_rows)
