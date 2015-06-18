"""
(c) 2015
author: Ghazaleh Esmaili
writes results in csv file
"""
import csv

def write_results_to_csv_file(results,output_file_name):
    file_open = open(output_file_name, 'w', newline='')
    file_csv = csv.writer(file_open)
        
    for row in results:
        file_csv.writerow(row)       
    file_open.close()
    return



