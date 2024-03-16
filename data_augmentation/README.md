1. Make an input folder and open it. Put `attributes` in attributes.txt. If an `attribute` has `sub-attributes`, type the name of the `sub-attributes` after a space. If not, type "not" after a space.
2. For each `attribute`, create a folder named after the attribute and, within that folder, create a txt file also named after the attribute. Add instances of the attribute in the txt file.
3. If an `attribute` has `sub-attributes`, then for each `attribute instance` in the txt file named after the attribute, create a folder named after the `attribute instance` within the folder named after the attribute. In each new folder, place txt files named after the `sub-attributes`, containing instances corresponding to the `sub-attributes` of the `attribute`.
4. put examples(jsonl) in example folder.
5. python DA.py --input_folder "your input folder" --num_of_output "Number of items generated" --output_file "name of output file"