    Instructions for the preparation of the training data, and the training and evaluation of the models of the main approach
    
    1. Obtain the clusterings via cluster_plotting.py 
        ◦ copy the obtained cluster csv file into the preprocessing_clustering folder
    2. create the embedding csv files in the directory generate_embeddings/UNI/notebooks/ via the dedicated python script depending on the path type you want
        ◦ a folder containing the tif images must be manually added
        ◦ copy the desired embedding into the preprocessing_clustering/embeddings folder 
    3. prepare the training data:
        ◦ for the patch-centered approach:
            ▪ prepare_training_data_centered.py with centered embeddings
        ◦ for the grid-based approach:
            ▪ prepare_training_data_with conf.py with standard embeddings
        ◦ configurations can be made in the first few lines of the python scripts
        1. copy the training data into the main/training_data folder
    4. train the models
        1. run the training_combined_with_svm.ipynb
            ▪ configurations can be made after the markdown cells Part 1 and Part 2
    5. test the models against the acinar tissue slice when trained on adenocarcinoma
        1. prepare the training data for the acinar tissue slice similar as for the adenocarcinoma slice
        2. run the dnn_all_data_training.ipynb notebook