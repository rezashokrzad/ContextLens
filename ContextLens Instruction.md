# Using ContextLens:
1. **Accessing the tool**:
    + a. Open a web browser and navigate to https://contextlens.cls.ru.nl.
    + b. You should see the ContextLens dashboard.
2. **Uploading data**: 
    + a. Click on the "Choose File" button to select a ".csv" or ".xlsx" file to upload. 
    + b. The file should contain a one to three column(s) of up to 2000 sentences.
    + c. The file should be header free
    + d. Three possible versions of the file:
        - one single column (includes only the texts)
         ![image](https://user-images.githubusercontent.com/72080909/232730312-4456ef99-ccfe-4510-a6a5-a7028e55f266.png)

        - two columns (include texts and user label 1)
        ![image](https://user-images.githubusercontent.com/72080909/232730645-09ec894b-7cf8-4ca1-830e-4a712bcadce2.png)

        - three columns (include texts, user lable 1 and user label 2) 
        ![image](https://user-images.githubusercontent.com/72080909/232730843-1be6f345-45e6-4346-b729-7077515915f5.png)
    + e. The labels are available in both menus of shape and color.
3. **Specifying your desired word**:
    + a. In the "Desired word" text box, enter the word you want to explore.
    + b. You can also enter multiple distinct words by separating them with commas.
4. **Specifying the number of clusters**:
    + a. In the "Number of clusters" text box, in case you have a single word, enter the number of clusters you want to generate. 
    + b. Note that:
      + b.1. enter an integer (int) not a string as the number of clusters.
      + b.2. if you are entering list of words in the other text box, you do not need to specify the number of clusters as that is the number of words.
5. **Generating embeddings and clusters**:
    + a. Click on the "Process" button to generate the BERT-base embeddings, PCA and UMAP projections, and combined clustering labels. 
    + b. you can find the result, called df_user, in the leftmost drop-down menu, data frame. 
6. **Exploring the visualizations**: 
    + a. Use the two other drop-down menus on the left side of the dashboard to explore different options for visualizing the embeddings and clusters. For the latter note that if you are entering a single word, one of the voting result shuld be selected, in the other case word-label is also the option.
    + b. Use the three drop-down menus on the right side of the dashboard to select your preferred dimensions for the PCA and UMAP projections.
    + c. Hover over data points to see further detail, including the sentence the word is in, the index of the sentence in the data frame, and the label of the word.
7. **Exporting data**:
   + a. Click on the hyperlink below the data frame menu to download a file containing the columns: sentences, dimensions after reduction procedures, and annotated labels.
   + b. in case you do need to have word embeddings click on "without embedding" version. 

> Note: ContextLens is designed to be user-friendly and interactive, allowing linguists to explore and interpret word embeddings quickly and easily. If you have any questions or encounter any issues while using the tool, refer to the additional resources on the GitHub repository or contact the developer for assistance.

![image](https://user-images.githubusercontent.com/72080909/232738606-66c0e6a4-f81e-441a-a3c4-3197a956f2fe.png)
