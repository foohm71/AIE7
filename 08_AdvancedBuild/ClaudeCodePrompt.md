# Claude Code Prompt used to generate the baseline and experiment to evaluate how a semantic chunking approach performs compared to a baseline Naive RAG 

Please do the following:
1. Review the Python Notebook `Evaluating_RAG_with_Ragas_(2025)_AI_Makerspace.ipynb` that is in the workspace
2. Create the Python Notebook `Evaluating_Semantic_based_Chunking_with_Ragas.ipynb` with the following:
   - We want the same baseline Ragas evaluation of the Naive RAG ie. cells up to the cell with title "Making Adjustments and Re-evaluating"
   - Remove all markdown text that are related to the lesson and keep only key points to help the reader navigate the notebook
   - Remove all "Question" and "Answer" markdown cells
3. After the baseline RAG evaluation in the new notebook, I would like to have a new section called "Evaluating the TOCChunker" 
   - create a new QDrant vector store like what was done in the baseline
   - in this section, I would like the same code as the baseline but instead of using the RecursiveCharacterTextSplitter to chunk the documents into the vector I want to use the TOCChunker. Please perform the necessary document manipulations to arrive at this. 
   - I want to then run the same RAGAS evaluation as what was done in the baseline
   - Make sure to include explanation markdown sections to explain what is being done

Make sure to:
1. include a "pip install" section in the beginning for all python dependencies   