# Requirements for a LangGraph Based Multi-Agent System to post to social media about a given research paper

## Overall Requirement

We want to build a LangGraph multi-agent system where we take in 3 parameters:
1. The research paper
2. The target social network - Linkedin, X, Facebook 
3. The overall objective of the social media post

And it produces the social media post as a text file (with markdown formatting)

## Overall design

The system will have the following LangGraph agents:
1. A SEARCH_AGENT that will take in the description of the paper and use the Arxiv search tool (from langchain_community) to search for the paper and return the text of that paper
2. A SOCIAL_MEDIA_AGENT that will take in the text of the paper, the target social media network and the objective of the post and creates a social media post
3. A COPY_EDITOR that checks to see if the social media post created by the SOCIAL_MEDIA_AGENT fits the tone of the social network specified and makes changes to the copy of the social media post 
4. A SUPERVISOR agent to coordinate the work between the other 3 agents

### Tools

The agents will need the following tools:
1. A file_read tool that reads the text file and concats the lines in the file into a string to be used
2. A file_write tool that takes in a string and writes it to a file
3. A file_edit tool that inserts lines into the text file

Most of these you can get ideas from the `Multi_Agent_RAG_LangGraph.ipynb` in this workspace

## Other considerations

Please include the necessary configurations to troubleshoot this system via LangSmith 