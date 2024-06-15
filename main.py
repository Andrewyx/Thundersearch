# import our MyGPT4ALL class from mode module
# import MyKnowledgeBase class from our knowledgebase module

from model import MyGPT4ALL
from knowledgebase import MyKnowledgeBase
from knowledgebase import (
    DOCUMENT_SOURCE_DIRECTORY
)
# import all the langchain modules
from langchain_huggingface import HuggingFaceEmbeddings
from constants import *
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.runnables import RunnableParallel

llm = MyGPT4ALL(
    model_folder_path=GPT4ALL_MODEL_FOLDER_PATH,
    model_name=GPT4ALL_MODEL_NAME,
    allow_streaming=GPT4ALL_ALLOW_STREAMING,
    allow_download=GPT4ALL_ALLOW_DOWNLOAD
)

embeddings = HuggingFaceEmbeddings()

kb = MyKnowledgeBase(
    pdf_source_folder_path=DOCUMENT_SOURCE_DIRECTORY
)
kb.initiate_document_injetion_pipeline()
# get the retriver object from the vector db 

retriever = kb.return_retriever_from_persistant_vector_db(embeddings)

# Prompt
prompt = PromptTemplate.from_template(
    "Summarize the main themes in these retrieved docs: {docs}"
)


# Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = hub.pull("rlm/rag-prompt")
rag_prompt.messages

qa_chain = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | rag_prompt 
    | llm
    | StrOutputParser()
)


qa_chain_with_docs = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=qa_chain)

def main():
    while True:
        query = input("What's on your mind: ")
        if query == 'exit':
            break
        # result = qa_chain(query)
        # answer, docs = result['result'], result['source_documents']
        result = qa_chain_with_docs.invoke(query)

        answer, docs = result['answer'], result['context']
        print(answer)

        print("#"* 30, "Sources", "#"* 30)
        for document in docs:
            print("\n> SOURCE: " + document.metadata["source"] + ":")
            print(document.page_content)
        print("#"* 30, "Sources", "#"* 30)

if __name__ == '__main__':
    main()