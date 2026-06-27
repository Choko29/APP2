# teacher_qa.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import DB_PATH, EMBEDDING_MODEL, LLM_MODEL, SEARCH_RESULTS_COUNT

def answer_question(question):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    docs = vectorstore.similarity_search(question, k=SEARCH_RESULTS_COUNT)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    template = """შენ ხარ Teacher Assistant სისტემა. შენი ამოცანაა უპასუხო სტუდენტის შეკითხვას მხოლოდ მოცემული დოკუმენტის მიხედვით.
    წესები:
      - გამოიყენე მხოლოდ კონტექსტში მოცემული ინფორმაცია.
      - თუ პასუხი კონტექსტში არ ჩანს, დაწერე: "ამ დოკუმენტში ამ კითხვაზე პასუხი არ არის მოცემული."
      - უპასუხე ქართულად.
      - პასუხი უნდა იყოს გასაგები, სასწავლო სტილის და კონკრეტული.

    კონტექსტი: {context}
    სტუდენტის შეკითხვა: {question}
    პასუხი: """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | ChatOllama(model=LLM_MODEL, temperature=0.2) | StrOutputParser()
    return chain.invoke({"context": context, "question": question})
