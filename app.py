import os
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

def load_env():
    # Ensure your OpenAI API key is loaded from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    return api_key

def main():
    load_env()

    # Create a prompt template for the chatbot
    template = """You are a helpful chatbot that can answer questions in a conversational style. Given the following input, generate a helpful response:
    
    User: {question}
    
    Chatbot:"""

    # Initialize the LLM chain with the template
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = OpenAI(temperature=0.7)  # You can adjust temperature for creative output

    # Chain the prompt with the LLM
    chain = LLMChain(llm=llm, prompt=prompt)

    # Simple loop for user interaction
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = chain.run(question=user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
