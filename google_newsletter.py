import os
import streamlit as st
from langchain.agents import Tool
from langchain.agents import load_tools
from crewai import Agent, Task, Process, Crew
from langchain.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq

# Set up environment variables
os.environ["SERPER_API_KEY"] = "3464cc676dfe52ef67ba5de1e719f6704750e8fdbfe16653c94023b4f92eb0d9"
os.environ["OPENAI_API_KEY"] =" asst_T5mYw2PsjRc3Ij1g3UBnlYGm"
os.environ["GROQ_API_KEY"] = "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"

llm = ChatGroq(
    model="llama3-groq-8b-8192-tool-use-preview",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# Initialize search tool
search = GoogleSerperAPIWrapper()
search_tool = Tool(
    name="Scrape google searches",
    func=search.run,
    description="Useful for when you need the agent to search the internet.",
)

# Load Human Tools
human_tools = load_tools(["human"])

# Define agents
explorer = Agent(
    role="Senior Researcher",
    goal="Find and explore the most exciting projects and companies in the AI and machine learning space in 2024.",
    backstory="""You are an expert strategist in spotting emerging trends and companies in AI, tech, and machine learning. 
    You excel at finding interesting projects on LocalLLama subreddit and turning scraped data into detailed reports with names 
    of the most exciting projects and companies in the AI/ML world. ONLY use scraped data from the internet for the report.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm= llm,
)

writer = Agent(
    role="Senior Technical Writer",
    goal="Write engaging and interesting blog posts about the latest AI projects using simple, layman vocabulary.",
    backstory="""You are an expert writer on technical innovation, especially in AI and machine learning. You know how to write in 
    an engaging, straightforward, and concise manner, presenting complicated technical terms to a general audience in 
    a fun way using layman terms. ONLY use scraped data from the internet for the blog.""",
    verbose=True,
    allow_delegation=True,
    llm= llm,
)

critic = Agent(
    role="Expert Writing Critic",
    goal="Provide feedback and critique blog post drafts, ensuring tone and writing style is compelling, simple, and concise.",
    backstory="""You are an expert at providing feedback to technical writers. You can tell when a blog text isn't concise,
    simple, or engaging enough. You provide helpful feedback to improve any text, ensuring that it remains technical and insightful 
    while using layman terms.""",
    verbose=True,
    allow_delegation=True,
    llm=llm,
)

# Define tasks
task_report = Task(
    description="""Use and summarize scraped data from the internet to make a detailed report on the latest rising projects in AI. Use ONLY 
    scraped data to generate the report. Your final answer MUST be a full analysis report, text only, ignoring any code or anything that 
    isn't text. The report should include 5-10 exciting new AI projects and tools with each bullet point containing 3 sentences about 
    a specific AI company, product, model, or anything you found on the internet.""",
    agent=explorer,
    expected_output="A detailed report with 5-10 bullet points, each containing 3 sentences about a specific AI project or tool.",
    llm=llm,
)

task_blog = Task(
    description="""Write a blog article with text only, including a short but impactful headline and at least 10 paragraphs. The blog should summarize 
    the report on the latest AI tools found on LocalLLama subreddit. The style and tone should be compelling, concise, fun, and technical, 
    but also use layman words for the general public. Name specific new, exciting projects, apps, and companies in the AI world. Use the following markdown format:
    
## [Title of post](link to project)
    - Interesting facts
    - Own thoughts on how it connects to the overall theme of the newsletter
    ## [Title of second post](link to project)
    - Interesting facts
    - Own thoughts on how it connects to the overall theme of the newsletter
    """,
    agent=writer,
    expected_output="A blog article with a headline and at least 10 paragraphs, following the specified markdown format."
)

task_critique = Task(
    description="""The Output MUST have the following markdown format:
    
## [Title of post](link to project)
    - Interesting facts
    - Own thoughts on how it connects to the overall theme of the newsletter
    ## [Title of second post](link to project)
    - Interesting facts
    - Own thoughts on how it connects to the overall theme of the newsletter

    Ensure it follows the format and rewrite if necessary.""",
    agent=critic,
    expected_output="A critique and potential rewrite of the blog article, ensuring it follows the specified markdown format."
)

# Instantiate crew of agents
crew = Crew(
    agents=[explorer, writer, critic],
    tasks=[task_report, task_blog, task_critique],
    verbose=True,  # Set verbose to True or False, not an integer
    process=Process.sequential,
)

# Streamlit UI
st.set_page_config(page_title="AI Project Explorer", page_icon="🤖", layout="wide")
st.title("AI Project Explorer")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about the latest AI projects"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate crew response
    with st.spinner("Crew is working on your request..."):
        result = crew.kickoff()
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(result)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": result})

# Sidebar with information about the AI Project Explorer
st.sidebar.title("About AI Project Explorer")
st.sidebar.info(
    "This AI-powered tool uses a crew of specialized agents to explore and report on "
    "the latest AI projects and developments. It scrapes data from the internet, "
    "particularly the LocalLLama subreddit, to provide up-to-date information on "
    "exciting AI tools and companies."
)

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Welcome to the AI Project Explorer! Ask a question to get started.")
