from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_experimental.agents.agent_toolkits import create_python_agent, create_csv_agent
from langchain_experimental.agents.agent_toolkits.python.prompt import PREFIX
from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool

def main():
    print("Starting main...")
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # # doesn't seem to be prepending the python code prompt from invoke on its own
    # # prepending manually fixes it
    def invoke_py_agent(prompt):
        python_agent_executor.invoke(f"""{PREFIX}

        {prompt}""")
    
    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="episode_info.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    # csv_agent.invoke("how many columns are there in file episode_info.csv")
    # csv_agent.invoke("print the seasons by ascending number of episodes")

    grand_agent = initialize_agent(tools=[
        Tool(
            name="PythonAgent",
            func=invoke_py_agent,
            description="""useful when you need to transform natural language to a 'python result',
                        where 'python result' is the end result of writing a python script, running it, and returning the result of its execution
                        DO NOT SEND PYTHON CODE TO THIS TOOL, its only suitable input is natural language"""
        ),
        Tool(
            name="CSVAgent",
            func=csv_agent.invoke,
            description="""useful when you need to answer question about episode_info.csv,
                        takes an input as a natural language question and returns the answer after running pandas calculations"""
        )
    ],
    llm=ChatOpenAI(temperature=0, model="gpt-4"),
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True
    )

    grand_agent.invoke("""generate and save in current working directory 15 QRcodes
                       that point to www.udemy.com/course/langchain, 
                       you have qrcode package installed already, so there is no need to pip install""")

if __name__ == "__main__":
    main()
