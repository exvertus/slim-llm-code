from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_python_agent
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

    # doesn't seem to be prepending the python code prompt from invoke on its own
    # prepending manually fixes
    python_agent_executor.invoke(f"""{PREFIX}

generate and save in current working directory 15 QRcodes
that point to www.udemy.com/course/langchain
you already have the qrcode package installed in the REPL environment""")

if __name__ == "__main__":
    main()
