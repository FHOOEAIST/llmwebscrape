import logging
import os


from typing import Tuple, Any

from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent, tool
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


class Formatter:

    def __init__(self) -> None:
        load_dotenv()
        self.content = ""
        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.tools = [
            Tool(
                name="test_code",
                func=test_code,
                description="useful for when you want to test your code, pass your code and content seperated by |||, get te result in return."
                            "If the code throws an error the error message is returned. In case of an error try agin with different code.",
                handle_tool_error=_handle_error,
            )
        ]

    @staticmethod
    def create_llm(model: str = "gpt-4o-mini", **kwargs: Any):
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), model=model, **kwargs
        )

    def configure_agent_statement(self, format: str) -> Tuple[AgentExecutor, ChatPromptTemplate]:

        template: str = f"""As a professional web developer, your task is to write python code that
            can be used to format previously crawled information. The result should follow following format:\n{format}.
            You can use libraries like BeautifulSoup, json or re, but they must be imported before using.
            For testing you can use the test_code tool, only return code that returns the right answer when using this tool."""
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                system_message_prompt,
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        llm = Formatter.create_llm(
            temperature=1,
            client=None,
        )

        agent = create_openai_functions_agent(tools=self.tools, llm=llm, prompt=chat_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=False, max_iterations=6)
        return agent_executor, chat_prompt

    def configure_agent_revise_statement(self) -> Tuple[AgentExecutor, ChatPromptTemplate]:

        template: str = """As a professional web developer, you need to write python code that transforms a given input into a specified format.
        Only Return working code, no explanation. Assume the content variable is already assigned (do not assign the variable). 
        Add no wrapping to the code."""
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                system_message_prompt,
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        llm = Formatter.create_llm(
            temperature=0,
            client=chat_prompt,
        )
        agent = create_openai_functions_agent(
            tools=self.tools,
            llm=llm,
            prompt=chat_prompt,
        )
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=False)
        return agent_executor, chat_prompt

    def run(
            self,
            content: str,
            target_format: str,
    ) -> str:

        try:
            self.content = content
            agent, _ = self.configure_agent_statement(target_format)
            human_template = f"""Please write python one liner, without imports or assignments (it should be usable as lambda) to transform this input stored in a variable called content:\n{content} \n
            Into that format: \n{target_format}.
            Only Return working code, no explanation. Assume the content variable is already assigned. 
            Add no wrapping to the code. Do not put your code into unnecessary parenthesis.
            Test your code before returning."""
            result: str = agent.invoke(
                {"messages": [HumanMessage(content=human_template)]}
            )["output"]
        except ValueError as e:
            logging.error(f"Error: {e}")
            raise ValueError(f"Error: {e}") from e

        return result

    @staticmethod
    def apply_code( code: str, content:str) -> str:
        print(f"Apply {code} to content: {content}")
        try:
            if "lambda" in code:
                code = ":".join(code.split(":")[1:])
            if "\n" in code:
                code = code.split("\n")[0]
            if "content =" in code:
                code = code.replace("content =", "")
            print(code)
            result = eval(code, {"content": content})

            return str(result)
        except Exception as e:
            logging.error(f"Error: {e}")
            return str(e)

@tool
def test_code(code_content ) -> str:
    """This is useful for when you want to test your code. Pass first the code and then the content, seperated by '|||',
    you want to test and get the result in return. If the code throws an error the error message is returned"""
    print("Test code...")
    try:
        code, content = code_content.split("|||")
    except ValueError:
        return "Parameters passed are not valid. Please pass the code and content seperated by |||.\nExample: 'lambda content: content.split(',')|||foo,bar,foo'"
    try:
        return Formatter.apply_code(code, content)
    except Exception:
        # try params the other way around before returning a exception
        try:
            return Formatter.apply_code(content, code)
        except Exception as e:
            return str(e)

def _handle_error(error: Exception) -> str:
    return (
        "The following errors occurred during tool execution:"
        + error.args[0]
        + "Please try another tool. Or try calling it differently"
    )
