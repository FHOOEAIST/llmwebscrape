import logging
import os
import re

from typing import List, Optional, Tuple, Any

import requests
from bs4 import BeautifulSoup
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent, tool
from langchain.docstore.document import Document
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


class PageFinder:

    def __init__(self) -> None:
        load_dotenv()
        self.url = ""
        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.vs_connection = FAISS.from_documents(
            [Document(page_content="init", metadata={"page": 1})], self.embeddings
        )

    @staticmethod
    def create_llm(model: str = "gpt-4o-mini", **kwargs: Any):
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), model=model, **kwargs
        )

    def configure_agent_statement(self, question: str = "Get the kununu-score") -> Tuple[AgentExecutor, ChatPromptTemplate]:
        """
        Configures an agent.

        :return: The agent used by the PageFinder and the chat_prompt template.
        """
        template: str = f"""Your jop is to find the right subpage of a webpage where the information is needed 
        to answer following question: {question}, using following website: {self.url}. In case the needed information 
        can already be found on the provided url, return the url an look no further. As it is possible that multiple 
        urls are needed, format your answer as a array. For example if the question is to gather all article headlines 
        from https://www.tt.com/ posted the last two days, the answer would be ["https://www.tt.com/newsticker?size=61"]. """
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                system_message_prompt,
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        llm = PageFinder.create_llm(
            temperature=0,
            client=None,
        )
        tools = [
            Tool(
                name="process_thoughts",
                func=process_thoughts,
                description="useful for when you have a thought that you want to use in a task, but you want to "
                            "make sure it's formatted correctly",
            ),
            Tool(
                name="get_subpage",
                func=PageFinder.get_subpage,
                description="useful when you need the content of a subpage, pass the needed url, "
                            "get the page content in return. If the page does not exist a error message is returned.",
            )
        ]
        agent = create_openai_functions_agent(tools=tools, llm=llm, prompt=chat_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=6)
        return agent_executor, chat_prompt

    @staticmethod
    def configure_agent_revise_statement() -> Tuple[AgentExecutor, ChatPromptTemplate]:
        """
        Configures the statement revision agent.

        :return: The agent used by the KununuScraper and the chat_prompt template.
        """
        template: str = """You need to check if following URLs a formatted correctly into a array. 
        A correctly formated answer would be: ['https://url1', 'https://url2']. Add no further information. """
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                system_message_prompt,
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        llm = PageFinder.create_llm(
            temperature=0,
            client=None,
        )
        tools = [
            Tool(
                name="process_thoughts",
                func=process_thoughts,
                description="useful for when you have a thought that you want to use in a task, but you want to "
                            "make sure it's formatted correctly",
            ),
            Tool(
                name="get_subpage",
                func=PageFinder.get_subpage,
                description="useful when you need the content of a subpage, pass the needed url, "
                            "get the page content in return. If the page does not exist a error message is returned.",
            )
        ]
        agent = create_openai_functions_agent(tools=tools, llm=llm, prompt=chat_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=6)
        return agent_executor, chat_prompt

    def run(
            self,
            question: str,
            index_name: Optional[str] = "index",
            namespace: Optional[str] = "",
    ) -> str:
        if not isinstance(question, str) or not question:
            raise ValueError("Argument element must be a non empty string.")
        if not isinstance(index_name, str):
            raise ValueError("Argument index_name must be a string.")
        if not isinstance(namespace, str):
            raise ValueError("Argument namespace must be a string.")
        try:
            statement_revised: str = ""
            agent, _ = self.configure_agent_statement(question)
            human_template = f"""Please find the subpage of the website needed to answer the question: {question}."""

            statement: str = agent.invoke(
                {"messages": [HumanMessage(content=human_template)]}
            )["output"]

            agent_revise, _ = self.configure_agent_revise_statement()
            human_template = (
                f"""This is the website: {statement}. I want it to: {question}."""
            )
            statement_revised = agent_revise.invoke(
                {"messages": [HumanMessage(content=human_template)]}
            )["output"]

        except ValueError as e:
            logging.error(f"Error: {e}")
            raise ValueError(f"Error: {e}") from e

        return statement_revised if statement_revised else statement

    @staticmethod
    def shorten_content_with_rules(content: str):
        content = content.replace('<div>|</div>|<div >|< div >|<div>|< /div>|</div >|< /div >', '')
        content = re.sub(r'{["/\w\.\s\(\),\[\]=>:;+-]*}', '', content)  # remove code between {}
        content = re.sub(r'[\w=\-]*id[\w=\-]*"[^"]*"', '', content)  # remove ids
        content = re.sub(r'[\w\-=]*tracking[\w\-=]*"[^"]*"', '', content)  # remove tracking
        content = re.sub(r'src="[^"]*"', '', content)  # remove file links (images)
        content = re.sub(r"<style.*>[^<]*</style>", '', content)  # remove styling
        content = re.sub(r"<path[^>]*>", '', content)  # remove path svg images
        content = re.sub(r'[^"\s]+\.(jpg|jpeg|png|gif|bmp|svg)', '', content)  # remove images
        content = re.sub(r'height=".+?"|width=".+?"|class=".+?"', '', content)  # remove some properties

        content = re.sub(r'\s\s+', ' ', content)  # remove multiple whitespaces
        return content

    @staticmethod
    def get_page_content(url: str) -> str:
        """
        Crawls a web page and returns the html content of the page.

        :param url: The url of the page to be crawled.
        :return: The html content of the page.
        :raises ValueError: If the url is not a non empty string.
        """
        if not isinstance(url, str) or not url:
            raise ValueError("Argument url must be a non empty string.")
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0"
            }
            with requests.get(url, headers=headers, timeout=10) as response:
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "lxml")
                page_content = soup.select_one("main")
        except requests.RequestException as e:
            logging.error(f"Error: {e}")
            raise ValueError(f"Error: {e}") from e
        return PageFinder.shorten_content_with_rules(str(page_content))


    def persist_web_page(
            self,
            url: str,
            namespace: Optional[str] = "",
            index_name: Optional[str] = "index",
            metric: Optional[str] = "cosine",
            pod_type: Optional[str] = "p1.x1",
    ) -> None:
        """
        Crawls a web page and persists the html content of the page in the vectorstore.

        :param url: The url to be crawled.
        :param namespace: The namespace of the vectorstore.
        :param index_name: The name of the index in the vectorstore.
        :param metric: The metric to be used for the index.
        :param pod_type: The pod type to be used for the index.
        :return: None
        """
        if not isinstance(url, str) or not url:
            raise ValueError("Argument company must be a non empty string.")
        if not isinstance(namespace, str):
            raise TypeError("Argument namespace must be a string.")
        if not isinstance(index_name, str):
            raise TypeError("Argument index_name must be a string.")
        if not isinstance(metric, str):
            raise TypeError("Argument metric must be a string.")
        if not isinstance(pod_type, str):
            raise TypeError("Argument pod_type must be a string.")
        try:

            self.url = url

            # Get the page content
            page_content: str = self.get_page_content(url)

            # Split the page content into chunks
            documents: List[Document] = [
                Document(page_content=page_content, metadata={"url": url})
            ]
            chunk_size: int = 1000
            chunk_overlap: int = 200
            split_documents: List[Document] = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            ).split_documents(documents=documents)

            # Persist the page content in the vectorstore
            self.vs_connection = FAISS.from_documents(split_documents, self.embeddings)
        except (ValueError, TypeError) as e:
            logging.error(f"Error: {e}")
            raise ValueError(f"Error: {e}") from e

    @tool
    def get_subpage(url: str) -> str:
        """
        Crawls a web page and returns the html content of the page. If the URL is not found a error message is returned.
        """
        print(f"Get subpage {url}...")
        try:
            content = PageFinder.get_page_content(url)
            return content
        except ValueError as e:
            return "Error: Url not found"




@tool
def process_thoughts(thought: str) -> str:
    """This is useful for when you have a thought that you want to use in a task,
    but you want to make sure it's formatted correctly.
    Input is your thought and self-critique and output is the processed thought."""
    logging.info("Processing thought...")
    return thought



