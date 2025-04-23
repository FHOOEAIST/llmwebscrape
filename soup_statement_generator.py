import logging
import os

from typing import List, Optional, Tuple, Any

import requests

from bs4 import BeautifulSoup
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent, tool
from langchain.chains import RetrievalQA
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


class SoupStatementGenerator:

    def __init__(self) -> None:
        load_dotenv()
        self.url = ""
        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.vs_connection = FAISS.from_documents(
            [Document(page_content="init", metadata={"page": 1})], self.embeddings
        )
        self.tools = [
            Tool(
                name="process_thoughts",
                func=process_thoughts,
                description="useful for when you have a thought that you want to use in a task, but you want to "
                            "make sure it's formatted correctly",
            ),
            Tool(
                name="get_soup_examples",
                func=get_soup_syntax,
                description="useful when you need to crate a soup statement and you need to know the syntax or want to see some examples"
            ),
            Tool(
                name="test_soup_statement",
                func=test_soup,
                description="This is useful for when you want to test your soup statement. Pass first the soup statement and then the content, seperated by '|||',"
                            "you want to test and get the result in return. If the statement throws an error the error message is returned. In case of an error try agin with different soup"
            )
        ]

    @staticmethod
    def create_llm(model: str = "gpt-4o-mini", **kwargs: Any):
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), model=model, **kwargs
        )

    def configure_agent_statement(self,question: str) -> Tuple[AgentExecutor, ChatPromptTemplate]:
        template: str = f"""As a professional web developer, your task is to write a crawler statement in Python that
            can be used to retrieve specific elements. You will receive a description of what to retrieve and the DOM
            representation of the element and its parents. Your final answer should be just the BeautifulSoup statement
            with which one can retrieve the elements in question. Do not add any prefix or suffix to your final answer.
            Also do not add ``` to your answer. Escape special characters like '+' with double backslashes.
            Example: Please find the BeautifulSoup statement to answer following question: {question}.
            For Example: If the question is to get the kununu-score and the DOM element describing the kununu score is: <button class='index__profileMetrics__UCF9e'
            id='ph-kununu-score' type='button'><div class='index__metric__j4df6' data-testid='desktop_kununu_score'>
            <div class='index__score__Ric3z'><span class='index__value__ApL+4 h2 h3-semibold-tablet'>3,7</span>
            <span class='index__stars__nfK6S index__large__9C47L' data-fillcolor="butterscotch' data-score='3.5'>
            </span></div><span class='index__title__Sxu8z h6 h5-tablet'>kununu Score</span><span class='helper-regular
             p-tiny-regular-tablet text-dark-53'>294 Bewertungen</span>
            </div></button>.
            The answer would be: soup.select_one('.index__value_n_ApL\\\\+4.h2.h3-semibold-tablet')
            For testing you can use the test_soup_statement tool, only return a statement that returns the right answer when using this tool."""
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                system_message_prompt,
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        llm = SoupStatementGenerator.create_llm(
            temperature=0,
            client=None,
        )

        agent = create_openai_functions_agent(tools=self.tools, llm=llm, prompt=chat_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=False)
        return agent_executor, chat_prompt


    def configure_agent_revise_statement(self) -> Tuple[AgentExecutor, ChatPromptTemplate]:
        """
        Configures the statement revision agent.

        :return: The agent used by the KununuScraper and the chat_prompt template.
        """
        template: str = """As a professional web developer, your task is to configure a crawler statement in Python
             that can be used to retrieve specific elements. You will receive a statement and a goal which describes how
             you should change the statement. Your final answer should be just the BeautifulSoup selector with which
             one can retrieve the elements in question. Do not add any prefix or suffix to your final answer.
             Also do not add ``` to your answer. Escape special characters like '+' with double backslashes."""
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                system_message_prompt,
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        llm = SoupStatementGenerator.create_llm(
            temperature=0,
            client=chat_prompt,
        )
        agent = create_openai_functions_agent(
            tools=self.tools,
            llm=llm,
            prompt=chat_prompt,
        )
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=False, max_iterations=6)
        return agent_executor, chat_prompt

    def query_web_page(
            self, element: str, index_name: str = "index", namespace: str = ""
    ) -> str:
        """
        Queries a crawled web page for html elements.

        :param element: The element to be retrieved.
        :param index_name: The name of the index in the vectorstore.
        :param namespace: The namespace of the vectorstore.
        :return: The html element that describes the element in question.
        :raises ValueError: If the element is not a non empty string.
        :raise ValueError: If the index_name is not a non empty string.
        :raise ValueError: If the namespace is not a non empty string.
        """
        if not element or not isinstance(element, str):
            raise ValueError("Argument element must be a non empty string.")
        if not isinstance(index_name, str):
            raise ValueError("Argument index_name must be a string.")
        if namespace != "" and not isinstance(namespace, str):
            raise ValueError("Argument namespace must be a string.")
        try:
            qa = RetrievalQA.from_chain_type(
                llm=SoupStatementGenerator.create_llm(
                    temperature=0,
                ),
                chain_type="map_reduce",
                retriever=self.vs_connection.as_retriever()
            )
            result: str = qa.run(
                f"""What part of the HTML contains content related to following question? If unsure return all. {element}?"""
            )
        except ValueError as e:
            logging.error(f"Error: {e}")
            raise
        return result

    def run(
            self,
            question: str,
            goal: Optional[str] = "",
            index_name: Optional[str] = "index",
            namespace: Optional[str] = "",
    ) -> str:
        """
        Runs the BeautifulSoup statement generation pipeline for provided url assuming a natural language description
        of the desired element is provided.

        :param question: The description for the desired element.
        :param goal: The goal of the task (eg retrieve the 2 item in the list, retrieve all items in the list, etc.).
        :param index_name: The name of the index in the vectorstore.
        :param namespace: The namespace of the vectorstore.
        :return: A BeautifulSoup statement that can be used to retrieve the desired elements.
        :raises ValueError: If the element is not a non empty string.
        :raise ValueError: If the index_name is not a string.
        :raise ValueError: If the namespace is not a string.
        """
        if not isinstance(question, str) or not question:
            raise ValueError("Argument element must be a non empty string.")
        if not isinstance(goal, str):
            raise ValueError("Argument goal must be a string.")
        if not isinstance(index_name, str):
            raise ValueError("Argument index_name must be a string.")
        if not isinstance(namespace, str):
            raise ValueError("Argument namespace must be a string.")
        try:
            # Get the BeautifulSoup statement
            statement_revised: str = ""
            dom_element: str = self.get_urls_in_one_html(self.url)

            # Get the BeautifulSoup statement
            agent, _ = self.configure_agent_statement(question)
            human_template = f"""Please find the BeautifulSoup statement for retrieving the Information needed to 
            answer the question: {question}. Based on following HTML input: \n{dom_element}."""
            statement: str = agent.invoke(
                {"messages": [HumanMessage(content=human_template)]}
            )["output"]

            # Revise the BeautifulSoup statement if a goal is provided
            if goal != "":
                agent_revise, _ = self.configure_agent_revise_statement()
                human_template = (
                    f"""This is the statement: {statement}. I want it to: {goal}."""
                )
                statement_revised = agent_revise.invoke(
                    {"messages": [HumanMessage(content=human_template)]}
                )["output"]

        except ValueError as e:
            logging.error(f"Error: {e}")
            raise ValueError(f"Error: {e}") from e

        return statement_revised if statement_revised else statement

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
        return str(page_content)


    def persist_multiple_web_page(
            self,
            urls: List[str],
            namespace: Optional[str] = "",
            index_name: Optional[str] = "index",
            metric: Optional[str] = "cosine",
            pod_type: Optional[str] = "p1.x1",
    ) -> None:
        """
                Crawls a web page and persists the html content of the page in the vectorstore.

                :param urls: The urls to be crawled.
                :param namespace: The namespace of the vectorstore.
                :param index_name: The name of the index in the vectorstore.
                :param metric: The metric to be used for the index.
                :param pod_type: The pod type to be used for the index.
                :return: None
                """
        if not isinstance(urls, list) or not urls:
            raise ValueError("Argument urls must be a list.")
        if not isinstance(namespace, str):
            raise TypeError("Argument namespace must be a string.")
        if not isinstance(index_name, str):
            raise TypeError("Argument index_name must be a string.")
        if not isinstance(metric, str):
            raise TypeError("Argument metric must be a string.")
        if not isinstance(pod_type, str):
            raise TypeError("Argument pod_type must be a string.")
        try:

            self.url = urls

            page_content = self.get_urls_in_one_html(urls)

            # Split the page content into chunks
            documents: List[Document] = [
                Document(page_content=page_content, metadata={"urls": urls})
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

    def get_urls_in_one_html(self, urls):
        # Get the page contents
        page_content: str = "<div>"
        for url in urls:
            page_content += f"<div id=\"{url}\">{self.get_page_content(url)}</div>"
        page_content += "</div>"
        return page_content

    def apply_soup_statement(self, statement: str) -> str:
        """
        Function that retrieves the score of a company on kununu.com with a given BeautifulSoup statement
        :param statement: The BeautifulSoup statement to retrieve the score (string)
        :return: The score of the company (string)
        """
        urls = self.url

        response = self.get_urls_in_one_html(urls)

        soup = BeautifulSoup(
            response, "html.parser"
        )
        if statement.startswith("soup."):
            try:
                safe_dict = {"soup": soup}

                result = eval(
                    statement, {"__builtins__": None}, safe_dict
                )

                return str(result)
            except SyntaxError as syntax_error:
                return f"Syntax-Error: {syntax_error}"
            except NameError as name_error:
                return f"Name Error: {name_error}"
            except AttributeError as attr_error:
                return f"Attribute Error: {attr_error}"
        else:
            return "Invalid statement."

def test_soup_statement(statement: str, content: str) -> str:
    soup = BeautifulSoup(
        content, "html.parser"
    )
    if statement.startswith("soup."):
        safe_dict = {"soup": soup}

        result = eval(
            statement, {"__builtins__": None}, safe_dict
        )
        return str(result)

@tool
def test_soup(soup_content ) -> str:
    """This is useful for when you want to test your soup statement. Pass first the soup statement and then the content, seperated by '|||',
    you want to test and get the result in return. If the statement throws an error the error message is returned"""
    print("Test code...")
    try:
        statement, content = soup_content.split("|||")
    except ValueError:
        return "Parameters passed are not valid. Please pass the code and content seperated by |||.\nExample: 'soup.select_one('h1')|||<h1>foo</h1>'"
    try:
        return test_soup_statement(statement,content)
    except Exception:
        # try params the other way around before returning a exception
        try:
            return test_soup_statement(statement,content)
        except Exception as e:
            return str(e)

@tool
def process_thoughts(thought: str) -> str:
    """This is useful for when you have a thought that you want to use in a task,
    but you want to make sure it's formatted correctly.
    Input is your thought and self-critique and output is the processed thought."""
    logging.info("Processing thought...")
    return thought

@tool
def get_soup_syntax():
    """
    Returns some examples for how to write soup statements.
    """
    soup_syntax = """
        soup.find(“header”)	Find element(s) by tag name
        soup.find(id=”unique_id”)	Find an element by its ID.
        soup.find(‘div’, {‘id’:’unique_id’})	Find a DIV Element element by its ID.
        soup.find_all(class_=”class_name”)	Find all elements with the class name.
        soup.find_all(a, {‘class’:’class_name’})	Find all anchor elements with the CSS class name.
        soup.find_all(string=”text”)	Find all elements elements containing the text.
        soup.find_all(text=”Example”, limit=3)	Find first 3 elements containing the text “Example”
        soup.find_all(“a”)[“href”]	Get the ‘href’ attribute of the ‘anchor’ tags
        soup.find_all(text=re.compile(pattern))	Searches for elements containing text matching the given pattern.
        soup.find_all(attrs={‘attribute’: re.compile(pattern)})	Searches for elements with attribute values matching the pattern.
        soup.select(‘tag:contains(pattern)’)	Uses the :contains pseudo-class to select elements by tag names containing specific text.
        soup.select(‘element’)	Selects all elements with the specified tag name.
        soup.select(‘.class’)	Selects all elements with the specified class.
        soup.select(‘#id’)	Selects the element with the specified ID.
        soup.select(‘element#id’)	Selects elements with a specific tag, ID, or class.
        soup.select(‘element.class1.class2’)	Selects elements with specified multiple classes.
        soup.select(‘element[attribute=”value”]’)	Selects elements with a specified attribute name and value.
        soup.select(“p nth-of-type(3)”)	Selects the third <p> element.
        soup.select(“p > a:nth-of-type(2)”)	Selects the second <a> element that is a direct child of a <p> element.
        soup.select(“#link1 ~ .sister”)	Selects all elements with the class sister that are siblings to the element with ID ‘link1’.
        element.find_next(tag)	Find and return the first occurrence of the tag AFTER the current element.
        element.find_all_next(tag)	Find and return a list of all occurrences of a tag AFTER the current element.
        element.find_previous(tag)	Find and return the first occurrence of the tag BEFORE the current element.
        element.find_all_previous(tag)	Find and return the first occurrence of the tag BEFORE the current element.
        element.find_parent(tag)	Find and return the first occurrence of the tag in the parent elements.
        element.find_all_parents(tag)	Find and return a list of all occurrences of the tag in the parent elements.
    """
    return soup_syntax

