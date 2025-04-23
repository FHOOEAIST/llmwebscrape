import re

from formatter import Formatter
from soup_statement_generator import SoupStatementGenerator
from page_finder import PageFinder


def main():

    #get_info_from_url("https://www.kununu.com/at", "Get the top Companies, based on the kununu score.",'[{{"CompanyName": <CompanyName>}}]')
    #get_info_from_url("https://www.kununu.com/at", "Get kununu score from the FH Oberösterreich","Company,Score\nInsertCompanyName,InsertScore")
    #get_info_from_url("https://orf.at/", "Get the last 5 articles (title + content) for upper austria",'[{{"Title": <Title>, "Content": <Content>}}]')
    #get_info_from_url("https://orf.at/", "Get the links to news articles",                     '[{{"Link": <Link>}}]')
    #get_info_from_url("https://www.reddit.com", "Gib mir 10 Einträge (Titel + Inhalt) zur FH Oberösterreich.",'[{{"Title": <Title>, "Content": <Content>}}]')
    get_info_from_url("https://fh-ooe.at/", "Gib mir Social Media Accounts der FH Oberösterreich.","Accounts: 'Url1', 'Url2', ...")
    #get_info_from_url("https://www.kununu.com/at", "Get all Companies with their the kununu score.",'[{{"CompanyName": <CompanyName>,"Score": <Score>}}]')


def get_urls(urls: str) -> list:
    return re.findall(r'(?:["\'])([^"\']+)(?:["\'])', urls)

def get_info_from_url(url: str, question: str, target_format:str):
    print("++++++++++++++PageFinder++++++++++++++++++++")
    page_finder = PageFinder()
    page_finder.persist_web_page(url)
    needed_urls = page_finder.run(question)
    print(needed_urls)


    extracted_urls = get_urls(needed_urls)
    print(f"Extracted urls: {extracted_urls}")

    if len(extracted_urls) == 0:
        print(f"No urls found. Are you sure the needed information can be found on the provided website: {url}?")
        return

    print("++++++++++++++++SoupStatementGenerator++++++++++++++++++")
    general_scraper = SoupStatementGenerator()
    general_scraper.persist_multiple_web_page(extracted_urls)
    statement = general_scraper.run(question)
    print(statement)
    content =general_scraper.apply_soup_statement(statement)
    print(content[:1000])
    print("++++++++++++++++Formatter++++++++++++++++++")

    formatter = Formatter()
    code = formatter.run(content,target_format)
    print(code)
    print(formatter.apply_code(code,content)[:1000])

if __name__=="__main__":
    main()