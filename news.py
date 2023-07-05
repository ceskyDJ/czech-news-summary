import requests as requests
import trafilatura
from lxml import etree
from transformers import pipeline


def main() -> None:
    # Download na parse main page with links to most important articles
    response = requests.get("https://www.ceskenoviny.cz/")
    parsed_page = etree.HTML(response.content.decode(response.encoding))

    # Extract link elements from parsed DOM
    links = parsed_page.xpath("//ul/li[contains(@class, 'list-item') and not(contains(@class, 'grid-sizer'))]/a")

    # Summarize found articles
    for link in links:
        # There could be some links, that don't link to articles
        if "/zpravy/" not in link.attrib['href']:
            continue

        # Download article's webpage and extract the main content
        downloaded_page = trafilatura.fetch_url(f"https://www.ceskenoviny.cz/{link.attrib['href']}")
        clean_content = trafilatura.extract(downloaded_page, include_comments=False, include_tables=False)

        # Summarize content of the article
        summarizer = pipeline('summarization', model="facebook/bart-large-cnn")
        summarizer_output = summarizer(clean_content, truncation=True, max_length=300, min_length=100)

        # Send result of summarization to the standard output
        print(f"{summarizer_output[0]['summary_text']}\n")


if __name__ == "__main__":
    main()
