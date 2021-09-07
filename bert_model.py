import numpy
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers


from haystack.document_store.memory import InMemoryDocumentStore
document_store = InMemoryDocumentStore()

doc_dir = "data/article_txt_got"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

document_store.write_documents(dicts)

from haystack.retriever.sparse import TfidfRetriever
retriever = TfidfRetriever(document_store=document_store)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

from haystack.pipeline import ExtractiveQAPipeline
pipe = ExtractiveQAPipeline(reader, retriever)

def answering(question):
    prediction = pipe.run(query=question, top_k_retriever=10, top_k_reader=5)
    e=print_answers(prediction)
    return e


   


